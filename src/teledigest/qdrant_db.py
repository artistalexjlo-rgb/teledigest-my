"""qdrant_db.py — обёртка над qdrant-client для Teledigest.

Self-hosted Qdrant заменяет Firestore-vector-search для МОЗГ. Schema
collections — две: wisdom_base (мудрости от Apps Script) и
wikivoyage_base (импорт из вики). Идентификаторы точек = Firestore
doc_id (стабильный SHA1-хэш).

Payload в Qdrant хранит ту же мета что у нас в Firestore-доках:
    country, title, tag, instruction, source, sourceTitle, sourceUrl,
    createdAt (ISO string), importedAt (ISO string), embedding_model,
    embedded_text, text_length, _source (label "База данных" / "WikiVoyage")

Поток данных:
1. Apps Script + wikivoyage_import продолжают писать в Firestore
2. scripts/sync_firestore_to_qdrant.py (cron каждый час) — забирает
   новые/обновлённые доки из Firestore, эмбеддит при необходимости
   через Gemini free-tier, upsert в Qdrant
3. Bot МОЗГ-query — find_nearest в Qdrant, минуя Firestore

Если в config [qdrant] не задан host (default ""), модуль считает что
Qdrant не настроен — caller'ы должны делать fallback на Firestore-путь.
"""

from __future__ import annotations

import datetime as dt
import threading
import uuid
from typing import Any, Optional

from .config import get_config, log

_CLIENT: Optional[Any] = None
_CLIENT_LOCK = threading.Lock()


def is_configured() -> bool:
    """True если в config задан Qdrant host."""
    try:
        return bool(get_config().qdrant.host)
    except Exception:
        return False


def get_client():
    """Singleton qdrant_client.QdrantClient. Lazy-import — модуль
    `qdrant-client` тащим только когда реально надо."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        from qdrant_client import QdrantClient

        cfg = get_config().qdrant
        if not cfg.host:
            raise RuntimeError(
                "qdrant_db.get_client: cfg.qdrant.host is empty — "
                "Qdrant не сконфигурирован"
            )
        _CLIENT = QdrantClient(
            host=cfg.host,
            port=cfg.port,
            api_key=cfg.api_key or None,
            # ВАЖНО: qdrant-client при заданном api_key по умолчанию ставит
            # https=True. Мы ходим на внутренний qdrant:6333 по плоскому HTTP
            # (dokploy-network), поэтому форсим https=False — иначе клиент шлёт
            # TLS в HTTP-порт и всё падает с SSL WRONG_VERSION_NUMBER. api-key
            # уходит в заголовке независимо от транспорта.
            https=False,
            # Short timeout — мы не хотим чтобы МОЗГ-запрос завис на 30+ сек
            # если Qdrant умер. Fallback на Firestore лучше чем повисший бот.
            timeout=10,
        )
        log.info("qdrant_db: connected to %s:%d", cfg.host, cfg.port)
        return _CLIENT


def reset_client() -> None:
    """Для тестов / переконфигурации."""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None


def ensure_collection(name: str, vector_dim: Optional[int] = None) -> None:
    """Создать коллекцию если её нет. Idempotent."""
    from qdrant_client.http import models as qmodels

    if vector_dim is None:
        vector_dim = get_config().qdrant.vector_dim
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    log.info("qdrant_db: created collection %s (dim=%d, COSINE)", name, vector_dim)


def _serialize_payload(data: dict) -> dict:
    """Конвертировать Firestore-doc dict в JSON-сериализуемый payload.
    Datetime → ISO string. Игнорим Vector-поле (embedding) — оно идёт
    отдельным аргументом в upsert."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k == "embedding":
            continue
        if isinstance(v, dt.datetime):
            out[k] = v.isoformat()
        elif hasattr(v, "to_pydatetime"):
            out[k] = v.to_pydatetime().isoformat()
        else:
            out[k] = v
    return out


def upsert_point(
    collection: str,
    point_id: str,
    vector: list[float],
    payload: dict,
) -> None:
    """Положить одну точку (создать или обновить по id)."""
    from qdrant_client.http import models as qmodels

    client = get_client()
    client.upsert(
        collection_name=collection,
        points=[
            qmodels.PointStruct(
                id=_point_id(point_id),
                vector=vector,
                payload=_serialize_payload(payload),
            )
        ],
    )


def upsert_points_batch(
    collection: str,
    items: list[tuple[str, list[float], dict]],
) -> None:
    """Batch upsert. items = list of (doc_id, vector, payload-dict).
    Qdrant batch-upsert эффективен — один HTTP-вызов на N точек."""
    if not items:
        return
    from qdrant_client.http import models as qmodels

    points = [
        qmodels.PointStruct(
            id=_point_id(doc_id),
            vector=vec,
            payload=_serialize_payload(payload),
        )
        for doc_id, vec, payload in items
    ]
    client = get_client()
    client.upsert(collection_name=collection, points=points)


_QDRANT_POINT_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(doc_id: str) -> str:
    """Qdrant принимает либо UUID-строку, либо unsigned int.

    Наши doc_id — 24-символьные sha1-hex prefix'ы (например
    'a742c0e01993385d2c0741ec'), это НЕ валидный UUID и НЕ uint. Qdrant
    на upsert такого ID возвращает 400 ("not a valid point ID").

    Решение: оборачиваем в UUID5 c фиксированным namespace. uuid5
    детерминистичен — тот же doc_id всегда даёт тот же UUID, поэтому
    idempotency на повторных upsert/retrieve не ломается."""
    return str(uuid.uuid5(_QDRANT_POINT_UUID_NAMESPACE, doc_id))


def point_exists(collection: str, doc_id: str) -> bool:
    """Проверка существования точки. Используется sync-скриптом чтобы
    не перезаписывать уже мигрированное."""
    try:
        client = get_client()
        result = client.retrieve(
            collection_name=collection,
            ids=[_point_id(doc_id)],
            with_payload=False,
            with_vectors=False,
        )
        return bool(result)
    except Exception:
        return False


def count(collection: str) -> int:
    """Сколько точек в коллекции. Для status-dashboard."""
    try:
        client = get_client()
        result = client.count(collection_name=collection, exact=True)
        return int(result.count)
    except Exception as e:
        log.warning("qdrant_db.count(%s) failed: %s", collection, e)
        return -1


def find_nearest(
    collection: str,
    query_vector: list[float],
    limit: int,
    source_label: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> list[dict]:
    """Vector search — заменяет Firestore find_nearest.

    Возвращает list[dict] в том же формате что отдавал старый
    `_fetch_by_vector` — payload-словарь + `_source` label.

    score_threshold: если задан, отбрасывает точки с cosine < threshold.
    Помогает МОЗГу не тонуть в нерелевантном шуме на коротких запросах
    (например 'спф' возвращал 150 wisdom-док из которых реально похожих
    меньше десятка, остальные модель пыталась "объяснить" и галлюцинировала).
    """
    try:
        client = get_client()
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        ).points
    except Exception as e:
        log.warning("qdrant find_nearest(%s) failed: %s", collection, e)
        return []

    out = []
    for hit in hits:
        payload = dict(hit.payload or {})
        if source_label is not None:
            payload["_source"] = source_label
        out.append(payload)
    return out
