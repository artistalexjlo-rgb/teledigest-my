#!/usr/bin/env python3
"""sync_firestore_to_qdrant.py — incremental Firestore → Qdrant sync.

Один и тот же скрипт работает для:
- Initial bootstrap миграции (полный проход по обеим коллекциям)
- Регулярного cron-sync'а (idempotent, пропускает уже-присутствующие)

Стратегия:
1. Для каждой коллекции (wisdom_base, wikivoyage_base):
   - Страница за страницей читаем Firestore (cursor по __name__)
   - Для каждого дока: если он уже в Qdrant (по point_id == doc_id) → skip
   - Если в Firestore-доке есть `embedding` → берём вектор, upsert в Qdrant
   - Если `embedding` нет → эмбеддим через Gemini free-tier
     (compute_document_embeddings_v2), потом upsert в Qdrant
2. Гонит batch'ами — Gemini API эмбеддит до 100 текстов за вызов,
   Qdrant принимает batch upsert.

Защита от Gemini-квоты:
- Если embedding падает (квота на день выжжена) — skip этих доков,
  при следующем запуске cron возьмём их снова. Не теряем уже синкнутое.

Запуск:
    python /app/scripts/sync_firestore_to_qdrant.py --config /config/teledigest.conf

Через curl-pipe:
    curl -sS https://raw.githubusercontent.com/.../scripts/sync_firestore_to_qdrant.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf

Для одной коллекции:
    ... --collections wisdom_base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Размер пачки для эмбеддинга и upsert'а. 100 — лимит Gemini batch
# (фактически возвращает 1 вектор, но мы используем per-text loop
# внутри compute_document_embeddings_v2). Для Qdrant batch — комфортно.
BATCH_SIZE = 50

# Длительность фазы без прогресса до того как печатать строку «жив».
PROGRESS_INTERVAL_S = 10.0


def _doc_to_point_payload(data: dict, source_label: str) -> dict:
    """Build Qdrant payload from Firestore doc data."""
    payload = {k: v for k, v in data.items() if k != "embedding"}
    payload["_source"] = source_label
    return payload


def sync_collection(
    db,
    qd_collection: str,
    firestore_collection: str,
    source_label: str,
    embed_missing: bool,
    page_size: int,
) -> tuple[int, int, int, int]:
    """Returns (copied_with_existing_vec, embedded_and_copied, skipped_existing, failed)."""
    from teledigest.gemini_brain import compute_document_embeddings_v2
    from teledigest.qdrant_db import (
        ensure_collection,
        point_exists,
        upsert_points_batch,
    )

    ensure_collection(qd_collection)
    coll = db.collection(firestore_collection)

    copied = 0
    embedded = 0
    skipped_existing = 0
    failed = 0
    scanned = 0

    pending_upserts: list[tuple[str, list[float], dict]] = []
    pending_to_embed: list[tuple[str, dict]] = []  # (doc_id, payload-dict без embed)

    last_doc_id: str | None = None
    t_start = time.time()
    t_last_print = t_start

    def flush_upserts():
        nonlocal copied
        if not pending_upserts:
            return
        try:
            upsert_points_batch(qd_collection, pending_upserts)
            copied += len(pending_upserts)
        except Exception as e:
            print(
                f"  WARN: qdrant batch upsert failed ({len(pending_upserts)} pts): {e}",
                flush=True,
            )
            nonlocal failed
            failed += len(pending_upserts)
        pending_upserts.clear()

    def flush_to_embed():
        nonlocal embedded, failed
        if not pending_to_embed:
            return
        texts = []
        ids_payloads = []
        for doc_id, data in pending_to_embed:
            text = data.get("embedded_text") or _build_embed_text_fallback(data)
            if not text:
                failed += 1
                continue
            texts.append(text)
            ids_payloads.append((doc_id, data, text))
        if not texts:
            pending_to_embed.clear()
            return
        try:
            vectors = compute_document_embeddings_v2(texts)
        except Exception as e:
            print(f"  WARN: embedding batch failed: {e}", flush=True)
            failed += len(texts)
            pending_to_embed.clear()
            return
        new_upserts: list[tuple[str, list[float], dict]] = []
        for (doc_id, data, text), vec in zip(ids_payloads, vectors):
            if vec is None:
                failed += 1
                continue
            payload = _doc_to_point_payload(data, source_label)
            payload.setdefault("embedded_text", text)
            new_upserts.append((doc_id, vec, payload))
        if new_upserts:
            try:
                upsert_points_batch(qd_collection, new_upserts)
                embedded += len(new_upserts)
            except Exception as e:
                print(
                    f"  WARN: qdrant batch upsert (post-embed) failed "
                    f"({len(new_upserts)} pts): {e}",
                    flush=True,
                )
                failed += len(new_upserts)
        pending_to_embed.clear()

    print(f"\n=== Syncing {firestore_collection} → Qdrant.{qd_collection} ===")
    while True:
        q = coll.order_by("__name__").limit(page_size)
        if last_doc_id is not None:
            q = q.start_after(coll.document(last_doc_id).get())
        page = list(q.stream())
        if not page:
            break
        for snap in page:
            scanned += 1
            doc_id = snap.id
            data = snap.to_dict() or {}
            # Already in Qdrant?
            if point_exists(qd_collection, doc_id):
                skipped_existing += 1
                continue
            emb_field = data.get("embedding")
            if emb_field is not None:
                # Path A: вектор уже есть в Firestore — переносим как есть
                try:
                    vec = _extract_vector(emb_field)
                except Exception as e:
                    print(
                        f"  WARN: vector extract failed for {doc_id}: {e}", flush=True
                    )
                    failed += 1
                    continue
                payload = _doc_to_point_payload(data, source_label)
                pending_upserts.append((doc_id, vec, payload))
                if len(pending_upserts) >= BATCH_SIZE:
                    flush_upserts()
            elif embed_missing:
                # Path B: нет вектора — нужно эмбеддить через Gemini API
                pending_to_embed.append((doc_id, data))
                if len(pending_to_embed) >= BATCH_SIZE:
                    flush_to_embed()
            else:
                # Нет вектора, эмбеддинг отключён — скип
                skipped_existing += 1

        last_doc_id = page[-1].id
        now = time.time()
        if now - t_last_print >= PROGRESS_INTERVAL_S:
            elapsed = now - t_start
            rate = scanned / elapsed if elapsed else 0
            print(
                f"  scanned={scanned} copied={copied} embedded={embedded} "
                f"skipped={skipped_existing} failed={failed} "
                f"({rate:.0f} docs/sec)",
                flush=True,
            )
            t_last_print = now
        if len(page) < page_size:
            break

    flush_upserts()
    flush_to_embed()

    elapsed = time.time() - t_start
    print(
        f"  {firestore_collection} DONE: scanned={scanned} "
        f"copied(with_vec)={copied} embedded={embedded} "
        f"skipped(already-in-qdrant)={skipped_existing} failed={failed} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return copied, embedded, skipped_existing, failed


def _extract_vector(emb_field) -> list[float]:
    """Firestore Vector type unwrap. Vector class из firestore_v1.vector."""
    # Try several known shapes
    try:
        return list(emb_field._value)
    except Exception:
        pass
    try:
        return list(emb_field)
    except Exception:
        pass
    raise ValueError(f"unknown Vector shape: {type(emb_field)}")


def _build_embed_text_fallback(data: dict) -> str:
    """Build embed text from data if embedded_text not present.
    Same format as Apps Script / migrate scripts: 'Country. Title. Tag. Instruction'."""
    from teledigest.country_codes import country_full_name_en

    parts = []
    cf = country_full_name_en(data.get("country") or "")
    if cf:
        parts.append(cf)
    title = (data.get("title") or data.get("sourceTitle") or "").strip()
    if title:
        parts.append(title)
    tag = (data.get("tag") or "").strip()
    if tag:
        parts.append(tag)
    instruction = (data.get("instruction") or "").strip()
    if instruction:
        parts.append(instruction)
    return ". ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collections",
        default="wisdom_base,wikivoyage_base",
        help="Comma-separated firestore collection names to sync.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Firestore pagination page size.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Не эмбеддить doc'и без vector'а. Если включено, скип их. "
        "Полезно для phase-1 sync (только copy-existing).",
    )
    args = parser.parse_args()

    from teledigest.config import get_config, init_config

    init_config(Path(args.config))
    cfg = get_config()

    if not cfg.qdrant.host:
        print(
            "ERROR: cfg.qdrant.host пустой — Qdrant не настроен. "
            "Добавь в /config/teledigest.conf:\n"
            "  [qdrant]\n"
            '  host = "localhost"  # или имя контейнера в Dokploy network'
        )
        return 1

    from teledigest.gemini_brain import _build_firestore_client

    db = _build_firestore_client()
    label_map = {
        cfg.google.assistant_collection: ("wisdom_base", "База данных"),
        "wikivoyage_base": ("wikivoyage_base", "WikiVoyage"),
    }

    totals = [0, 0, 0, 0]
    for fs_coll in [c.strip() for c in args.collections.split(",") if c.strip()]:
        qd_coll, label = label_map.get(fs_coll, (fs_coll, fs_coll))
        c, e, s, f = sync_collection(
            db,
            qd_coll,
            fs_coll,
            label,
            embed_missing=not args.no_embed,
            page_size=args.page_size,
        )
        totals = [totals[i] + v for i, v in enumerate([c, e, s, f])]

    print(
        f"\n=== TOTAL: copied={totals[0]} embedded={totals[1]} "
        f"skipped={totals[2]} failed={totals[3]} ==="
    )
    return 0 if totals[3] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
