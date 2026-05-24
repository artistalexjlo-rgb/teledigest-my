"""embed_pump.py — pending-pattern → Gemini embedding → Qdrant.

Отдельный от extraction.py / wikivoyage_import.py — потому что:
- generate_content (extraction) и embed_content — разные модели =
  разные RPD-квоты у Gemini free-tier. Один может работать когда другой
  выжжен.
- Embedding это idempotent retry-friendly операция. Pending-pattern
  можно эмбеддить когда угодно.

Что делает:
1. fetch_pending_extracted(WISDOM, limit) → patterns
2. для каждого batch: build embed_text → compute_document_embeddings_v2
3. upsert в Qdrant wisdom_base
4. mark_embedded → patterns больше не pending
5. То же для STORIES collection
6. fetch_pending_wiki → embed → upsert в wikivoyage_base

Запускается отдельным cron-job-ом в боте каждые N минут.
"""

from __future__ import annotations

from typing import Any

from .config import get_config, log
from .country_codes import country_full_name_en
from .extraction_db import (
    COLLECTION_WIKI,
    COLLECTION_WISDOM,
    fetch_pending_extracted,
    fetch_pending_wiki,
    init_extraction_tables,
    mark_embed_failed,
    mark_embedded,
)


def _build_embed_text(country: str, title: str, tag: str, text_body: str) -> str:
    """Unified format 'Country. Title. Tag. Instruction' — совместимо с
    embed_text который Apps Script и migrate_*.py использовали в Firestore."""
    parts = []
    cf = country_full_name_en(country)
    if cf:
        parts.append(cf)
    if title:
        parts.append(title.strip())
    if tag:
        parts.append(tag.strip())
    if text_body:
        parts.append(text_body.strip())
    return ". ".join(parts)


def _pump_extracted_collection(
    qd_collection: str, source_label: str, batch_size: int = 50
) -> tuple[int, int]:
    """Process pending patterns for one extracted-collection target.
    Returns (embedded, failed)."""
    from .gemini_brain import compute_document_embeddings_v2
    from .qdrant_db import ensure_collection, upsert_points_batch

    ensure_collection(qd_collection)
    embedded = 0
    failed = 0

    while True:
        rows = fetch_pending_extracted(qd_collection, limit=batch_size)
        if not rows:
            break

        texts = []
        ids_payload = []
        for r in rows:
            # Для wisdom embed_text использует ai_lesson; для stories — human_story
            body = r.get("ai_lesson") or r.get("human_story") or ""
            country = r.get("country") or ""
            title = r.get("title") or ""
            tag = r.get("tag") or ""
            text = _build_embed_text(country, title, tag, body)
            if not text:
                mark_embed_failed("extracted_patterns", r["id"], "empty embed text")
                failed += 1
                continue
            texts.append(text)
            ids_payload.append((r, text))

        if not texts:
            continue

        try:
            vectors = compute_document_embeddings_v2(
                texts, min_interval_s=10.0, use_persistent_quota=True
            )
        except Exception as e:
            log.warning("embed_pump extracted batch failed (%s): %s", qd_collection, e)
            for r, _t in ids_payload:
                mark_embed_failed("extracted_patterns", r["id"], str(e)[:200])
                failed += 1
            continue

        upsert_items: list[tuple[str, list[float], dict[str, Any]]] = []
        completed_ids: list[str] = []
        for (r, text), vec in zip(ids_payload, vectors):
            if vec is None:
                mark_embed_failed(
                    "extracted_patterns", r["id"], "embedding returned None"
                )
                failed += 1
                continue
            payload: dict[str, Any] = {
                "country": r.get("country") or "",
                "title": r.get("title") or "",
                "tag": r.get("tag") or "",
                "routing": r.get("routing") or "both",
                "ai_lesson": r.get("ai_lesson") or None,
                "human_story": r.get("human_story") or None,
                "target_languages": r.get("target_languages") or [],
                "source_country_file": r.get("source_country_file"),
                "extracted_at": r.get("extracted_at"),
                "embedded_text": text,
                "_source": source_label,
            }
            upsert_items.append((r["id"], vec, payload))
            completed_ids.append(r["id"])

        if upsert_items:
            try:
                upsert_points_batch(qd_collection, upsert_items)
                mark_embedded("extracted_patterns", completed_ids)
                embedded += len(upsert_items)
                log.info(
                    "embed_pump.extracted[%s]: +%d → Qdrant",
                    qd_collection,
                    len(upsert_items),
                )
            except Exception as e:
                log.warning(
                    "embed_pump extracted Qdrant upsert failed (%s): %s",
                    qd_collection,
                    e,
                )
                for cid in completed_ids:
                    mark_embed_failed(
                        "extracted_patterns", cid, f"qdrant upsert: {e}"[:200]
                    )
                    failed += 1

        # Если batch вернулся меньше чем batch_size, значит pending закончился
        if len(rows) < batch_size:
            break

    return embedded, failed


def _pump_wiki(batch_size: int = 50) -> tuple[int, int]:
    """Process pending wiki patterns. Returns (embedded, failed)."""
    from .gemini_brain import compute_document_embeddings_v2
    from .qdrant_db import ensure_collection, upsert_points_batch

    ensure_collection(COLLECTION_WIKI)
    embedded = 0
    failed = 0

    while True:
        rows = fetch_pending_wiki(limit=batch_size)
        if not rows:
            break

        texts = []
        ids_payload = []
        for r in rows:
            country = r.get("country") or ""
            title = r.get("title") or r.get("source_title") or ""
            tag = r.get("tag") or "Travel"
            instruction = r.get("instruction") or ""
            text = _build_embed_text(country, title, tag, instruction)
            if not text:
                mark_embed_failed("wikivoyage_patterns", r["id"], "empty embed text")
                failed += 1
                continue
            texts.append(text)
            ids_payload.append((r, text))

        if not texts:
            continue

        try:
            vectors = compute_document_embeddings_v2(
                texts, min_interval_s=10.0, use_persistent_quota=True
            )
        except Exception as e:
            log.warning("embed_pump wiki batch failed: %s", e)
            for r, _t in ids_payload:
                mark_embed_failed("wikivoyage_patterns", r["id"], str(e)[:200])
                failed += 1
            continue

        upsert_items: list[tuple[str, list[float], dict[str, Any]]] = []
        completed_ids: list[str] = []
        for (r, text), vec in zip(ids_payload, vectors):
            if vec is None:
                mark_embed_failed(
                    "wikivoyage_patterns", r["id"], "embedding returned None"
                )
                failed += 1
                continue
            payload = {
                "country": r.get("country") or "",
                "title": r.get("title") or "",
                "tag": r.get("tag") or "Travel",
                "instruction": r.get("instruction") or "",
                "sourceTitle": r.get("source_title") or "",
                "sourceUrl": r.get("source_url") or "",
                "importedAt": r.get("imported_at"),
                "embedded_text": text,
                "source": "wikivoyage",
                "_source": "WikiVoyage",
            }
            upsert_items.append((r["id"], vec, payload))
            completed_ids.append(r["id"])

        if upsert_items:
            try:
                upsert_points_batch(COLLECTION_WIKI, upsert_items)
                mark_embedded("wikivoyage_patterns", completed_ids)
                embedded += len(upsert_items)
                log.info("embed_pump.wiki: +%d → Qdrant", len(upsert_items))
            except Exception as e:
                log.warning("embed_pump wiki Qdrant upsert failed: %s", e)
                for cid in completed_ids:
                    mark_embed_failed(
                        "wikivoyage_patterns", cid, f"qdrant upsert: {e}"[:200]
                    )
                    failed += 1

        if len(rows) < batch_size:
            break

    return embedded, failed


def run_embed_pass(batch_size: int = 50) -> dict[str, dict[str, int]]:
    """Один полный проход по всем pending'ам. Возвращает stats per collection.

    Использовать из cron / scheduler. Безопасно вызывать часто — если ничего
    нет, выходит мгновенно. Если Gemini API падает — пропускает, на
    следующем проходе попробует снова (счётчик failures на pattern, 5
    раз — выкидываем из retry queue)."""
    init_extraction_tables()
    cfg = get_config()

    if not cfg.qdrant.host:
        log.warning("embed_pump: cfg.qdrant.host пустой — пропускаем")
        return {}
    if not cfg.gemini.api_key:
        log.warning("embed_pump: cfg.gemini.api_key пустой — Gemini key required")
        return {}

    results: dict[str, dict[str, int]] = {}

    # Только wisdom уходит в Qdrant. Stories (human_story для канала) НЕ
    # векторим — в Firestore-эпохе их тоже не эмбедили, эти записи нужны
    # только postera как pending queue. Они остаются в SQLite с
    # embedded_at=NULL навсегда, что для логики поста не имеет значения.
    for label, qcoll, src_label in [
        ("wisdom", COLLECTION_WISDOM, "База данных"),
    ]:
        emb, fail = _pump_extracted_collection(qcoll, src_label, batch_size)
        results[label] = {"embedded": emb, "failed": fail}

    emb, fail = _pump_wiki(batch_size)
    results["wiki"] = {"embedded": emb, "failed": fail}

    log.info("embed_pump pass complete: %s", results)
    return results
