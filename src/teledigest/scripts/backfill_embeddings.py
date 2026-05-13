"""
backfill_embeddings.py — Вычисляет и записывает эмбеддинги в Firestore.

Сканирует wisdom_base и wikivoyage_base, находит документы без поля
`embedding`, батчами по 100 вызывает text-embedding-004, обновляет
документы. Безопасен для повторного запуска (пропускает уже готовые).

Запуск вручную:
    python -m teledigest.scripts.backfill_embeddings [--collection wisdom_base|wikivoyage_base|all]

Автоматически вызывается из wikivoyage_batch.py после импорта страны.

Quota: text-embedding-004 free tier = 1500 RPM, batch=100 → 15 rps.
При 10K документов = ~100 батчей = <7 секунд (если quota позволяет).
"""

from __future__ import annotations

import argparse
import sys
import time

from ..config import get_config, log
from ..gemini_brain import _build_firestore_client, compute_embeddings_batch

_BATCH_SIZE = 100
_INTER_BATCH_SLEEP = 0.5  # sec — stay well under 1500 RPM


def _backfill_collection(
    db, collection_name: str, instruction_field: str
) -> tuple[int, int]:
    """
    Scan collection for docs without `embedding` field.
    Compute embeddings and write back.

    Returns (processed, skipped) counts.
    """
    processed = 0
    skipped = 0

    # Stream ALL docs — we need to find those without `embedding`.
    # Firestore doesn't support "field does not exist" filter natively,
    # so we fetch everything. For large collections this is fine since
    # we only do this once and then all new docs get embedding on write.
    log.info("backfill_embeddings: scanning %s …", collection_name)

    docs_to_embed: list[tuple[str, str]] = []  # (doc_id, instruction_text)

    try:
        stream = db.collection(collection_name).stream()
        for snap in stream:
            data = snap.to_dict()
            if not data:
                skipped += 1
                continue
            if data.get("embedding"):
                skipped += 1
                continue
            text = (data.get(instruction_field) or "").strip()
            if not text:
                skipped += 1
                continue
            docs_to_embed.append((snap.id, text))
    except Exception as e:
        log.error("backfill_embeddings: scan failed for %s: %s", collection_name, e)
        return 0, 0

    total = len(docs_to_embed)
    log.info(
        "backfill_embeddings: %d docs need embedding in %s", total, collection_name
    )

    # Process in batches of _BATCH_SIZE
    for i in range(0, total, _BATCH_SIZE):
        batch_slice = docs_to_embed[i : i + _BATCH_SIZE]
        ids = [row[0] for row in batch_slice]
        texts = [row[1] for row in batch_slice]

        embeddings = compute_embeddings_batch(texts)

        # Firestore batch write
        batch = db.batch()
        written = 0
        for doc_id, emb in zip(ids, embeddings):
            if emb is None:
                log.warning(
                    "backfill_embeddings: no embedding for doc %s — skip", doc_id
                )
                continue
            ref = db.collection(collection_name).document(doc_id)
            batch.update(ref, {"embedding": emb})
            written += 1

        try:
            batch.commit()
            processed += written
            log.info(
                "backfill_embeddings: committed %d/%d (batch %d)",
                i + written,
                total,
                i // _BATCH_SIZE + 1,
            )
        except Exception as e:
            log.error("backfill_embeddings: batch commit failed: %s", e)

        if i + _BATCH_SIZE < total:
            time.sleep(_INTER_BATCH_SLEEP)

    return processed, skipped


def run_backfill(collections: list[str]) -> None:
    cfg = get_config()
    if not cfg.google.firestore_project_id:
        log.error("backfill_embeddings: firestore_project_id not configured")
        sys.exit(1)

    try:
        db = _build_firestore_client()
    except Exception as e:
        log.error("backfill_embeddings: Firestore init failed: %s", e)
        sys.exit(1)

    collection_map = {
        "wisdom_base": (cfg.google.assistant_collection, "instruction"),
        "wikivoyage_base": ("wikivoyage_base", "instruction"),
    }

    total_processed = 0
    total_skipped = 0

    for coll_key in collections:
        if coll_key not in collection_map:
            log.warning("backfill_embeddings: unknown collection %r — skip", coll_key)
            continue
        coll_name, field = collection_map[coll_key]
        p, s = _backfill_collection(db, coll_name, field)
        total_processed += p
        total_skipped += s

    log.info(
        "backfill_embeddings: DONE — processed=%d, skipped=%d",
        total_processed,
        total_skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Firestore embeddings")
    parser.add_argument(
        "--collection",
        choices=["wisdom_base", "wikivoyage_base", "all"],
        default="all",
        help="Which collection to backfill (default: all)",
    )
    args = parser.parse_args()

    if args.collection == "all":
        collections = ["wisdom_base", "wikivoyage_base"]
    else:
        collections = [args.collection]

    run_backfill(collections)


if __name__ == "__main__":
    main()
