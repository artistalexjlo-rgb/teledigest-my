#!/usr/bin/env python3
"""migrate_to_cipher_v2_parallel.py — v2 миграции с параллельным эмбеддером.

Отличия от migrate_to_cipher_v2.py:
- Использует embedder_v2_parallel.embed_documents_parallel вместо
  compute_document_embeddings_v2_batch
- Параллельная обработка по всем доступным ключам с per-key TPM-pacing
- Большие batch (по дефолту 200 текстов вместо 100) — выгоднее когда
  параллельные воркеры разгребают сразу

Скрипт лежит отдельно чтобы не ломать старый. Запуск:

    python scripts/migrate_to_cipher_v2_parallel.py --config /config/teledigest.conf

Через curl-pipe:

    curl -s .../scripts/migrate_to_cipher_v2_parallel.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Country English name lookup — single source of truth in country_codes.py.
# Same import as scripts/migrate_to_cipher_v2.py uses.
from teledigest.country_codes import country_full_name_en as country_full_name


def build_embed_text(doc: dict) -> str:
    """Unified text: 'Country. Title. Tag. Instruction'."""
    parts = []
    cf = country_full_name(doc.get("country") or "")
    if cf:
        parts.append(cf)
    title = (doc.get("title") or doc.get("sourceTitle") or "").strip()
    if title:
        parts.append(title)
    tag = (doc.get("tag") or "").strip()
    if tag:
        parts.append(tag)
    instruction = (doc.get("instruction") or "").strip()
    if instruction:
        parts.append(instruction)
    return ". ".join(parts)


def migrate_collection(
    db,
    collection_name: str,
    model_tag: str,
    batch_size: int = 200,
    sleep_between_batches: float = 0.0,
) -> tuple[int, int, int]:
    """Returns (migrated, skipped_already_v2, failed)."""
    from google.cloud.firestore_v1.vector import Vector

    from teledigest.embedder_v2_parallel import embed_documents_parallel

    coll = db.collection(collection_name)
    migrated = 0
    skipped = 0
    failed = 0

    pending: list[tuple[str, dict, str]] = []  # (doc_id, src_data, embed_text)

    def flush_batch() -> None:
        nonlocal migrated, failed
        if not pending:
            return
        texts = [t for _, _, t in pending]
        t0 = time.time()
        vectors = embed_documents_parallel(texts)
        dt = time.time() - t0
        ok_count = sum(1 for v in vectors if v is not None)
        print(
            f"  embedded {ok_count}/{len(texts)} in {dt:.1f}s "
            f"({len(texts) / dt:.1f} docs/sec)"
        )
        for (doc_id, src, text), vec in zip(pending, vectors):
            if vec is None:
                failed += 1
                continue
            instr = src.get("instruction") or ""
            payload = {
                **{k: v for k, v in src.items() if k != "embedding"},
                "embedding": Vector(vec),
                "embedding_model": model_tag,
                "embedded_text": text,
                "text_length": len(instr),
                "needs_chunking": len(instr) > 500,
            }
            coll.document(doc_id).set(payload)
            migrated += 1
        pending.clear()

    print(f"\n=== Migrating {collection_name} ===")
    PAGE_SIZE = 500
    last_doc_id: str | None = None
    while True:
        q = coll.order_by("__name__").limit(PAGE_SIZE)
        if last_doc_id is not None:
            q = q.start_after(coll.document(last_doc_id).get())
        page = list(q.stream())
        if not page:
            break
        for snap in page:
            data = snap.to_dict() or {}
            if data.get("embedding_model") == model_tag:
                skipped += 1
                continue
            pending.append((snap.id, data, build_embed_text(data)))
            if len(pending) >= batch_size:
                flush_batch()
                print(
                    f"  {collection_name}: migrated={migrated} "
                    f"skipped(already-v2)={skipped} failed={failed}"
                )
                if sleep_between_batches > 0:
                    time.sleep(sleep_between_batches)
        last_doc_id = page[-1].id
        if len(page) < PAGE_SIZE:
            break

    flush_batch()
    print(
        f"  {collection_name} DONE: migrated={migrated} "
        f"skipped(already-v2)={skipped} failed={failed}"
    )
    return migrated, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collections",
        default="wisdom_base,wikivoyage_base",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Texts per parallel embed pass (default 200 — workers will "
        "split into smaller HTTP chunks internally)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between Firestore-batches (default 0 — pacing is "
        "handled per-key inside the embedder)",
    )
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import _EMBEDDING_MODEL_TAG_V2, _build_firestore_client

    db = _build_firestore_client()
    total_migrated = total_skipped = total_failed = 0
    for name in [c.strip() for c in args.collections.split(",") if c.strip()]:
        m, s, f = migrate_collection(
            db, name, _EMBEDDING_MODEL_TAG_V2, args.batch_size, args.sleep
        )
        total_migrated += m
        total_skipped += s
        total_failed += f

    print(
        f"\n=== TOTAL: migrated={total_migrated} "
        f"skipped={total_skipped} failed={total_failed} ==="
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
