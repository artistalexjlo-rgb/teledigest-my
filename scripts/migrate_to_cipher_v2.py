#!/usr/bin/env python3
"""migrate_to_cipher_v2.py — Phase 1.6 of roadmap_embedding_cipher_fix.md.

Walks wisdom_base and wikivoyage_base, recomputes the embedding with the v2
cipher (gemini-embedding-2, 1536 dim, task_type=RETRIEVAL_DOCUMENT, unified
text "Country. Title. Tag. Instruction") and writes the result back in place.

Idempotent: any doc whose embedding_model field already matches the current
v2 tag is skipped.

Pre-requisites (manual):
  1. Run scripts/backup_collections.py once (wisdom_v1_backup,
     wikivoyage_v1_backup must exist).
  2. Drop the old 768-dim vector indexes on wisdom_base.embedding and
     wikivoyage_base.embedding via Firestore console.
  3. Run scripts/create_vector_indexes.py to create new 1536-dim indexes
     (5-30 min build time).

During the index drop → create → migration window find_nearest is broken.
We're in build-mode so the bot's МОЗГ degrading temporarily is acceptable.

Usage inside the container:
    python /app/scripts/migrate_to_cipher_v2.py
    # or piped via stdin from host:
    curl -s .../scripts/migrate_to_cipher_v2.py | docker exec -i $CID python -

Resume-safe: stop with Ctrl-C, re-run, it'll continue from where embedding_model
field is missing or stale.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ISO → English name (must match writers — Apps Script and wikivoyage_import).
COUNTRY_NAMES: dict[str, str] = {
    "th": "Thailand",
    "br": "Brazil",
    "ar": "Argentina",
    "id": "Indonesia",
    "lk": "Sri Lanka",
    "tr": "Turkey",
    "vn": "Vietnam",
    "fr": "France",
    "ph": "Philippines",
    "bg": "Bulgaria",
}


def country_full_name(code: str) -> str:
    c = (code or "").lower()
    return COUNTRY_NAMES.get(c, c.upper())


def build_embed_text(doc: dict) -> str:
    """Unified text: 'Country. Title. Tag. Instruction'."""
    parts = []
    cf = country_full_name(doc.get("country") or "")
    if cf:
        parts.append(cf)
    title = (doc.get("title") or "").strip()
    if title:
        parts.append(title)
    tag = (doc.get("tag") or "").strip()
    if tag:
        parts.append(tag)
    # wisdom_base uses "instruction" key; wikivoyage_base same.
    instruction = (doc.get("instruction") or "").strip()
    if instruction:
        parts.append(instruction)
    return ". ".join(parts)


def migrate_collection(
    db,
    collection_name: str,
    model_tag: str,
    batch_size: int = 50,
    sleep_between_batches: float = 1.0,
) -> tuple[int, int, int]:
    """Returns (migrated, skipped_already_v2, failed)."""
    from google.cloud.firestore_v1.vector import Vector

    from teledigest.gemini_brain import compute_document_embeddings_v2

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
        vectors = compute_document_embeddings_v2(texts)
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
        help="Comma-separated collection names to migrate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Texts per embedding call group"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between batches (rate-limit cushion)",
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
