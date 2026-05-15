#!/usr/bin/env python3
"""backup_collections.py — pre-migration backup of wisdom_base and
wikivoyage_base into _v1_backup collections.

Runs ONCE before Phase 1.6 of roadmap_embedding_cipher_fix.md so that if the
re-embed migration goes sideways we still have the pre-cipher-fix state.

Usage inside the container:
    python /app/scripts/backup_collections.py
    # or via stdin:
    curl -s .../scripts/backup_collections.py | docker exec -i $CID python -

Idempotent: re-running overwrites the backup target docs (deterministic ids).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def copy_collection(db, src_name: str, dst_name: str, page_size: int = 500) -> int:
    """Paginated copy — large collections (wikivoyage_base has ~9K+ docs)
    time out if streamed in a single query. Walk via order_by(__name__) +
    start_after cursor, page_size docs at a time."""
    src = db.collection(src_name)
    dst = db.collection(dst_name)
    n = 0
    last_doc_id: str | None = None
    BATCH_LIMIT = 400  # Firestore batch hard limit is 500 ops
    while True:
        q = src.order_by("__name__").limit(page_size)
        if last_doc_id is not None:
            # start_after expects a document snapshot or matching field values;
            # for __name__ we pass the doc reference.
            q = q.start_after(src.document(last_doc_id).get())
        page = list(q.stream())
        if not page:
            break
        batch = db.batch()
        batch_n = 0
        for snap in page:
            batch.set(dst.document(snap.id), snap.to_dict() or {})
            batch_n += 1
            n += 1
            if batch_n >= BATCH_LIMIT:
                batch.commit()
                batch = db.batch()
                batch_n = 0
        if batch_n:
            batch.commit()
        last_doc_id = page[-1].id
        print(f"  {src_name} → {dst_name}: {n} copied so far...")
        if len(page) < page_size:
            break
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import _build_firestore_client

    db = _build_firestore_client()

    pairs = [
        ("wisdom_base", "wisdom_v1_backup"),
        ("wikivoyage_base", "wikivoyage_v1_backup"),
    ]
    for src, dst in pairs:
        print(f"\n=== Backing up {src} → {dst} ===")
        n = copy_collection(db, src, dst)
        print(f"  Done: {n} docs copied to {dst}")
    print("\nAll backups complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
