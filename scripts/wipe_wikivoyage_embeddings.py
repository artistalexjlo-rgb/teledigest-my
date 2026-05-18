#!/usr/bin/env python3
"""wipe_wikivoyage_embeddings.py — очистка vector-side полей.

Удаляет из wikivoyage_base три поля у каждого документа:
- embedding (сам вектор)
- embedding_model (тег версии)
- embedded_text (текст что шёл в эмбеддер)

Все import-side поля (country, title, instruction, sourceUrl, tag, ...) —
не трогаем.

Use case: после расширения COUNTRY_NAMES + фикса пулемёта, хотим
переэмбеддить всё с нуля без legacy-багов.

ВАЖНО:
- wisdom_base не трогается
- wikivoyage_batch (автодобавление новых статей) должен быть на паузе
  во время операции, иначе будем стирать поля у только что вставленных
- сначала запускаем с --dry-run чтобы посмотреть сколько затронем
- backup делать перед запуском (даже если устарел — структура важнее)

Usage:
    # dry-run (только посчитать)
    python wipe_wikivoyage_embeddings.py --config /config/teledigest.conf --dry-run

    # реальный прогон
    python wipe_wikivoyage_embeddings.py --config /config/teledigest.conf

    # через docker-pipe
    curl -s .../wipe_wikivoyage_embeddings.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

COLLECTION = "wikivoyage_base"
FIELDS_TO_DELETE = ("embedding", "embedding_model", "embedded_text")


def wipe_collection(db, dry_run: bool) -> tuple[int, int, int]:
    """Возвращает (total, would_modify, untouched).

    would_modify = сколько документов имеют хотя бы одно из удаляемых полей.
    untouched = сколько уже чисто (без vector-полей)."""
    from google.cloud import firestore

    coll = db.collection(COLLECTION)
    total = 0
    to_modify = 0
    untouched = 0

    print(f"\n=== Scanning {COLLECTION} ===")
    if dry_run:
        print("(DRY-RUN — ничего не удаляется, только считаем)")

    PAGE_SIZE = 500
    BATCH_WRITE_SIZE = 400  # Firestore batch write limit is 500
    last_doc_id: str | None = None
    pending_updates: list[tuple[str, dict]] = []

    def flush_writes() -> None:
        nonlocal pending_updates
        if not pending_updates or dry_run:
            pending_updates = []
            return
        batch = db.batch()
        for doc_id, update_data in pending_updates:
            batch.update(coll.document(doc_id), update_data)
        batch.commit()
        pending_updates = []

    while True:
        q = coll.order_by("__name__").limit(PAGE_SIZE)
        if last_doc_id is not None:
            q = q.start_after(coll.document(last_doc_id).get())
        page = list(q.stream())
        if not page:
            break

        for snap in page:
            total += 1
            data = snap.to_dict() or {}
            update_data = {}
            for field in FIELDS_TO_DELETE:
                if field in data:
                    update_data[field] = firestore.DELETE_FIELD
            if update_data:
                to_modify += 1
                pending_updates.append((snap.id, update_data))
                if len(pending_updates) >= BATCH_WRITE_SIZE:
                    flush_writes()
            else:
                untouched += 1

        last_doc_id = page[-1].id
        if total % 1000 == 0:
            print(f"  scanned={total} to_modify={to_modify} untouched={untouched}")
        if len(page) < PAGE_SIZE:
            break

    flush_writes()

    return total, to_modify, untouched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только посчитать сколько документов затронули бы",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Не спрашивать подтверждение",
    )
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import _build_firestore_client

    db = _build_firestore_client()

    if not args.dry_run and not args.yes:
        print(f"⚠️  Будут удалены поля {FIELDS_TO_DELETE} из ВСЕХ документов")
        print(f"⚠️  коллекции {COLLECTION}.")
        print("⚠️  Документы сами останутся (country, title, instruction и т.д.).")
        print("⚠️  Это необратимо без backup.")
        confirm = input("\nПродолжить? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Отменено.")
            return 1

    t0 = time.time()
    total, modified, untouched = wipe_collection(db, args.dry_run)
    dt = time.time() - t0

    print()
    print("=" * 50)
    if args.dry_run:
        print(f"DRY-RUN РЕЗУЛЬТАТ за {dt:.1f}s:")
        print(f"  total docs: {total}")
        print(f"  would modify: {modified}")
        print(f"  already clean: {untouched}")
        print("  Запусти без --dry-run чтобы применить.")
    else:
        print(f"ГОТОВО за {dt:.1f}s:")
        print(f"  total docs: {total}")
        print(f"  modified: {modified}")
        print(f"  already clean: {untouched}")
        print("\nТеперь можно запускать миграцию заново —")
        print("все документы будут проэмбеддены с нуля.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
