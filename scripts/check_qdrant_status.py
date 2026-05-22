#!/usr/bin/env python3
"""check_qdrant_status.py — dashboard состояния миграции Firestore → Qdrant.

Показывает counts в обоих хранилищах + gap (сколько ещё надо синкнуть).

Запуск:
    python /app/scripts/check_qdrant_status.py --config /config/teledigest.conf

Через curl-pipe:
    curl -sS https://raw.githubusercontent.com/.../scripts/check_qdrant_status.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def count_firestore(coll, has_embedding_filter: bool = False) -> int:
    """Сосчитать все доки в Firestore-коллекции, или только с embedding."""
    PAGE_SIZE = 500
    total = 0
    with_emb = 0
    last_doc_id = None
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
            if data.get("embedding") is not None:
                with_emb += 1
        last_doc_id = page[-1].id
        if len(page) < PAGE_SIZE:
            break
    return with_emb if has_embedding_filter else total, total - (
        with_emb if has_embedding_filter else 0
    )


def _stats_firestore(coll) -> tuple[int, int]:
    """Возвращает (total, with_embedding)."""
    PAGE_SIZE = 500
    total = 0
    with_emb = 0
    last_doc_id = None
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
            if data.get("embedding") is not None:
                with_emb += 1
        last_doc_id = page[-1].id
        if len(page) < PAGE_SIZE:
            break
    return total, with_emb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    args = parser.parse_args()

    from teledigest.config import get_config, init_config

    init_config(Path(args.config))
    cfg = get_config()

    from teledigest.gemini_brain import _build_firestore_client

    db = _build_firestore_client()

    print("=" * 60)
    print("Firestore status")
    print("=" * 60)
    for label, coll_name in [
        ("wisdom_base", cfg.google.assistant_collection),
        ("wikivoyage_base", "wikivoyage_base"),
    ]:
        coll = db.collection(coll_name)
        total, with_emb = _stats_firestore(coll)
        without_emb = total - with_emb
        print(
            f"  {label}: total={total} with_embedding={with_emb} "
            f"without_embedding={without_emb}"
        )

    print()
    print("=" * 60)
    print("Qdrant status")
    print("=" * 60)
    if not cfg.qdrant.host:
        print("  [qdrant] host не настроен — Qdrant пропущен.")
        return 0

    try:
        from teledigest.qdrant_db import count, get_client

        client = get_client()
        existing = {c.name for c in client.get_collections().collections}
        for label, coll_name in [
            ("wisdom_base", cfg.qdrant.wisdom_collection),
            ("wikivoyage_base", cfg.qdrant.wiki_collection),
        ]:
            if coll_name not in existing:
                print(f"  {label}: коллекция {coll_name} НЕ создана")
                continue
            n = count(coll_name)
            print(f"  {label}: total={n}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    print()
    print("=" * 60)
    print("Sync gap (in Firestore but not in Qdrant — approx)")
    print("=" * 60)
    print(
        "  (Точный gap считается сравнением doc_id, занимает время. "
        "Запусти sync_firestore_to_qdrant.py для актуализации.)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
