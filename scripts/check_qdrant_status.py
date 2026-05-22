#!/usr/bin/env python3
"""check_qdrant_status.py — dashboard состояния пайплайна Qdrant.

Показывает:
1. SQLite pending-queue (extraction_db.stats): сколько паттернов извлечено,
   сколько ещё ждёт эмбеддинга
2. Qdrant collections: реальное количество точек в wisdom_base /
   telegram_queue / wikivoyage_base

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    args = parser.parse_args()

    from teledigest.config import get_config, init_config

    init_config(Path(args.config))
    cfg = get_config()

    print("=" * 60)
    print("SQLite pending queue (extraction → embed pump)")
    print("=" * 60)
    try:
        from teledigest.extraction_db import init_extraction_tables, stats

        init_extraction_tables()
        st = stats()
        for label, info in st.items():
            print(
                f"  {label}: total={info['total']} "
                f"embedded={info['embedded']} pending={info['pending']}"
            )
            breakdown = info.get("breakdown_by_collection")
            if isinstance(breakdown, dict):
                for coll, b in breakdown.items():
                    print(f"    └─ {coll}: total={b['total']} embedded={b['embedded']}")
    except Exception as e:
        print(f"  ERROR querying SQLite: {e}")

    print()
    print("=" * 60)
    print("Qdrant collections")
    print("=" * 60)
    if not cfg.qdrant.host:
        print("  [qdrant] host НЕ настроен — Qdrant пропущен.")
        print("  Добавь в /config/teledigest.conf:")
        print("    [qdrant]")
        print('    host = "qdrant"')
        print("    port = 6333")
        return 0

    try:
        from teledigest.extraction_db import (
            COLLECTION_STORIES,
            COLLECTION_WIKI,
            COLLECTION_WISDOM,
        )
        from teledigest.qdrant_db import count, get_client

        client = get_client()
        existing = {c.name for c in client.get_collections().collections}
        for label, coll_name in [
            ("wisdom (мухи)", COLLECTION_WISDOM),
            ("stories (котлеты)", COLLECTION_STORIES),
            ("wikivoyage", COLLECTION_WIKI),
        ]:
            if coll_name not in existing:
                print(f"  {label} ({coll_name}): коллекция не создана")
                continue
            n = count(coll_name)
            print(f"  {label} ({coll_name}): total={n}")
    except Exception as e:
        print(f"  ERROR querying Qdrant: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
