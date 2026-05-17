#!/usr/bin/env python3
"""migrate_to_cipher_v2.py — DUMB test version.

Один текст за раз. Один ключ. Минута между запросами.
Никакого batch, ротации, cooldown, sweep.

Цель: проверить — проходит ли вообще single-text запрос через
gemini-embedding-2 если не пулеметить.

Использование:
    python /app/scripts/migrate_to_cipher_v2.py --limit 10
    # или через curl-pipe:
    curl -s .../scripts/migrate_to_cipher_v2.py \\
        | docker exec -i $CID python - --config /config/teledigest.conf --limit 10

Параметры:
    --collection wikivoyage_base  какую коллекцию мигрировать
    --key-index 0                 какой ключ из GEMINI_API_KEYS (default 0)
    --gap-seconds 60              сколько секунд между запросами
    --limit 10                    сколько документов обработать максимум
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 1536
EMBEDDING_MODEL_TAG = "gemini-embedding-2-1536"

COUNTRY_NAMES: dict[str, str] = {
    "th": "Thailand", "br": "Brazil", "ar": "Argentina", "id": "Indonesia",
    "lk": "Sri Lanka", "tr": "Turkey", "vn": "Vietnam", "fr": "France",
    "ph": "Philippines", "bg": "Bulgaria",
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
    instruction = (doc.get("instruction") or "").strip()
    if instruction:
        parts.append(instruction)
    return ". ".join(parts)


def embed_one_text(text: str, api_key: str) -> tuple[list[float] | None, str]:
    """Один POST на :embedContent. Возвращает (vector, status_str)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBEDDING_MODEL}:embedContent?key={api_key}"
    )
    payload = {
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": EMBEDDING_DIM,
        "taskType": "RETRIEVAL_DOCUMENT",
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
    except Exception as e:
        return None, f"HTTP_ERROR: {e}"

    if r.status_code != 200:
        return None, f"HTTP_{r.status_code}: {r.text[:300]}"

    data = r.json()
    vec = data.get("embedding", {}).get("values")
    if not vec:
        return None, "EMPTY_VECTOR"
    return vec, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collection",
        default="wikivoyage_base",
        help="Какую коллекцию мигрировать (default wikivoyage_base)",
    )
    parser.add_argument(
        "--key-index",
        type=int,
        default=0,
        help="Какой ключ из GEMINI_API_KEYS использовать (default 0 = первый)",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=60.0,
        help="Сколько секунд ждать между запросами (default 60)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Сколько документов обработать максимум (default 10)",
    )
    args = parser.parse_args()

    from google.cloud.firestore_v1.vector import Vector

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import (
        _build_firestore_client,
        _get_embedding_api_keys,
    )

    keys = _get_embedding_api_keys()
    if not keys:
        print("! no API keys configured")
        return 1
    if args.key_index >= len(keys):
        print(f"! key-index {args.key_index} out of range "
              f"(have {len(keys)} keys, indexes 0..{len(keys) - 1})")
        return 1

    api_key = keys[args.key_index]
    print(f"=== DUMB MIGRATION TEST ===")
    print(f"Collection: {args.collection}")
    print(f"Using key: #{args.key_index} (of {len(keys)} total)")
    print(f"Gap between requests: {args.gap_seconds}s")
    print(f"Limit: {args.limit} documents")
    print()

    db = _build_firestore_client()
    coll = db.collection(args.collection)

    processed = 0
    ok = 0
    fail = 0
    fail_reasons: dict[str, int] = {}

    # Берём с запасом, чтобы пропустить уже мигрированные.
    for snap in coll.limit(args.limit * 10).stream():
        if processed >= args.limit:
            break

        data = snap.to_dict() or {}
        if data.get("embedding_model") == EMBEDDING_MODEL_TAG:
            continue  # уже мигрирован

        text = build_embed_text(data)
        if not text.strip():
            continue

        processed += 1
        text_len = len(text)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{processed}/{args.limit}] doc={snap.id} "
              f"text_len={text_len}")

        t0 = time.time()
        vec, status = embed_one_text(text, api_key)
        dt = time.time() - t0

        if vec is None:
            fail += 1
            reason = status.split(":")[0]
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            print(f"  -> FAIL after {dt:.1f}s: {status[:200]}")
        else:
            ok += 1
            print(f"  -> OK ({dt:.1f}s, dims={len(vec)})")
            payload = {
                **{k: v for k, v in data.items() if k != "embedding"},
                "embedding": Vector(vec),
                "embedding_model": EMBEDDING_MODEL_TAG,
                "embedded_text": text,
                "text_length": len(data.get("instruction") or ""),
            }
            coll.document(snap.id).set(payload)

        if processed < args.limit:
            print(f"  sleeping {args.gap_seconds}s...")
            time.sleep(args.gap_seconds)

    print()
    print(f"=== DONE: ok={ok} fail={fail} of {processed} processed ===")
    if fail_reasons:
        print(f"Fail breakdown: {fail_reasons}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
