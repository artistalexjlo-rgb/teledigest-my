#!/usr/bin/env python3
"""pilot_cipher_v2.py — Phase 1.4 of roadmap_embedding_cipher_fix.md.

Migrates a PILOT subset (default: 50 docs likely to contain "car rental"
content for Thailand) from wikivoyage_base → wikivoyage_v2 using the v2
embedder (gemini-embedding-2, 3072 dim, task_type=DOCUMENT, country prefix
in embedded text). Then runs the "rent a car in Thailand" eval query against
the v2 collection and prints what comes back.

Success criteria (per roadmap):
- In top-10 results, at least 5 are real car-rental docs.
- If not → STOP, diagnose, do not proceed to full migration.

Usage inside the container:
    python /app/scripts/pilot_cipher_v2.py
    # or with custom config path:
    python /app/scripts/pilot_cipher_v2.py --config /config/teledigest.conf

The script is idempotent: re-running it overwrites the same target docs
(deterministic doc_id = source doc_id).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Country ISO → English name. Only the ones we actually have in the base.
# Add more as countries get imported; missing codes fall back to uppercase ISO.
_COUNTRY_NAMES: dict[str, str] = {
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
    """ISO code → full English name, fallback to uppercase code."""
    return _COUNTRY_NAMES.get(code, (code or "").upper())


def build_document_text(doc: dict) -> str:
    """The unified text we actually embed.

    Format: "Country. Title. Tag. Instruction."
    Country first so the embedder gets a strong country signal — wiki
    instructions mostly mention cities (Phuket, Bangkok) not the country
    itself. Title and tag give topic priors. Instruction is the raw fact.
    """
    country = country_full_name((doc.get("country") or "").lower())
    title = (doc.get("title") or "").strip()
    tag = (doc.get("tag") or "").strip()
    instruction = (doc.get("instruction") or "").strip()
    parts = [p for p in (country, title, tag, instruction) if p]
    return ". ".join(parts)


# Phrases that almost certainly indicate a rental fact. Same regex used to
# pick the 50 pilot docs as we used earlier in diagnostics.
_RENTAL_RE = re.compile(
    r"\b(car rental|rent a car|car hire|rent.{0,15}(vehicle|motorbike|scooter|bike|auto)"
    r"|аренд\w* (авто|машин|байк|скутер))\b",
    re.IGNORECASE,
)


def _is_real_rental(doc: dict) -> bool:
    text = doc.get("instruction") or ""
    return bool(_RENTAL_RE.search(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--limit", type=int, default=50, help="Pilot doc count (default 50)"
    )
    parser.add_argument(
        "--country", default="th", help="Country ISO for pilot (default th)"
    )
    parser.add_argument(
        "--source-collection", default="wikivoyage_base", help="Read from"
    )
    parser.add_argument("--target-collection", default="wikivoyage_v2", help="Write to")
    parser.add_argument(
        "--eval-query",
        default="rent a car in Thailand",
        help="Query to evaluate retrieval after migration",
    )
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
    from google.cloud.firestore_v1.vector import Vector

    from teledigest.gemini_brain import (
        _EMBEDDING_DIM_V2,
        _EMBEDDING_MODEL_TAG_V2,
        _build_firestore_client,
        compute_document_embeddings_v2,
        compute_query_embedding_v2,
    )

    db = _build_firestore_client()

    print(
        f"=== PILOT v2 ===\n"
        f"  source:      {args.source_collection}\n"
        f"  target:      {args.target_collection}\n"
        f"  country:     {args.country}\n"
        f"  limit:       {args.limit}\n"
        f"  embedder:    {_EMBEDDING_MODEL_TAG_V2}\n"
        f"  eval query:  {args.eval_query!r}\n"
    )

    # --- 1. Pick the pilot set: rental-related docs in the country ---
    print("[1/4] Selecting rental-related docs from source...")
    candidates: list[tuple[str, dict]] = []
    for snap in (
        db.collection(args.source_collection)
        .where("country", "==", args.country)
        .stream()
    ):
        data = snap.to_dict() or {}
        if _is_real_rental(data):
            candidates.append((snap.id, data))
        if len(candidates) >= args.limit:
            break
    print(f"      picked {len(candidates)} docs")
    if not candidates:
        print("FAIL: no candidates matched — check source collection / filters")
        return 1

    # --- 2. Embed them as DOCUMENT and write to target collection ---
    print("[2/4] Embedding pilot docs (task_type=RETRIEVAL_DOCUMENT)...")
    texts = [build_document_text(d) for _, d in candidates]
    vectors = compute_document_embeddings_v2(texts)
    written = 0
    skipped = 0
    for (doc_id, src_data), vec, text in zip(candidates, vectors, texts):
        if vec is None:
            skipped += 1
            continue
        payload = {
            **{k: v for k, v in src_data.items() if k != "embedding"},
            "embedding": Vector(vec),
            "embedding_model": _EMBEDDING_MODEL_TAG_V2,
            "embedded_text": text,
            "needs_chunking": len((src_data.get("instruction") or "")) > 500,
            "text_length": len(src_data.get("instruction") or ""),
            "source_collection": args.source_collection,
        }
        db.collection(args.target_collection).document(doc_id).set(payload)
        written += 1
    print(f"      wrote {written}, skipped (no embedding) {skipped}")

    # --- 3. Eval — embed the query as QUERY and find_nearest ---
    print("[3/4] Embedding eval query (task_type=RETRIEVAL_QUERY)...")
    q_vec = compute_query_embedding_v2(args.eval_query)
    if q_vec is None:
        print("FAIL: query embedding returned None")
        return 1
    if len(q_vec) != _EMBEDDING_DIM_V2:
        print(f"FAIL: query dim {len(q_vec)} != expected {_EMBEDDING_DIM_V2}")
        return 1

    print("[4/4] find_nearest on target collection (top-10)...")
    try:
        results = list(
            db.collection(args.target_collection)
            .find_nearest(
                vector_field="embedding",
                query_vector=Vector(q_vec),
                distance_measure=DistanceMeasure.COSINE,
                limit=10,
            )
            .stream()
        )
    except Exception as e:
        print(f"FAIL: find_nearest error: {e}")
        print("HINT: vector index may not be ready yet, or wrong dimension.")
        print("      Run scripts/create_vector_indexes.py and wait 5-30 min.")
        return 1

    real_hits = 0
    print()
    for i, snap in enumerate(results, 1):
        d = snap.to_dict() or {}
        is_rental = _is_real_rental(d)
        if is_rental:
            real_hits += 1
        mark = "✓" if is_rental else "·"
        title = d.get("title", "(no title)")
        country = d.get("country", "??")
        snippet = (d.get("instruction") or "")[:120].replace("\n", " ")
        print(f"  {mark} [{i:2d}] [{country}] {title}")
        print(f"        {snippet}")

    print()
    print(f"=== RESULT: {real_hits}/10 real rental docs in top-10 ===")
    if real_hits >= 5:
        print("PASS — pilot meets success criteria, proceed to full migration.")
        return 0
    print("FAIL — pilot below threshold (need >=5). Diagnose before proceeding:")
    print("  - Embedding actually using v2 model? (check log lines above)")
    print("  - Index built and dimension matches 3072?")
    print("  - Eval query matches the topic of the pilot docs?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
