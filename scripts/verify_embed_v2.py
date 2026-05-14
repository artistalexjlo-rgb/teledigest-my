#!/usr/bin/env python3
"""verify_embed_v2.py — sanity-check the v2 embedder against the live Gemini API.

Run this once after deploying the v2 functions to confirm the cipher fix
actually works as claimed:

1. task_type=RETRIEVAL_QUERY and task_type=RETRIEVAL_DOCUMENT produce
   DIFFERENT vectors for the same text (proves task_type is honored).
2. Vector length matches the requested dimensionality.
3. Cosine similarity between query/doc vectors of related text is high.
4. Cosine similarity between query/doc vectors of unrelated text is lower.

Costs: ~6 embedding calls per run. Negligible.

Usage:
    docker exec <container> python -m scripts.verify_embed_v2
    # or locally with config:
    python scripts/verify_embed_v2.py --config /path/to/teledigest.conf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import (
        _EMBEDDING_DIM_V2,
        compute_document_embeddings_v2,
        compute_query_embedding_v2,
    )

    text_query = "rent a car in Thailand"
    text_related_doc = (
        "Thailand. Khao Lak Rentals. Transportation. "
        "Car rental available, daily 09:00-18:00, price from 1200 baht/day. "
        "International driving permit required."
    )
    text_unrelated_doc = (
        "Thailand. Khao Lak Coffee Shop. Food. "
        "Best Italian espresso on the beach, daily 07:00-22:00."
    )

    print(f"Testing v2 embedder (model=gemini-embedding-2, dim={_EMBEDDING_DIM_V2})\n")

    q_vec = compute_query_embedding_v2(text_query)
    d_related, d_unrelated = compute_document_embeddings_v2(
        [text_related_doc, text_unrelated_doc]
    )

    # Also embed the SAME text as both QUERY and DOCUMENT to see if task_type matters.
    q_of_doc = compute_query_embedding_v2(text_related_doc)

    if not (q_vec and d_related and d_unrelated and q_of_doc):
        print("FAIL: some embeddings returned None — see warnings above.")
        return 1

    print(f"Query vec len:      {len(q_vec)}")
    print(f"Doc-related len:    {len(d_related)}")
    print(f"Doc-unrelated len:  {len(d_unrelated)}")
    assert len(q_vec) == _EMBEDDING_DIM_V2, "Query dim mismatch"

    sim_q_related = cosine(q_vec, d_related)
    sim_q_unrelated = cosine(q_vec, d_unrelated)
    sim_task_type = cosine(q_of_doc, d_related)  # same text, different task_type

    print()
    print(f"cos('{text_query}' QUERY vs related DOC):    {sim_q_related:.4f}")
    print(f"cos('{text_query}' QUERY vs unrelated DOC):  {sim_q_unrelated:.4f}")
    print(f"cos(same text as QUERY vs as DOCUMENT):      {sim_task_type:.4f}")

    ok_related_beats_unrelated = sim_q_related > sim_q_unrelated
    ok_task_type_differs = sim_task_type < 0.999  # not exact match → task_type matters

    print()
    print(
        "PASS"
        if ok_related_beats_unrelated
        else "FAIL"
        + f" — related-doc cos ({sim_q_related:.4f}) should be > unrelated-doc cos "
        + f"({sim_q_unrelated:.4f})"
    )
    print(
        "PASS"
        if ok_task_type_differs
        else "FAIL"
        + f" — same text with different task_type should differ (got {sim_task_type:.4f}, expected <0.999)"
    )

    return 0 if (ok_related_beats_unrelated and ok_task_type_differs) else 1


if __name__ == "__main__":
    sys.exit(main())
