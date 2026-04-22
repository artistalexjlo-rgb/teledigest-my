"""
knowledge_loader.py — Load claims into the knowledge table.

Two modes:
1. Bulk load from Codex's unified_claims.jsonl (backfill)
2. Daily increment from daily_artifact claims (automated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import log
from .knowledge_merger import merge_qa_pair


def load_unified_claims(path: str | Path, country: str = "br") -> dict[str, int]:
    """
    Bulk-load claims from Codex's unified_claims.jsonl into knowledge table.

    Args:
        path: Path to unified_claims.jsonl
        country: Default country code (used if claim has no country field)

    Returns:
        Dict with stats: loaded, skipped, errors
    """
    path = Path(path)
    if not path.exists():
        log.error("Claims file not found: %s", path)
        return {"loaded": 0, "skipped": 0, "errors": 0}

    stats = {"loaded": 0, "skipped": 0, "errors": 0}

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                claim = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Line %d: invalid JSON: %s", line_num, e)
                stats["errors"] += 1
                continue

            answer = (claim.get("answer") or "").strip()
            if not answer or len(answer) < 10:
                stats["skipped"] += 1
                continue

            question = (claim.get("question") or "").strip()
            category = claim.get("category", "other")
            tags = claim.get("tags", [])
            source_msgs = claim.get("source_msgs", [])
            confidence = claim.get("confidence", "medium")
            claim_country = claim.get("country", country)

            try:
                merge_qa_pair(
                    country=claim_country,
                    category=category,
                    question=question,
                    answer=answer,
                    source_msgs=source_msgs,
                    confidence=confidence,
                    tags=tags,
                )
                stats["loaded"] += 1
            except Exception as e:
                log.warning("Line %d: merge failed: %s", line_num, e)
                stats["errors"] += 1

            if line_num % 500 == 0:
                log.info("Loading claims: %d processed, %d loaded...", line_num, stats["loaded"])

    log.info(
        "Bulk load complete: %d loaded, %d skipped, %d errors (from %s)",
        stats["loaded"], stats["skipped"], stats["errors"], path,
    )
    return stats


def load_daily_claims(
    claims: list[dict[str, Any]],
    country: str,
) -> int:
    """
    Load daily artifact claims into knowledge table.

    Args:
        claims: List of claim dicts from artifact_claims_as_dicts()
        country: Country code

    Returns:
        Number of claims loaded.
    """
    loaded = 0
    for claim in claims:
        answer = (claim.get("answer") or "").strip()
        if not answer or len(answer) < 10:
            continue

        question = (claim.get("question") or "").strip()

        try:
            merge_qa_pair(
                country=country,
                category=claim.get("category", "other"),
                question=question,
                answer=answer,
                source_msgs=claim.get("source_msgs", []),
                confidence=claim.get("confidence", "medium"),
                tags=claim.get("tags", []),
            )
            loaded += 1
        except Exception as e:
            log.warning("Daily claim merge failed: %s", e)

    log.info("Daily claims loaded: %d / %d for country=%s", loaded, len(claims), country)
    return loaded
