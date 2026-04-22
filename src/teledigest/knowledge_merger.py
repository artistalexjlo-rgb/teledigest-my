"""
knowledge_merger.py — Deduplication and merge logic for knowledge base.

Uses word overlap (>60%) to detect duplicate questions within the same
country + category.  Merges answers, bumps confidence, and tracks freshness.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import log
from .knowledge_db import (
    get_all_knowledge_for_category,
    insert_knowledge,
    update_knowledge,
    VALID_CATEGORIES,
)

# Minimum word overlap ratio to consider two questions as duplicates.
SIMILARITY_THRESHOLD = 0.60

# Russian stop-words to exclude from comparison.
_STOP_WORDS = frozenset({
    "и", "в", "на", "с", "по", "не", "что", "как", "это", "для",
    "из", "то", "а", "но", "или", "ли", "бы", "же", "от", "до",
    "за", "при", "об", "его", "её", "их", "мы", "вы", "он", "она",
    "они", "все", "так", "уже", "ещё", "вот", "да", "нет", "где",
    "кто", "когда", "можно", "нужно", "надо", "есть", "тоже", "ещё",
    "очень", "только", "если", "чтобы", "был", "была", "было", "были",
})

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE | re.UNICODE)


def _normalize_words(text: str) -> set[str]:
    """Extract meaningful words from text, lowercased, without stop-words."""
    words = set(_WORD_RE.findall(text.lower()))
    return words - _STOP_WORDS


def _word_overlap(text_a: str, text_b: str) -> float:
    """
    Compute symmetric word overlap ratio between two texts.

    Returns a value in [0.0, 1.0].  1.0 means all words match.
    """
    words_a = _normalize_words(text_a)
    words_b = _normalize_words(text_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    # Symmetric: use the smaller set as denominator
    smaller = min(len(words_a), len(words_b))
    return len(intersection) / smaller


def _compute_confidence(msg_count: int) -> str:
    """Determine confidence level based on total supporting messages."""
    if msg_count >= 6:
        return "high"
    if msg_count >= 3:
        return "high"  # 3-5 without contradictions = high per spec
    return "low" if msg_count <= 2 else "medium"


def merge_qa_pair(
    country: str,
    category: str,
    question: str,
    answer: str,
    source_msgs: list[str],
    confidence: str,
    tags: list[str],
) -> int:
    """
    Merge a new Q&A pair into the knowledge base.

    If a similar question exists (word overlap > 60%), updates the
    existing entry.  Otherwise inserts a new one.

    Returns:
        The knowledge entry ID (new or existing).
    """
    if not question.strip() or not answer.strip():
        return 0

    # Validate category
    if category not in VALID_CATEGORIES:
        log.warning("Invalid category '%s', defaulting to 'other'.", category)
        category = "other"

    # Validate confidence
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"

    # Search existing entries in same country + category
    existing = get_all_knowledge_for_category(country, category)

    best_match: dict[str, Any] | None = None
    best_score = 0.0

    for entry in existing:
        score = _word_overlap(question, entry["question"])
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match and best_score >= SIMILARITY_THRESHOLD:
        # MERGE with existing entry
        kid = best_match["id"]
        existing_msgs = json.loads(best_match["source_msgs"])
        merged_msgs = list(set(existing_msgs + source_msgs))
        new_count = best_match["msg_count"] + len(source_msgs)
        new_confidence = _compute_confidence(new_count)

        # Use newer answer if it's longer / more detailed
        new_answer = answer if len(answer) > len(best_match["answer"]) else best_match["answer"]

        # Merge tags
        existing_tags = json.loads(best_match["tags"])
        merged_tags = list(set(existing_tags + tags))

        update_knowledge(
            kid,
            answer=new_answer,
            source_msgs=json.dumps(merged_msgs, ensure_ascii=False),
            msg_count=new_count,
            confidence=new_confidence,
            tags=json.dumps(merged_tags, ensure_ascii=False),
        )

        log.debug(
            "Merged Q&A into #%d (overlap=%.0f%%): %s",
            kid, best_score * 100, question[:60],
        )
        return kid
    else:
        # INSERT new entry
        kid = insert_knowledge(
            country=country,
            category=category,
            question=question,
            answer=answer,
            source_msgs=source_msgs,
            msg_count=len(source_msgs),
            confidence=confidence,
            tags=tags,
        )
        log.debug("Inserted new Q&A #%d: %s", kid, question[:60])
        return kid
