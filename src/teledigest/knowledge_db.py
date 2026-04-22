"""
knowledge_db.py — Knowledge base tables and CRUD operations.

Adds three tables to the existing SQLite database:
  - sources_meta: tracks backfill progress per source chat
  - knowledge: extracted Q&A pairs
  - extraction_log: tracks which messages have been processed
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

from .config import log
from .db import get_db_connection


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_knowledge_tables() -> None:
    """Create knowledge-base tables if they don't exist yet."""
    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources_meta (
                chat_id       INTEGER PRIMARY KEY,
                chat_name     TEXT NOT NULL,
                country       TEXT NOT NULL,
                language      TEXT DEFAULT 'ru',
                backfill_done INTEGER DEFAULT 0,
                backfill_oldest_id INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                country       TEXT NOT NULL,
                category      TEXT NOT NULL,
                question      TEXT NOT NULL,
                answer        TEXT NOT NULL,
                source_msgs   TEXT NOT NULL,
                msg_count     INTEGER NOT NULL,
                confidence    TEXT DEFAULT 'medium'
                    CHECK(confidence IN ('low','medium','high')),
                first_seen    TEXT NOT NULL,
                last_updated  TEXT NOT NULL,
                is_outdated   INTEGER DEFAULT 0,
                tags          TEXT NOT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS extraction_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id       INTEGER NOT NULL,
                last_processed_msg_id INTEGER NOT NULL,
                messages_processed INTEGER NOT NULL,
                facts_extracted INTEGER NOT NULL,
                run_at        TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_country
            ON knowledge(country)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_category
            ON knowledge(country, category)
        """)

        # Add lemmas column if missing
        try:
            cur.execute("ALTER TABLE knowledge ADD COLUMN lemmas TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass

        log.info("Knowledge base tables initialized.")


# ---------------------------------------------------------------------------
# sources_meta CRUD
# ---------------------------------------------------------------------------

def get_source_meta(chat_id: int) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sources_meta WHERE chat_id = ?", (chat_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def upsert_source_meta(
    chat_id: int,
    chat_name: str,
    country: str,
    language: str = "ru",
    **updates: Any,
) -> None:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sources_meta (chat_id, chat_name, country, language)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                chat_name = excluded.chat_name,
                country   = excluded.country,
                language  = excluded.language
            """,
            (chat_id, chat_name, country, language),
        )
        if updates:
            sets = ", ".join(f"{k} = ?" for k in updates)
            vals = list(updates.values()) + [chat_id]
            cur.execute(
                f"UPDATE sources_meta SET {sets} WHERE chat_id = ?",
                vals,
            )


def update_source_meta(chat_id: int, **fields: Any) -> None:
    if not fields:
        return
    with get_db_connection() as conn:
        cur = conn.cursor()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [chat_id]
        cur.execute(f"UPDATE sources_meta SET {sets} WHERE chat_id = ?", vals)


# ---------------------------------------------------------------------------
# knowledge CRUD
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset({
    "visa", "documents", "finance", "housing", "transport",
    "health", "telecom", "safety", "food", "language",
    "work", "culture", "shopping", "other",
})


def insert_knowledge(
    country: str,
    category: str,
    question: str,
    answer: str,
    source_msgs: list[str],
    msg_count: int,
    confidence: str,
    tags: list[str],
) -> int:
    now = dt.datetime.utcnow().isoformat()
    # Pre-compute lemmas for search
    all_text = f"{question} {answer} {' '.join(tags)}"
    lemmas = _lemmatize(all_text)
    lemma_str = " ".join(sorted(set(lemmas)))

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO knowledge
                (country, category, question, answer, source_msgs,
                 msg_count, confidence, first_seen, last_updated, tags, lemmas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                country,
                category,
                question,
                answer,
                json.dumps(source_msgs, ensure_ascii=False),
                msg_count,
                confidence,
                now,
                now,
                json.dumps(tags, ensure_ascii=False),
                lemma_str,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def update_knowledge(kid: int, **fields: Any) -> None:
    if not fields:
        return
    fields["last_updated"] = dt.datetime.utcnow().isoformat()

    # Recompute lemmas if answer, question, or tags changed
    if any(k in fields for k in ("answer", "question", "tags")):
        q = fields.get("question", "")
        a = fields.get("answer", "")
        t = fields.get("tags", "")
        # If not all parts provided, fetch existing from DB
        if not (q and a):
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT question, answer, tags FROM knowledge WHERE id = ?", (kid,))
                row = cur.fetchone()
                if row:
                    q = q or row[0] or ""
                    a = a or row[1] or ""
                    t = t or row[2] or ""
        all_text = f"{q} {a} {t}"
        lemmas = _lemmatize(all_text)
        fields["lemmas"] = " ".join(sorted(set(lemmas)))

    with get_db_connection() as conn:
        cur = conn.cursor()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [kid]
        cur.execute(f"UPDATE knowledge SET {sets} WHERE id = ?", vals)


import re as _re

from pymorphy3 import MorphAnalyzer as _MorphAnalyzer

_morph = _MorphAnalyzer()
_WORD_RE = _re.compile(r"[а-яёa-z0-9]+", _re.IGNORECASE)

_SEARCH_STOP_WORDS = frozenset({
    "какой", "какая", "какое", "какие", "как", "где", "кто", "что",
    "лучше", "лучший", "хороший", "самый",
    "можно", "нужно", "надо", "есть", "нет", "это", "для", "или",
    "подсказать", "посоветовать", "рассказать", "сказать",
    "пожалуйста", "мочь", "быть",
    "весь", "привет", "здравствуйте", "добрый", "день",
})


def _lemmatize(text: str) -> list[str]:
    """Extract lemmatized words from text, filtering stop-words."""
    words = _WORD_RE.findall(text.lower())
    lemmas = []
    for w in words:
        if len(w) < 2:
            continue
        lemma = _morph.parse(w)[0].normal_form
        if lemma not in _SEARCH_STOP_WORDS:
            lemmas.append(lemma)
    return lemmas


def _build_lemma_index() -> None:
    """Pre-compute lemmas column in knowledge table for fast search."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        # Add column if missing
        try:
            cur.execute("ALTER TABLE knowledge ADD COLUMN lemmas TEXT DEFAULT ''")
            log.info("Added lemmas column to knowledge table.")
        except Exception:
            pass

        # Check how many need indexing
        cur.execute("SELECT COUNT(*) FROM knowledge WHERE lemmas = '' OR lemmas IS NULL")
        need = cur.fetchone()[0]
        if need == 0:
            return

        log.info("Building lemma index for %d knowledge entries...", need)
        cur.execute("SELECT id, question, answer, tags FROM knowledge WHERE lemmas = '' OR lemmas IS NULL")
        rows = cur.fetchall()
        for kid, q, a, t in rows:
            all_text = f"{q or ''} {a or ''} {t or ''}"
            lemmas = _lemmatize(all_text)
            lemma_str = " ".join(sorted(set(lemmas)))
            cur.execute("UPDATE knowledge SET lemmas = ? WHERE id = ?", (lemma_str, kid))

        log.info("Lemma index built for %d entries.", len(rows))


def search_knowledge(
    country: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Search knowledge base using pymorphy3 lemmatization.

    Uses pre-computed lemmas column for fast matching.
    """
    query_lemmas = _lemmatize(query)
    if not query_lemmas:
        return []

    query_lemmas = list(dict.fromkeys(query_lemmas))

    # Search in pre-computed lemmas column
    or_conditions = []
    params: list[Any] = [country]
    for lemma in query_lemmas:
        or_conditions.append("lemmas LIKE ?")
        params.append(f"%{lemma}%")

    where = " OR ".join(or_conditions)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT * FROM knowledge
            WHERE country = ?
              AND is_outdated = 0
              AND ({where})
            """,
            params,
        )
        rows = cur.fetchall()
        if not rows:
            return []
        cols = [d[0] for d in cur.description]
        candidates = [dict(zip(cols, row)) for row in rows]

    # Rank by lemma hits
    conf_weight = {"high": 1.2, "medium": 1.0, "low": 0.7}
    query_set = set(query_lemmas)

    scored = []
    for entry in candidates:
        lemmas_str = entry.get("lemmas") or ""
        entry_lemmas = set(lemmas_str.split())

        # How many query lemmas found
        hits = len(query_set & entry_lemmas)
        if hits == 0:
            continue

        # Density: hits relative to total lemmas
        total = max(len(entry_lemmas), 1)
        density = hits / (total / 10)

        # Also check tags/question specifically for bonus
        q_text = (entry.get("question") or "").lower()
        t_text = (entry.get("tags") or "").lower()
        tag_q_bonus = sum(1 for ql in query_lemmas if ql in t_text or ql in q_text)

        score = tag_q_bonus * 2.0 + density * 1.0

        score *= conf_weight.get(entry.get("confidence", "medium"), 1.0)
        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:limit]]


def get_knowledge_for_country(
    country: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get top knowledge entries for a country (for digest context)."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM knowledge
            WHERE country = ? AND is_outdated = 0
            ORDER BY confidence DESC, last_updated DESC
            LIMIT ?
            """,
            (country, limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]


def get_all_knowledge_for_category(
    country: str,
    category: str,
) -> list[dict[str, Any]]:
    """Get all active knowledge entries for country + category (for dedup)."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM knowledge
            WHERE country = ? AND category = ? AND is_outdated = 0
            ORDER BY last_updated DESC
            """,
            (country, category),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]


def mark_outdated(days: int = 90) -> int:
    """Mark knowledge entries older than `days` with no recent updates."""
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE knowledge
            SET is_outdated = 1
            WHERE is_outdated = 0 AND last_updated < ?
            """,
            (cutoff,),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# extraction_log CRUD
# ---------------------------------------------------------------------------

def get_last_processed_msg_id(chat_id: int) -> int:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT last_processed_msg_id FROM extraction_log
            WHERE chat_id = ?
            ORDER BY run_at DESC LIMIT 1
            """,
            (chat_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def log_extraction_run(
    chat_id: int,
    last_processed_msg_id: int,
    messages_processed: int,
    facts_extracted: int,
) -> None:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO extraction_log
                (chat_id, last_processed_msg_id, messages_processed, facts_extracted)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, last_processed_msg_id, messages_processed, facts_extracted),
        )
