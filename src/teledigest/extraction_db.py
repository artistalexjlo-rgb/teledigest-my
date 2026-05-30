"""extraction_db.py — SQLite-таблицы для extraction → embed → Qdrant конвейера.

Зачем pending-таблица между extraction и embed/Qdrant:
- extraction.py (Gemini gemini-3.1-flash-lite) и embed_pump.py
  (Gemini embedding) — РАЗНЫЕ модели = разные RPD-квоты у Google.
  Если эмбеддинг сегодня выжжен — extraction продолжает работать,
  накапливает текст. Когда квота восстановится — embed_pump подберёт.
- Idempotency: пересчёт embedding не теряет тексты которые уже
  извлечены.
- Возможность дебага: всегда видно что извлечено и не дошло до Qdrant.

Таблицы:
- `extracted_patterns` — wisdom (ai_lesson) + stories (human_story)
  от extraction.py. Поле `collection_target` указывает в какую Qdrant
  collection это попадёт (`wisdom_base` или `telegram_queue`).
- `wikivoyage_patterns` — wiki-импорт из wikivoyage_import.py. Отдельная
  таблица потому что у wiki другие источники (sourceTitle, sourceUrl)
  и другая частота обновления (ежедневный cron vs realtime extraction).

В обеих таблицах поле `embedded_at IS NULL` = "ещё не в Qdrant".
embed_pump.py выбирает такие, эмбеддит, заливает в Qdrant, проставляет
`embedded_at = now()`.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .config import log
from .db import get_db_connection

# Qdrant collection targets — совпадают с тем что было в Firestore.
COLLECTION_WISDOM = "wisdom_base"
COLLECTION_STORIES = "telegram_queue"  # legacy name, оставляем для совместимости
COLLECTION_WIKI = "wikivoyage_base"


_SCHEMA_EXTRACTED = """
CREATE TABLE IF NOT EXISTS extracted_patterns (
    id TEXT PRIMARY KEY,
    collection_target TEXT NOT NULL,
    country TEXT NOT NULL,
    title TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT 'General',
    routing TEXT NOT NULL DEFAULT 'both',
    ai_lesson TEXT,
    human_story TEXT,
    target_languages TEXT,           -- JSON array of ISO 639-1 codes
    source_country_file TEXT,         -- e.g. '2026-05-20_br_chatforum.txt'
    source_country_file_idx INTEGER,  -- index within file (для deterministic ID)
    extracted_at TEXT NOT NULL,
    embedded_at TEXT,
    embed_failed_count INTEGER NOT NULL DEFAULT 0,
    last_embed_error TEXT
);
"""

_SCHEMA_PATTERN_POSTS = """
CREATE TABLE IF NOT EXISTS pattern_posts (
    pattern_id TEXT NOT NULL,
    channel TEXT NOT NULL,             -- 'luky', 'vk_main', 'discord_en', ...
    posted_at TEXT NOT NULL,
    message_url TEXT,                  -- t.me link or platform-specific URL
    posted_text TEXT,                  -- the exact text that went out (debug)
    PRIMARY KEY (pattern_id, channel)
);
"""

_SCHEMA_QUOTA = """
CREATE TABLE IF NOT EXISTS gemini_quota (
    key_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    date_utc TEXT NOT NULL,        -- 'YYYY-MM-DD' UTC
    count INTEGER NOT NULL DEFAULT 0,
    banned INTEGER NOT NULL DEFAULT 0,  -- 1 = пара забанена до конца суток (например после 429)
    PRIMARY KEY (key_hash, model, date_utc)
);
"""

_SCHEMA_WIKI = """
CREATE TABLE IF NOT EXISTS wikivoyage_patterns (
    id TEXT PRIMARY KEY,
    country TEXT NOT NULL,
    title TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT 'Travel',
    instruction TEXT NOT NULL,
    source_title TEXT NOT NULL,   -- wiki page name e.g. "Bangkok"
    source_url TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    embedded_at TEXT,
    embed_failed_count INTEGER NOT NULL DEFAULT 0,
    last_embed_error TEXT
);
"""

# Индексы для embed_pump-запроса "выбери все без embedded_at, group by collection".
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_extracted_pending "
    "ON extracted_patterns(embedded_at, collection_target) "
    "WHERE embedded_at IS NULL;",
    "CREATE INDEX IF NOT EXISTS idx_wiki_pending "
    "ON wikivoyage_patterns(embedded_at) "
    "WHERE embedded_at IS NULL;",
    # Для status / count по странам.
    "CREATE INDEX IF NOT EXISTS idx_extracted_country "
    "ON extracted_patterns(country, collection_target);",
    "CREATE INDEX IF NOT EXISTS idx_wiki_country " "ON wikivoyage_patterns(country);",
]


def init_extraction_tables() -> None:
    """Создать таблицы и индексы если не существуют. Idempotent."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(_SCHEMA_EXTRACTED)
        cur.execute(_SCHEMA_WIKI)
        cur.execute(_SCHEMA_QUOTA)
        cur.execute(_SCHEMA_PATTERN_POSTS)
        for idx_sql in _INDEXES:
            cur.execute(idx_sql)
        conn.commit()
    log.info(
        "extraction_db: tables ensured "
        "(extracted_patterns + wikivoyage_patterns + gemini_quota)"
    )


# ---------------------------------------------------------------------------
# Gemini quota tracking (per key+model RPD, persistent across container restarts)
# ---------------------------------------------------------------------------


# Gemini free-tier RPD quota resets at midnight America/Los_Angeles (Pacific),
# NOT UTC. The quota "day" we count against must match Google's reset boundary,
# otherwise a long embed pass straddling the boundary makes inconsistent
# decisions (in-memory counter and persistent counter rolling at different
# instants) and we hit false 429s in the UTC↔PT skew window. Both the SQLite
# quota rows and the in-memory _key_rpd_count reset key off this same value.
# (The legacy column name `date_utc` is kept to avoid a schema migration; it
# now stores the Pacific quota-day.)
_QUOTA_TZ = ZoneInfo("America/Los_Angeles")


def _quota_day() -> str:
    return dt.datetime.now(_QUOTA_TZ).strftime("%Y-%m-%d")


# Backwards-compatible alias — some callers/tests still import the old name.
_today_utc = _quota_day


def _key_hash(api_key: str) -> str:
    """Короткий sha1 хэш для записи в gemini_quota (не храним сами ключи)."""
    import hashlib

    return hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:16]


def quota_state(key_hash: str, model: str) -> tuple[int, bool]:
    """Returns (today_count, banned) для пары (key_hash, model) на текущую UTC-дату."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT count, banned FROM gemini_quota "
            "WHERE key_hash = ? AND model = ? AND date_utc = ?",
            (key_hash, model, _today_utc()),
        )
        row = cur.fetchone()
    if not row:
        return 0, False
    return int(row[0] or 0), bool(row[1])


def quota_increment(key_hash: str, model: str) -> int:
    """Increment today's counter, returns new count."""
    today = _today_utc()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO gemini_quota (key_hash, model, date_utc, count) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(key_hash, model, date_utc) "
            "DO UPDATE SET count = count + 1",
            (key_hash, model, today),
        )
        conn.commit()
        cur.execute(
            "SELECT count FROM gemini_quota "
            "WHERE key_hash = ? AND model = ? AND date_utc = ?",
            (key_hash, model, today),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# pattern_posts — per-channel posting log (used by channel_poster.py)
# ---------------------------------------------------------------------------


def fetch_unposted_stories(
    channel: str,
    limit: int = 2000,
    excluded_countries: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return story patterns (collection_target='telegram_queue') that have
    NOT been posted to the given channel yet, oldest-first."""
    excluded = excluded_countries or set()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, country, title, tag, routing, human_story,
                   target_languages, source_country_file, extracted_at
            FROM extracted_patterns
            WHERE collection_target = 'telegram_queue'
              AND human_story IS NOT NULL
              AND human_story != ''
              AND NOT EXISTS (
                  SELECT 1 FROM pattern_posts pp
                  WHERE pp.pattern_id = extracted_patterns.id
                    AND pp.channel = ?
              )
            ORDER BY extracted_at
            LIMIT ?
            """,
            (channel, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [r for r in rows if (r.get("country") or "").lower() not in excluded]


def mark_pattern_posted(
    pattern_id: str,
    channel: str,
    posted_text: str,
    message_url: str | None = None,
) -> None:
    """Insert a row into pattern_posts. Idempotent on (pattern_id, channel)
    via PRIMARY KEY — repeated calls silently noop, which is the right
    behaviour if a post somehow gets retried with the same target."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO pattern_posts
                (pattern_id, channel, posted_at, message_url, posted_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pattern_id, channel, now, message_url, posted_text),
        )
        conn.commit()


def recent_posted_countries(channel: str, n: int = 3) -> list[str]:
    """Return countries of the most recent N posts to the channel,
    oldest-first (so list[-1] is the most recent). Used by channel_poster
    to avoid repeating any of the last N countries when picking the next.

    Persistent across container restarts — rotation memory survives Dokploy
    redeploys (which was the BR-spam bug in the Firestore-era).
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ep.country
            FROM pattern_posts pp
            JOIN extracted_patterns ep ON ep.id = pp.pattern_id
            WHERE pp.channel = ?
            ORDER BY pp.posted_at DESC
            LIMIT ?
            """,
            (channel, n),
        )
        rows = [(r[0] or "").lower() for r in cur.fetchall()]
    return list(reversed(rows))


def quota_ban_today(key_hash: str, model: str) -> None:
    """Пометить пару (key, model) как забаненную до конца UTC-суток
    (например после 429)."""
    today = _today_utc()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO gemini_quota (key_hash, model, date_utc, count, banned) "
            "VALUES (?, ?, ?, 0, 1) "
            "ON CONFLICT(key_hash, model, date_utc) "
            "DO UPDATE SET banned = 1",
            (key_hash, model, today),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Inserts (idempotent — REPLACE on duplicate id)
# ---------------------------------------------------------------------------


def insert_extracted_pattern(
    id_: str,
    collection_target: str,
    country: str,
    title: str,
    tag: str,
    routing: str,
    ai_lesson: Optional[str],
    human_story: Optional[str],
    target_languages: Optional[list[str]],
    source_country_file: Optional[str],
    source_country_file_idx: Optional[int],
) -> None:
    """Insert one pattern from extraction.py.

    INSERT OR IGNORE: если pattern с таким id уже есть (одинаковый сэмпл-файл
    + index) — не перетираем. Extraction идемпотентна по deterministic id.
    """
    target_langs_json = json.dumps(target_languages) if target_languages else None
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO extracted_patterns
                (id, collection_target, country, title, tag, routing,
                 ai_lesson, human_story, target_languages,
                 source_country_file, source_country_file_idx,
                 extracted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                id_,
                collection_target,
                country,
                title,
                tag,
                routing,
                ai_lesson,
                human_story,
                target_langs_json,
                source_country_file,
                source_country_file_idx,
                now,
            ),
        )
        conn.commit()


def insert_wiki_pattern(
    id_: str,
    country: str,
    title: str,
    tag: str,
    instruction: str,
    source_title: str,
    source_url: str,
) -> bool:
    """Returns True if inserted, False if id already existed."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO wikivoyage_patterns
                (id, country, title, tag, instruction,
                 source_title, source_url, imported_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (id_, country, title, tag, instruction, source_title, source_url, now),
        )
        inserted = cur.rowcount > 0
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Pending queue accessors (для embed_pump.py)
# ---------------------------------------------------------------------------


def fetch_pending_extracted(
    collection_target: str, limit: int = 100
) -> list[dict[str, Any]]:
    """SELECT pending patterns для конкретной Qdrant collection."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, collection_target, country, title, tag, routing,
                   ai_lesson, human_story, target_languages,
                   source_country_file, extracted_at
            FROM extracted_patterns
            WHERE embedded_at IS NULL
              AND collection_target = ?
              AND embed_failed_count < 5
            ORDER BY extracted_at
            LIMIT ?
            """,
            (collection_target, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    # Deserialize target_languages JSON
    for r in rows:
        if r.get("target_languages"):
            try:
                r["target_languages"] = json.loads(r["target_languages"])
            except (json.JSONDecodeError, TypeError):
                r["target_languages"] = []
        else:
            r["target_languages"] = []
    return rows


def fetch_pending_wiki(limit: int = 100) -> list[dict[str, Any]]:
    """SELECT pending wiki patterns."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, country, title, tag, instruction,
                   source_title, source_url, imported_at
            FROM wikivoyage_patterns
            WHERE embedded_at IS NULL
              AND embed_failed_count < 5
            ORDER BY imported_at
            LIMIT ?
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def mark_embedded(table: str, ids: list[str]) -> None:
    """UPDATE embedded_at = now WHERE id IN (...). table = 'extracted_patterns'
    или 'wikivoyage_patterns'."""
    if not ids:
        return
    if table not in ("extracted_patterns", "wikivoyage_patterns"):
        raise ValueError(f"unknown table {table!r}")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    placeholders = ",".join(["?"] * len(ids))
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {table} SET embedded_at = ? WHERE id IN ({placeholders})",
            (now, *ids),
        )
        conn.commit()


def mark_embed_failed(table: str, id_: str, error: str) -> None:
    """Increment embed_failed_count + last_embed_error. После 5 фейлов
    запись больше не попадает в fetch_pending."""
    if table not in ("extracted_patterns", "wikivoyage_patterns"):
        raise ValueError(f"unknown table {table!r}")
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {table} "
            "SET embed_failed_count = embed_failed_count + 1, "
            "last_embed_error = ? "
            "WHERE id = ?",
            (error[:500], id_),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Stats (для check_qdrant_status.py)
# ---------------------------------------------------------------------------


def stats() -> dict[str, dict[str, int]]:
    """Возвращает counts: какие коллекции pending / embedded."""
    out: dict[str, dict[str, int]] = {}
    with get_db_connection() as conn:
        cur = conn.cursor()
        for label, table, group_col in [
            ("extracted (wisdom+stories)", "extracted_patterns", "collection_target"),
            ("wiki", "wikivoyage_patterns", None),
        ]:
            cur.execute(f"SELECT COUNT(*), SUM(embedded_at IS NOT NULL) FROM {table}")
            total, embedded = cur.fetchone()
            out[label] = {
                "total": int(total or 0),
                "embedded": int(embedded or 0),
                "pending": int((total or 0) - (embedded or 0)),
            }
            # Breakdown by collection_target / country
            if group_col:
                cur.execute(
                    f"SELECT {group_col}, COUNT(*), SUM(embedded_at IS NOT NULL) "
                    f"FROM {table} GROUP BY {group_col}"
                )
                breakdown = {}
                for row in cur.fetchall():
                    breakdown[row[0]] = {
                        "total": int(row[1] or 0),
                        "embedded": int(row[2] or 0),
                    }
                out[label]["breakdown_by_collection"] = breakdown  # type: ignore[assignment]
    return out
