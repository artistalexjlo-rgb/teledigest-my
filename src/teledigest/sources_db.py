"""
sources_db.py — Dynamic channel/country management in SQLite.

Replaces static config-based channel lists with DB-driven sources
that can be added/removed via bot commands at runtime.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any

from .config import log
from .db import get_db_connection
from .country_codes import resolve_country as _resolve, display_name, COUNTRIES

# Display names from country_codes.py
COUNTRY_NAMES: dict[str, str] = {code: f"{flag} {name}" for code, (name, flag) in COUNTRIES.items()}


def resolve_country(text: str) -> tuple[str, str] | None:
    """
    Resolve user input to (country_code, display_name).
    Delegates to country_codes.resolve_country().
    """
    result = _resolve(text)
    if result:
        code, name, flag = result
        return code, f"{flag} {name}"
    return None


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

def init_sources_table() -> None:
    """Create sources table if not exists."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT NOT NULL,
                url TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'ru',
                digest_target TEXT DEFAULT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                added_at TEXT NOT NULL,
                UNIQUE(country, url)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sources_country
            ON sources(country)
        """)
        log.info("Sources table initialized.")


def migrate_from_config(channels: list[dict[str, str]],
                        digest_targets: dict[str, str]) -> int:
    """
    One-time migration: copy channels from config into sources table.

    Skips channels that already exist (by url).
    Returns number of new channels added.
    """
    added = 0
    now = dt.datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        cur = conn.cursor()
        for ch in channels:
            url = ch.get("url", "").strip()
            if not url:
                continue
            country = ch.get("country", "br")
            name = ch.get("name", "")
            language = ch.get("language", "ru")

            try:
                cur.execute(
                    "INSERT OR IGNORE INTO sources (country, url, name, language, added_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (country, url, name, language, now),
                )
                if cur.rowcount > 0:
                    added += 1
            except sqlite3.IntegrityError:
                pass

        # Migrate digest targets
        for country_code, target in digest_targets.items():
            cur.execute(
                "UPDATE sources SET digest_target = ? WHERE country = ? AND digest_target IS NULL",
                (target, country_code),
            )

    if added:
        log.info("Migrated %d channels from config to sources table.", added)
    return added


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def add_source(country: str, url: str, name: str = "",
               language: str = "ru") -> int:
    """Add a new source channel. Returns row id or 0 if duplicate."""
    now = dt.datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO sources (country, url, name, language, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (country, url, name, language, now),
            )
            return cur.lastrowid or 0
        except sqlite3.IntegrityError:
            return 0


def remove_source(country: str, url: str) -> bool:
    """Deactivate a source. Returns True if found."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sources SET active = 0 WHERE country = ? AND url = ?",
            (country, url),
        )
        return cur.rowcount > 0


def set_digest_target(country: str, target: str) -> None:
    """Set digest target channel for a country."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sources SET digest_target = ? WHERE country = ?",
            (target, country),
        )


def get_active_sources(country: str | None = None) -> list[dict[str, Any]]:
    """Get active sources, optionally filtered by country."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if country:
            cur.execute(
                "SELECT * FROM sources WHERE active = 1 AND country = ? ORDER BY added_at",
                (country,),
            )
        else:
            cur.execute("SELECT * FROM sources WHERE active = 1 ORDER BY country, added_at")
        return [dict(row) for row in cur.fetchall()]


def get_digest_target(country: str) -> str | None:
    """Get digest target channel for a country."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT digest_target FROM sources WHERE country = ? AND digest_target IS NOT NULL LIMIT 1",
            (country,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def get_active_countries() -> list[str]:
    """Get list of countries that have active sources."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT country FROM sources WHERE active = 1 ORDER BY country")
        return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Channel → country resolution
# ---------------------------------------------------------------------------

def _normalize_url_handle(url: str) -> str | None:
    """
    Extract a comparable handle from a sources.url value.

    Examples:
        '@Brazil_ChatForum'           -> 'Brazil_ChatForum'
        'https://t.me/balichat'       -> 'balichat'
        'https://t.me/+invite_hash'   -> '+invite_hash' (invite links — won't match a channel name)
        'http://t.me/foo/bar'         -> 'foo'
    """
    if not url:
        return None
    s = url.strip()
    if s.startswith("@"):
        return s[1:] or None
    for prefix in ("https://t.me/", "http://t.me/", "tg://resolve?domain="):
        if s.startswith(prefix):
            tail = s[len(prefix):]
            return tail.split("/")[0] or None
    return None


def resolve_country_for_channel(channel: str) -> str | None:
    """
    Resolve a `messages.channel` value to a country code via the `sources` table.

    Matching rules (first hit wins):
        1. sources.name == channel  (covers numeric chat_ids stored in name)
        2. handle extracted from sources.url == channel
           ('@Brazil_ChatForum' -> 'Brazil_ChatForum'; 'https://t.me/balichat' -> 'balichat')

    Args:
        channel: Value as stored in messages.channel — channel username or numeric chat_id.

    Returns:
        Country code (e.g. 'br') or None if no source matches.
    """
    if not channel:
        return None
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT country, url, name FROM sources WHERE active = 1")
        rows = cur.fetchall()
    for country, url, name in rows:
        if name and name == channel:
            return country
        handle = _normalize_url_handle(url or "")
        if handle and handle == channel:
            return country
    return None


def build_channel_country_map() -> dict[str, str]:
    """
    Build a {channel_value -> country_code} map from all active sources.

    Used at startup to populate an in-memory cache. Includes both the `name`
    field and the URL-extracted handle as separate keys for the same country.
    """
    mapping: dict[str, str] = {}
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT country, url, name FROM sources WHERE active = 1")
        for country, url, name in cur.fetchall():
            if name:
                mapping[name] = country
            handle = _normalize_url_handle(url or "")
            if handle:
                mapping[handle] = country
    return mapping
