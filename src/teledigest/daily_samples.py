"""
daily_samples.py — Plain-text daily chat samples for human/LLM review.

After the daily digest pipeline finishes, dump the previous day's raw
messages (sanitized text only — see text_sanitize) for a small set of
country/channel pairs into files alongside the SQLite DB. No bot
commands, no LLM, just `SELECT ... ORDER BY date`.

File layout:
    {db_dir}/samples/{country}/{YYYY-MM-DD}.txt

One line per message:
    [HH:MM] u/<sender_id>: text
    [HH:MM] u/<sender_id> ← reply <orig_msg_id>: text

Sender IDs are kept as numeric Telegram user IDs (no usernames, no
display names) — same impersonal contract as the rest of the pipeline.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import get_config, log
from .db import get_db_connection


# ---------------------------------------------------------------------------
# Configuration: which (country, channel) pairs to dump
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleTarget:
    """One country/channel pair to dump as a daily sample.

    `channel` must match the value stored in `messages.channel` (handle for
    public channels, numeric chat_id as string for invite-link channels).
    """
    country: str
    channel: str


# Picked one source per country to keep the sample stream readable
# (avoids mixing two channels of the same country in one file).
DEFAULT_SAMPLE_TARGETS: tuple[SampleTarget, ...] = (
    SampleTarget(country="br", channel="Brazil_ChatForum"),
    SampleTarget(country="id", channel="balichat"),
    SampleTarget(country="lk", channel="-1001605996131"),
)


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

def _fetch_messages(
    country: str, channel: str, day: dt.date,
) -> list[tuple]:
    """
    Pull a day's worth of human messages for a country/channel pair, sorted.

    Returns list of (id, date_iso, text, sender_id, reply_to_msg_id) tuples.
    Bot messages and empty-text rows are excluded.
    """
    day_str = day.isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, date, text, sender_id, reply_to_msg_id
            FROM messages
            WHERE country = ?
              AND channel = ?
              AND substr(date, 1, 10) = ?
              AND (is_bot = 0 OR is_bot IS NULL)
              AND text IS NOT NULL
              AND length(text) > 0
            ORDER BY date ASC, id ASC
            """,
            (country, channel, day_str),
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_line(date_iso: str, text: str, sender_id, reply_to_msg_id) -> str:
    """Build one line for the sample file. UTC time, impersonal sender id."""
    # date_iso has shape '2026-04-28T12:34:56+00:00' (UTC) — extract HH:MM
    try:
        hhmm = date_iso[11:16]
    except Exception:
        hhmm = "??:??"
    sid = f"u/{sender_id}" if sender_id is not None else "u/?"
    if reply_to_msg_id:
        # Reply markers help an LLM (or human reader) reconstruct chains.
        # Keep only the numeric tail of msg_id like "Brazil_ChatForum_408865".
        tail = str(reply_to_msg_id).rsplit("_", 1)[-1]
        head = f"[{hhmm}] {sid} ← reply {tail}"
    else:
        head = f"[{hhmm}] {sid}"
    # text is already sanitize_text()-ed; collapse any residual newlines so
    # one message = one line (LLM-friendly).
    flat = " ".join(text.split())
    return f"{head}: {flat}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_samples_dir() -> Path:
    """The samples directory, derived from db_path. Created if missing."""
    db_path = Path(get_config().storage.db_path)
    samples_dir = db_path.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def dump_country_samples(
    target: SampleTarget, day: dt.date, samples_dir: Path | None = None,
) -> tuple[Path, int]:
    """
    Write one country's daily sample file. Returns (path, message_count).

    The file is overwritten every run for the same (country, day) — safe to
    re-run; no append/dedup logic needed.
    """
    if samples_dir is None:
        samples_dir = get_samples_dir()

    rows = _fetch_messages(target.country, target.channel, day)

    country_dir = samples_dir / target.country
    country_dir.mkdir(parents=True, exist_ok=True)
    out_path = country_dir / f"{day.isoformat()}.txt"

    header = (
        f"# country={target.country} channel={target.channel} "
        f"day={day.isoformat()} (UTC times) messages={len(rows)}\n"
    )
    body_lines = [
        _format_line(date_iso, text, sender_id, reply_to_msg_id)
        for (_id, date_iso, text, sender_id, reply_to_msg_id) in rows
    ]
    out_path.write_text(header + "\n".join(body_lines) + ("\n" if body_lines else ""),
                        encoding="utf-8")

    return out_path, len(rows)


def dump_all_targets(
    day: dt.date, targets: tuple[SampleTarget, ...] = DEFAULT_SAMPLE_TARGETS,
) -> list[tuple[SampleTarget, Path, int]]:
    """Dump samples for all configured targets for a given day."""
    samples_dir = get_samples_dir()
    results = []
    for target in targets:
        try:
            path, count = dump_country_samples(target, day, samples_dir)
            log.info(
                "Sample dump: country=%s channel=%s day=%s -> %s (%d messages)",
                target.country, target.channel, day.isoformat(), path, count,
            )
            results.append((target, path, count))
        except Exception as e:
            log.error(
                "Sample dump failed for country=%s channel=%s: %s",
                target.country, target.channel, e,
            )
    return results
