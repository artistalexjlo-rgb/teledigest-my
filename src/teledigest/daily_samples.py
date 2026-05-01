"""
daily_samples.py — Plain-text daily chat samples for human/LLM review.

After the daily digest pipeline finishes, dump the previous day's raw
messages (sanitized text only — see text_sanitize) for every active source
in the `sources` DB. One file per (source, day). Files are written
alongside the SQLite DB so they can be uploaded to external processing
(Google Drive, GCS, etc.) by a separate step.

File layout:
    {db_dir}/samples/{country}/{YYYY-MM-DD}_{country}_{channel_slug}.txt

The country prefix in the filename keeps files self-identifying when they
are moved out of their parent directory (e.g. uploaded to a flat bucket).

One line per message:
    [HH:MM] u/<sender_id>: text
    [HH:MM] u/<sender_id> ← reply <orig_msg_id>: text

Sender IDs are kept as numeric Telegram user IDs (no usernames, no
display names) — same impersonal contract as the rest of the pipeline.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from .config import get_config, log
from .db import get_db_connection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleTarget:
    """One country/channel pair to dump as a daily sample.

    `channel` must match the value stored in `messages.channel` (handle for
    public channels, numeric chat_id as string for invite-link channels).
    """
    country: str
    channel: str


# Filename slug: strip leading dash, replace anything non-alphanumeric/underscore.
# Examples:
#   'Brazil_ChatForum' -> 'Brazil_ChatForum'
#   '-1001631614451'   -> '1001631614451'
#   'balichat'         -> 'balichat'
_SLUG_NON_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _channel_slug(channel: str) -> str:
    s = channel.lstrip("-")
    s = _SLUG_NON_SAFE_RE.sub("_", s)
    return s or "unknown"


def get_sample_targets() -> list[SampleTarget]:
    """All (country, channel) pairs to dump — read fresh from the sources DB.

    Picks the first active source-side channel value the bot actually saves
    under. We try, in order:
        1. sources.chat_id (as string) — invite-link channels save messages
           with this as `messages.channel`.
        2. URL handle (`@foo` -> `foo`) — public channels do.
        3. sources.name fallback — only when both above are unavailable.
    Sources without any usable identifier are skipped with a warning.
    """
    from .sources_db import get_active_sources, _normalize_url_handle

    targets: list[SampleTarget] = []
    for src in get_active_sources():
        country = src.get("country") or ""
        chat_id = src.get("chat_id")
        url = src.get("url") or ""
        handle = _normalize_url_handle(url)
        # Mirror how telegram_client tags messages: invite-link sources
        # produce numeric chat_id channel values; public channels produce
        # the handle. So prefer chat_id when it's a numeric channel.
        # In practice both work because get_active_sources doesn't tell us
        # which one will appear in messages.channel — we pick the most
        # likely one based on URL shape.
        is_invite = url.startswith("https://t.me/+") or url.startswith("http://t.me/+")
        if is_invite and chat_id is not None:
            channel = str(chat_id)
        elif handle:
            channel = handle
        elif chat_id is not None:
            channel = str(chat_id)
        else:
            log.warning(
                "Sample target skipped — no usable channel id for source id=%s url=%s",
                src.get("id"), url,
            )
            continue
        targets.append(SampleTarget(country=country, channel=channel))
    return targets


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
    Write one source's daily sample file. Returns (path, message_count).

    Output path:
        {samples_dir}/{country}/{date}_{country}_{channel-slug}.txt

    The file is overwritten every run for the same (target, day) — safe to
    re-run; no append/dedup logic needed.
    """
    if samples_dir is None:
        samples_dir = get_samples_dir()

    rows = _fetch_messages(target.country, target.channel, day)

    country_dir = samples_dir / target.country
    country_dir.mkdir(parents=True, exist_ok=True)
    slug = _channel_slug(target.channel)
    out_path = country_dir / f"{day.isoformat()}_{target.country}_{slug}.txt"

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
    day: dt.date, targets: list[SampleTarget] | None = None,
) -> list[tuple[SampleTarget, Path, int]]:
    """
    Dump samples for every active source in `sources` DB for a given day.

    By default reads targets fresh from DB so newly-added sources are picked
    up without redeploy.
    """
    if targets is None:
        targets = get_sample_targets()
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
