"""
channel_poster.py — Auto-post stories from Firestore to a Telegram channel.

Pipeline:
    Apps Script extracts stories → Firestore.telegram_queue
    ↓
    bot reads unposted, picks oldest with country rotation
    ↓
    bot formats message (canonical RU content + hashtags)
    ↓
    bot posts to configured Telegram channel
    ↓
    bot updates Firestore: postedTo.<channel>.posted = true

Schedule: N posts/day in a daytime window with light jitter so the rhythm
looks natural. Default 5 posts/day in 08:00-24:00 = every 3h12m ± 5min.

Country rotation: never two posts in a row from the same country (unless
queue has only one country left — then we accept it rather than starve).

Auth: same OAuth user creds as drive_uploader (token.json with both
`drive.file` and `datastore` scopes). No service account.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import random
from dataclasses import dataclass
from typing import Any

from .config import get_config, log

# --- Country code → human-readable Russian name (for hashtags) ----------------

# Lazy fallback; we try country_codes module first.
_COUNTRY_RU_FALLBACK: dict[str, str] = {
    "br": "Бразилия", "id": "Индонезия", "lk": "ШриЛанка",
    "mu": "Маврикий", "at": "Австрия", "ar": "Аргентина",
    "be": "Бельгия", "vn": "Вьетнам", "tr": "Турция",
    "any": "ЛайфхакиВПути",
}


def _country_hashtag(code: str) -> str:
    """Russian-name hashtag for a country code, e.g. 'br' -> '#Бразилия'."""
    code = (code or "").lower()
    name = None
    try:
        from . import country_codes
        info = country_codes.COUNTRIES.get(code)
        if info:
            name = info[0]  # display name
    except Exception:
        pass
    if not name:
        name = _COUNTRY_RU_FALLBACK.get(code, code.upper())
    # Telegram hashtags allow Cyrillic. Strip whitespace for tag-safety.
    return "#" + "".join(c for c in name if c.isalnum())


def _tag_hashtag(tag: str) -> str:
    """English tag → hashtag, e.g. 'Finance' -> '#Finance'."""
    if not tag:
        return ""
    return "#" + "".join(c for c in tag if c.isalnum())


# --- Firestore client ---------------------------------------------------------

def _build_firestore_client():
    """Build a Firestore client using the same OAuth token as Drive."""
    from google.oauth2.credentials import Credentials
    from google.cloud import firestore

    cfg = get_config()
    if not cfg.google.token_path.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {cfg.google.token_path}. "
            "Run scripts/drive_oauth_init.py with datastore scope."
        )
    if not cfg.google.firestore_project_id:
        raise RuntimeError("[google] firestore_project_id is not set in config.")

    creds = Credentials.from_authorized_user_file(
        str(cfg.google.token_path),
        scopes=["https://www.googleapis.com/auth/datastore"],
    )
    # Refresh now if needed; firestore client also refreshes on demand.
    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
        cfg.google.token_path.write_text(creds.to_json(), encoding="utf-8")

    return firestore.Client(
        project=cfg.google.firestore_project_id,
        database=cfg.google.firestore_database,
        credentials=creds,
    )


# --- Selection logic ----------------------------------------------------------

@dataclass
class PostCandidate:
    """One Firestore document picked for posting."""
    doc_id: str
    country: str
    title: str
    content: str
    tag: str
    created_at: dt.datetime | None


def _channel_field_safe(target: str) -> str:
    """
    Build a Firestore-safe field name for postedTo.<channel>.

    Telegram handles like '@luky_channel' contain '@' which is not allowed
    inside Firestore field paths. Strip non-alphanumeric, lowercase.
        '@luky_channel' -> 'luky_channel'
        '-1001234567890' -> '1001234567890'
    """
    return "".join(c for c in target.lower() if c.isalnum() or c == "_") or "default"


def select_next_candidate(
    db, collection: str, channel_target: str,
    last_country: str | None = None,
    excluded_countries: set[str] | None = None,
) -> PostCandidate | None:
    """
    Pick the oldest unposted doc, preferring a country different from
    `last_country` to keep the feed varied.

    Strategy:
        1. Read up to 50 oldest unposted docs (by createdAt asc).
        2. Filter out excluded countries.
        3. If `last_country` is set and any candidate has a different
           country — return the oldest of those.
        4. Otherwise return the oldest overall (allow repeat).
    """
    from google.cloud import firestore as fs

    channel_key = _channel_field_safe(channel_target)
    posted_field = f"postedTo.{channel_key}.posted"

    # Firestore can't query "field doesn't exist" easily, so we use a where
    # clause that excludes only `posted == true`. New docs without postedTo
    # are returned (treated as not posted).
    # Note: docs with postedTo.<key>.posted == false also returned.
    coll = db.collection(collection)
    # We pull a bigger batch and filter Python-side because compound queries
    # with map fields require composite indexes on Firestore.
    docs = list(coll.order_by("createdAt", direction=fs.Query.ASCENDING).limit(100).stream())

    candidates: list[PostCandidate] = []
    for d in docs:
        data = d.to_dict() or {}
        # Skip if already posted to this channel
        posted_to = data.get("postedTo") or {}
        ch = posted_to.get(channel_key) or {}
        if ch.get("posted") is True:
            continue
        country = (data.get("country") or "").lower()
        if excluded_countries and country in excluded_countries:
            continue
        if not (data.get("content") or "").strip():
            continue
        created = data.get("createdAt")
        if hasattr(created, "to_pydatetime"):
            created = created.to_pydatetime()
        candidates.append(PostCandidate(
            doc_id=d.id,
            country=country or "any",
            title=str(data.get("title") or ""),
            content=str(data.get("content") or ""),
            tag=str(data.get("tag") or ""),
            created_at=created if isinstance(created, dt.datetime) else None,
        ))

    if not candidates:
        return None

    # Country rotation: prefer different country than last
    if last_country:
        diff = [c for c in candidates if c.country != last_country]
        if diff:
            return diff[0]
    return candidates[0]


# --- Message formatting -------------------------------------------------------

def format_message(candidate: PostCandidate) -> str:
    """Compose final Telegram message: content + country and tag hashtags."""
    parts = [candidate.content.strip()]
    tags = []
    ch = _country_hashtag(candidate.country)
    if ch:
        tags.append(ch)
    th = _tag_hashtag(candidate.tag)
    if th:
        tags.append(th)
    if tags:
        parts.append(" ".join(tags))
    return "\n\n".join(parts)


# --- Posting + Firestore update ----------------------------------------------

def mark_posted(db, collection: str, doc_id: str, channel_target: str, posted_text: str) -> None:
    """Update postedTo.<channel> to mark this doc as posted with metadata."""
    channel_key = _channel_field_safe(channel_target)
    now = dt.datetime.now(dt.timezone.utc)
    db.collection(collection).document(doc_id).update({
        f"postedTo.{channel_key}.posted": True,
        f"postedTo.{channel_key}.posted_at": now,
        f"postedTo.{channel_key}.text": posted_text,
        f"postedTo.{channel_key}.target": channel_target,
    })


async def post_one(bot_client, last_country: str | None = None) -> str | None:
    """
    Pick + post one story to the configured channel.

    Returns the country code that was posted (so caller can pass it as
    `last_country` next time for rotation), or None if nothing to post / skip.
    """
    cfg = get_config()
    if not cfg.channel.enabled:
        return None
    if not cfg.google.firestore_project_id:
        log.warning("Channel poster: firestore_project_id not configured.")
        return None

    excluded = set(
        x.strip().lower()
        for x in (cfg.channel.exclude_countries or "").split(",")
        if x.strip()
    )

    try:
        db = _build_firestore_client()
    except Exception as e:
        log.error("Channel poster: firestore client init failed: %s", e)
        return None

    candidate = select_next_candidate(
        db, cfg.google.firestore_collection, cfg.channel.target,
        last_country=last_country, excluded_countries=excluded,
    )
    if candidate is None:
        log.info("Channel poster: no unposted stories found.")
        return None

    text = format_message(candidate)
    log.info(
        "Channel poster: posting doc=%s country=%s title=%r",
        candidate.doc_id, candidate.country, candidate.title[:60],
    )

    try:
        await bot_client.send_message(cfg.channel.target, text, link_preview=False)
    except Exception as e:
        log.error(
            "Channel poster: failed to send to %s: %s. Doc %s NOT marked posted.",
            cfg.channel.target, e, candidate.doc_id,
        )
        return None

    try:
        mark_posted(db, cfg.google.firestore_collection,
                    candidate.doc_id, cfg.channel.target, text)
    except Exception as e:
        # Posted to channel but failed to mark — duplicate risk on next run.
        log.error(
            "Channel poster: posted to channel but Firestore update FAILED for %s: %s. "
            "May result in duplicate on next run; investigate manually.",
            candidate.doc_id, e,
        )
    return candidate.country


# --- Schedule loop ------------------------------------------------------------

def _todays_slots(cfg) -> list[dt.time]:
    """
    Compute N evenly spaced posting times within the daytime window.

    For posts_per_day=5, window 8..24 -> 8:00, 11:12, 14:24, 17:36, 20:48.
    Jitter is added at runtime, not stored here.
    """
    n = max(1, cfg.channel.posts_per_day)
    start = cfg.channel.window_start_hour
    end = cfg.channel.window_end_hour
    span_minutes = (end - start) * 60
    if span_minutes <= 0 or n <= 0:
        return []
    step = span_minutes / n
    slots = []
    for i in range(n):
        offset = int(round(step * i))
        h, m = divmod(start * 60 + offset, 60)
        slots.append(dt.time(hour=h % 24, minute=m))
    return slots


def _next_slot_at(now_local: dt.datetime, slots: list[dt.time]) -> dt.datetime:
    """Find the next datetime (today or tomorrow) matching any slot."""
    today_slots = [now_local.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                   for t in slots]
    future = [t for t in today_slots if t > now_local]
    if future:
        return min(future)
    # All today's slots passed — first slot of tomorrow
    tomorrow = (now_local + dt.timedelta(days=1)).replace(
        hour=slots[0].hour, minute=slots[0].minute, second=0, microsecond=0,
    )
    return tomorrow


async def channel_poster_loop():
    """
    Long-running task: posts to channel on schedule.

    Runs alongside summary_scheduler. Wakes near each slot (with jitter),
    picks one story, posts it, updates Firestore, sleeps until next slot.
    """
    from zoneinfo import ZoneInfo
    from .telegram_client import get_bot_client

    cfg = get_config()
    if not cfg.channel.enabled:
        log.info("Channel poster: disabled (no [channel] config) — loop will idle.")
        # Idle forever; the rest of the bot runs normally.
        while True:
            await asyncio.sleep(3600)

    tz = ZoneInfo(cfg.bot.time_zone)
    bot_client = get_bot_client()
    slots = _todays_slots(cfg)
    if not slots:
        log.error("Channel poster: invalid schedule (no slots). Disabling loop.")
        while True:
            await asyncio.sleep(3600)

    log.info(
        "Channel poster started: target=%s posts/day=%d window=%02d:00-%02d:00 slots=%s",
        cfg.channel.target, cfg.channel.posts_per_day,
        cfg.channel.window_start_hour, cfg.channel.window_end_hour,
        ", ".join(t.strftime("%H:%M") for t in slots),
    )

    last_country: str | None = None

    while True:
        now = dt.datetime.now(tz)
        next_at = _next_slot_at(now, slots)
        # Jitter ±N minutes on the slot moment itself
        jitter = random.randint(-cfg.channel.jitter_minutes, cfg.channel.jitter_minutes)
        next_at = next_at + dt.timedelta(minutes=jitter)
        sleep_seconds = max(1, int((next_at - now).total_seconds()))
        log.info("Channel poster: next post at %s (sleep %ds)",
                 next_at.isoformat(timespec="minutes"), sleep_seconds)
        await asyncio.sleep(sleep_seconds)

        try:
            posted_country = await post_one(bot_client, last_country=last_country)
            if posted_country:
                last_country = posted_country
        except Exception as e:
            log.exception("Channel poster: unhandled error in slot: %s", e)
            # Continue loop — don't die on one bad slot
            await asyncio.sleep(60)
