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

from .config import get_config, log

# --- Country code → human-readable Russian name (for hashtags) ----------------

# Lazy fallback; we try country_codes module first.
_COUNTRY_RU_FALLBACK: dict[str, str] = {
    "br": "Бразилия",
    "id": "Индонезия",
    "lk": "ШриЛанка",
    "mu": "Маврикий",
    "at": "Австрия",
    "ar": "Аргентина",
    "be": "Бельгия",
    "vn": "Вьетнам",
    "tr": "Турция",
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
    """Build a Firestore client using service account."""
    from .google_auth import build_firestore_client

    return build_firestore_client()


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
    db,
    collection: str,
    channel_target: str,
    recent_countries: list[str] | None = None,
    excluded_countries: set[str] | None = None,
) -> PostCandidate | None:
    """
    Pick the oldest unposted doc, preferring a country NOT in the recent
    rotation window. Falls back gracefully if the queue is starved of
    variety.

    Strategy:
        1. Read up to 100 oldest unposted docs (by createdAt asc).
        2. Filter out excluded countries (operator-configured exclude list)
           and entries without content.
        3. Try widest variety first: candidates whose country is NOT in
           recent_countries (last N posted). If any — return oldest of those.
        4. If queue is dominated by a country that's in recent_countries
           (the typical situation after a historical-feed dump), shorten
           the rotation window step by step: try excluding only the
           last-2, then last-1, then nothing. This guarantees we don't
           starve when the feed is skewed.
    """
    from google.cloud import firestore as fs

    channel_key = _channel_field_safe(channel_target)

    # Firestore can't query "field doesn't exist" easily, so we use a where
    # clause that excludes only `posted == true`. New docs without postedTo
    # are returned (treated as not posted).
    # Note: docs with postedTo.<key>.posted == false also returned.
    coll = db.collection(collection)
    # We pull a bigger batch and filter Python-side because compound queries
    # with map fields require composite indexes on Firestore.
    docs = list(
        coll.order_by("createdAt", direction=fs.Query.ASCENDING).limit(100).stream()
    )

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
        if created is not None and hasattr(created, "to_pydatetime"):
            created = created.to_pydatetime()
        candidates.append(
            PostCandidate(
                doc_id=d.id,
                country=country or "any",
                title=str(data.get("title") or ""),
                content=str(data.get("content") or ""),
                tag=str(data.get("tag") or ""),
                created_at=created if isinstance(created, dt.datetime) else None,
            )
        )

    if not candidates:
        return None

    # Country rotation: try to avoid recent_countries entirely. If that
    # starves the pool, shrink the rotation window step by step.
    if recent_countries:
        for window_size in range(len(recent_countries), 0, -1):
            recent_set = set(recent_countries[-window_size:])
            diff = [c for c in candidates if c.country not in recent_set]
            if diff:
                return diff[0]
    return candidates[0]


def _load_recent_posted_countries(
    db, collection: str, channel_target: str, n: int = 3
) -> list[str]:
    """
    Read the most recent N posted docs for this channel and return their
    countries (oldest of the N first). Survives bot restarts: rotation
    state lives in Firestore, not in process memory.

    Why this exists: Dokploy redeploys reset the in-process `last_country`
    to None. Without persistence, every restart picks the oldest unposted
    overall — which after a historical-BR-feed run is always BR. Three
    redeploys in a row → three BR posts in a row.
    """
    from google.cloud import firestore as fs

    channel_key = _channel_field_safe(channel_target)
    try:
        docs = list(
            db.collection(collection)
            .where(f"postedTo.{channel_key}.posted", "==", True)
            .order_by(
                f"postedTo.{channel_key}.posted_at", direction=fs.Query.DESCENDING
            )
            .limit(n)
            .stream()
        )
    except Exception as e:
        log.warning(
            "Channel poster: could not load recent posted history "
            "(%s). Rotation will start fresh. May need a Firestore composite "
            "index for postedTo.%s.posted + postedTo.%s.posted_at.",
            e,
            channel_key,
            channel_key,
        )
        return []

    # Server returns newest first; we want oldest-first so the most-recent
    # is at the END of the list — matches how recent_countries grows.
    countries = []
    for d in reversed(docs):
        data = d.to_dict() or {}
        c = (data.get("country") or "").lower()
        if c:
            countries.append(c)
    return countries


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


def mark_posted(
    db, collection: str, doc_id: str, channel_target: str, posted_text: str
) -> None:
    """Update postedTo.<channel> to mark this doc as posted with metadata."""
    channel_key = _channel_field_safe(channel_target)
    now = dt.datetime.now(dt.timezone.utc)
    db.collection(collection).document(doc_id).update(
        {
            f"postedTo.{channel_key}.posted": True,
            f"postedTo.{channel_key}.posted_at": now,
            f"postedTo.{channel_key}.text": posted_text,
            f"postedTo.{channel_key}.target": channel_target,
        }
    )


async def post_one(bot_client, recent_countries: list[str] | None = None) -> str | None:
    """
    Pick + post one story to the configured channel.

    Returns the country code that was posted (so caller can append it to
    its `recent_countries` deque for the next slot), or None if nothing
    to post / skip.
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
        db,
        cfg.google.firestore_collection,
        cfg.channel.target,
        recent_countries=recent_countries,
        excluded_countries=excluded,
    )
    if candidate is None:
        log.info("Channel poster: no unposted stories found.")
        return None

    text = format_message(candidate)
    log.info(
        "Channel poster: posting doc=%s country=%s title=%r",
        candidate.doc_id,
        candidate.country,
        candidate.title[:60],
    )

    try:
        await bot_client.send_message(cfg.channel.target, text, link_preview=False)
    except Exception as e:
        log.error(
            "Channel poster: failed to send to %s: %s. Doc %s NOT marked posted.",
            cfg.channel.target,
            e,
            candidate.doc_id,
        )
        return None

    try:
        mark_posted(
            db,
            cfg.google.firestore_collection,
            candidate.doc_id,
            cfg.channel.target,
            text,
        )
    except Exception as e:
        # Posted to channel but failed to mark — duplicate risk on next run.
        log.error(
            "Channel poster: posted to channel but Firestore update FAILED for %s: %s. "
            "May result in duplicate on next run; investigate manually.",
            candidate.doc_id,
            e,
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
    today_slots = [
        now_local.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        for t in slots
    ]
    future = [t for t in today_slots if t > now_local]
    if future:
        return min(future)
    # All today's slots passed — first slot of tomorrow
    tomorrow = (now_local + dt.timedelta(days=1)).replace(
        hour=slots[0].hour,
        minute=slots[0].minute,
        second=0,
        microsecond=0,
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
        cfg.channel.target,
        cfg.channel.posts_per_day,
        cfg.channel.window_start_hour,
        cfg.channel.window_end_hour,
        ", ".join(t.strftime("%H:%M") for t in slots),
    )

    # Rotation memory persists across restarts via Firestore — without this
    # every Dokploy redeploy reset the in-memory last_country, and on a BR-
    # skewed queue every first-post-after-restart was BR. Three deploys in
    # a day → three BR posts in a row. _load_recent_posted_countries reads
    # the most recent N posted docs to this channel and seeds the rotation.
    from collections import deque

    ROTATION_WINDOW = 3  # avoid repeating any of last 3 posted countries
    try:
        db_for_history = _build_firestore_client()
        seed = _load_recent_posted_countries(
            db_for_history,
            cfg.google.firestore_collection,
            cfg.channel.target,
            n=ROTATION_WINDOW,
        )
    except Exception as e:
        log.warning(
            "Channel poster: rotation seed from Firestore failed (%s); "
            "starting with empty rotation.",
            e,
        )
        seed = []
    recent_countries: deque[str] = deque(seed, maxlen=ROTATION_WINDOW)
    if seed:
        log.info(
            "Channel poster: rotation seeded from Firestore history: %s",
            list(recent_countries),
        )

    last_slot: dt.datetime | None = None

    while True:
        now = dt.datetime.now(tz)
        # If the previous iteration fired with negative jitter we may have
        # posted BEFORE the slot's nominal moment — in that case _next_slot_at
        # would re-pick the same slot. Advance past it explicitly.
        search_from = now
        if last_slot is not None and search_from <= last_slot:
            search_from = last_slot + dt.timedelta(seconds=1)

        next_slot = _next_slot_at(search_from, slots)
        # Jitter ±N minutes on the slot moment itself
        jitter = random.randint(-cfg.channel.jitter_minutes, cfg.channel.jitter_minutes)
        next_at = next_slot + dt.timedelta(minutes=jitter)
        sleep_seconds = max(1, int((next_at - now).total_seconds()))
        log.info(
            "Channel poster: next post at %s (sleep %ds, recent=%s)",
            next_at.isoformat(timespec="minutes"),
            sleep_seconds,
            list(recent_countries),
        )
        await asyncio.sleep(sleep_seconds)

        last_slot = next_slot
        try:
            posted_country = await post_one(
                bot_client, recent_countries=list(recent_countries)
            )
            if posted_country:
                recent_countries.append(posted_country)
        except Exception as e:
            log.exception("Channel poster: unhandled error in slot: %s", e)
            # Continue loop — don't die on one bad slot
            await asyncio.sleep(60)
