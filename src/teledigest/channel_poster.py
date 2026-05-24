"""
channel_poster.py — Auto-post stories from SQLite to a Telegram channel.

Pipeline (post-Qdrant migration):
    extraction.py writes human_story patterns into SQLite
    extracted_patterns (collection_target='telegram_queue')
    ↓
    bot reads unposted (NOT EXISTS in pattern_posts for this channel),
    picks oldest with alphabetical country rotation
    ↓
    bot formats message (canonical RU content + #страна #tag hashtags)
    ↓
    bot posts to configured Telegram channel
    ↓
    bot inserts pattern_posts(pattern_id, channel, posted_at, ...)
    — the same story can later be posted to vk/discord/etc. without
    re-marking it "spent" globally.

Schedule: N posts/day in a daytime window with light jitter so the rhythm
looks natural. Default 5 posts/day in 08:00-24:00 = every 3h12m ± 5min.

Country rotation: alphabetical round-robin so every country with content
gets fair representation. Won't repeat any of the last N (default 3)
posted countries unless that's the only thing in the queue. Rotation
state is persistent — reads pattern_posts directly, so Dokploy redeploys
don't reset the rotation memory.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import random
from collections import deque
from dataclasses import dataclass

from .config import get_config, log
from .country_codes import COUNTRIES
from .extraction_db import (
    fetch_unposted_stories,
    init_extraction_tables,
    mark_pattern_posted,
    recent_posted_countries,
)


# --- Helpers -----------------------------------------------------------------


def _channel_field_safe(target: str) -> str:
    """Telegram handles like '@luky_channel' need a sanitized key for the
    pattern_posts.channel column. Strip non-alphanumeric, lowercase.
        '@luky_channel'   -> 'luky_channel'
        '-1001234567890'  -> '1001234567890'
    """
    return "".join(c for c in target.lower() if c.isalnum() or c == "_") or "default"


def _country_hashtag(code: str) -> str:
    """Russian-name hashtag for a country code, e.g. 'br' -> '#Бразилия'.

    Special-cases 'any' (universal pattern) → #ЛайфхакиВПути to keep the
    channel's tag bar useful instead of leaking the literal ANY code.
    """
    code = (code or "").lower()
    if code == "any":
        return "#ЛайфхакиВПути"
    info = COUNTRIES.get(code)
    name = info[0] if info else code.upper()
    return "#" + "".join(c for c in name if c.isalnum())


def _tag_hashtag(tag: str) -> str:
    """English tag → hashtag, e.g. 'Finance' -> '#Finance'."""
    if not tag:
        return ""
    return "#" + "".join(c for c in tag if c.isalnum())


# --- Candidate selection -----------------------------------------------------


@dataclass
class PostCandidate:
    """One SQLite row picked for posting."""

    pattern_id: str
    country: str
    title: str
    content: str
    tag: str


_CANDIDATE_BATCH_LIMIT = 2000


def select_next_candidate(
    channel: str,
    recent_countries: list[str] | None = None,
    excluded_countries: set[str] | None = None,
) -> PostCandidate | None:
    """Alphabetical round-robin across countries.

    1. Read up to _CANDIDATE_BATCH_LIMIT oldest unposted patterns for this
       channel. With a queue of ~2K, this is the whole queue in one shot.
    2. Group by country, each bucket sorted oldest-first.
    3. Walk countries alphabetically, advancing past the most-recent posted
       country (recent_countries[-1]). Return the oldest pattern of the
       first country with content available.

    Why round-robin and not "oldest globally": a single big-import (e.g.
    historical BR feed) would dominate the oldest window forever, starving
    every other country. Round-robin gives every country with content a
    1/N rate fair slot.
    """
    rows = fetch_unposted_stories(
        channel,
        limit=_CANDIDATE_BATCH_LIMIT,
        excluded_countries=excluded_countries,
    )
    if not rows:
        return None

    by_country: dict[str, list[PostCandidate]] = {}
    for r in rows:
        country = (r.get("country") or "").lower() or "any"
        content = (r.get("human_story") or "").strip()
        if not content:
            continue
        cand = PostCandidate(
            pattern_id=r["id"],
            country=country,
            title=str(r.get("title") or ""),
            content=content,
            tag=str(r.get("tag") or ""),
        )
        by_country.setdefault(country, []).append(cand)

    if not by_country:
        return None

    countries_sorted = sorted(by_country.keys())

    # Find rotation start. recent_countries[-1] = most recent post; we
    # want the next country after it (and ideally not any of the last N).
    recent_set = set(recent_countries or [])
    start_idx = 0
    if recent_countries:
        last = recent_countries[-1]
        if last in countries_sorted:
            start_idx = (countries_sorted.index(last) + 1) % len(countries_sorted)

    # First pass: try to skip any country in recent_set.
    n = len(countries_sorted)
    for offset in range(n):
        country = countries_sorted[(start_idx + offset) % n]
        if country in recent_set:
            continue
        bucket = by_country.get(country) or []
        if bucket:
            return bucket[0]

    # Second pass: queue only has recently-posted countries left, take any.
    # Better to repeat a country than to starve.
    for offset in range(n):
        country = countries_sorted[(start_idx + offset) % n]
        bucket = by_country.get(country) or []
        if bucket:
            return bucket[0]

    return None


# --- Message formatting ------------------------------------------------------


def format_message(candidate: PostCandidate) -> str:
    """Compose final Telegram message: content + #страна #tag hashtags."""
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


# --- Post one ----------------------------------------------------------------


async def post_one(bot_client, recent_countries: list[str] | None = None) -> str | None:
    """Pick + post one story to the configured channel.

    Returns the country code that was posted (so the caller can append to
    its rotation deque), or None if nothing to post.
    """
    cfg = get_config()
    if not cfg.channel.enabled:
        return None

    channel_key = _channel_field_safe(cfg.channel.target)
    excluded = {
        x.strip().lower()
        for x in (cfg.channel.exclude_countries or "").split(",")
        if x.strip()
    }

    candidate = select_next_candidate(
        channel_key,
        recent_countries=recent_countries,
        excluded_countries=excluded,
    )
    if candidate is None:
        log.info("Channel poster: no unposted stories found.")
        return None

    text = format_message(candidate)
    log.info(
        "Channel poster: posting id=%s country=%s title=%r",
        candidate.pattern_id[:8],
        candidate.country,
        candidate.title[:60],
    )

    try:
        msg = await bot_client.send_message(
            cfg.channel.target, text, link_preview=False
        )
    except Exception as e:
        log.error(
            "Channel poster: send_message to %s failed: %s — "
            "pattern %s NOT marked posted",
            cfg.channel.target,
            e,
            candidate.pattern_id,
        )
        return None

    # Build a message URL if Telethon gave us a message id and the target
    # is a public channel (has @username). Otherwise leave None.
    message_url: str | None = None
    try:
        if msg and getattr(msg, "id", None) and cfg.channel.target.startswith("@"):
            handle = cfg.channel.target.lstrip("@")
            message_url = f"https://t.me/{handle}/{msg.id}"
    except Exception:
        pass

    try:
        mark_pattern_posted(
            candidate.pattern_id, channel_key, text, message_url=message_url
        )
    except Exception as e:
        # Posted to channel but failed to mark — duplicate risk on next run.
        log.error(
            "Channel poster: posted to %s but pattern_posts INSERT FAILED for "
            "%s: %s. Investigate manually to avoid duplicate.",
            cfg.channel.target,
            candidate.pattern_id,
            e,
        )
    return candidate.country


# --- Schedule loop -----------------------------------------------------------


def _todays_slots(cfg) -> list[dt.time]:
    """Compute N evenly spaced posting times within the daytime window.

    posts_per_day=5, window 8..24  ->  8:00, 11:12, 14:24, 17:36, 20:48.
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
    tomorrow = (now_local + dt.timedelta(days=1)).replace(
        hour=slots[0].hour,
        minute=slots[0].minute,
        second=0,
        microsecond=0,
    )
    return tomorrow


async def channel_poster_loop():
    """Long-running task: posts to channel on schedule.

    Runs alongside summary_scheduler. Wakes near each slot (with jitter),
    picks one story, posts it, updates pattern_posts, sleeps until next slot.
    """
    from zoneinfo import ZoneInfo

    from .telegram_client import get_bot_client

    cfg = get_config()
    if not cfg.channel.enabled:
        log.info(
            "Channel poster: disabled (no [channel] config or enabled=false) — "
            "loop will idle."
        )
        while True:
            await asyncio.sleep(3600)

    # Ensure pattern_posts and friends exist — first run on a fresh deploy.
    init_extraction_tables()

    tz = ZoneInfo(cfg.bot.time_zone)
    bot_client = get_bot_client()
    slots = _todays_slots(cfg)
    if not slots:
        log.error("Channel poster: invalid schedule (no slots). Disabling loop.")
        while True:
            await asyncio.sleep(3600)

    channel_key = _channel_field_safe(cfg.channel.target)

    log.info(
        "Channel poster started: target=%s key=%s posts/day=%d "
        "window=%02d:00-%02d:00 slots=%s",
        cfg.channel.target,
        channel_key,
        cfg.channel.posts_per_day,
        cfg.channel.window_start_hour,
        cfg.channel.window_end_hour,
        ", ".join(t.strftime("%H:%M") for t in slots),
    )

    # Rotation memory persists across restarts via pattern_posts.
    ROTATION_WINDOW = 3
    try:
        seed = recent_posted_countries(channel_key, n=ROTATION_WINDOW)
    except Exception as e:
        log.warning(
            "Channel poster: rotation seed from SQLite failed (%s); "
            "starting with empty rotation.",
            e,
        )
        seed = []
    recent_countries: deque[str] = deque(seed, maxlen=ROTATION_WINDOW)
    if seed:
        log.info(
            "Channel poster: rotation seeded from pattern_posts: %s",
            list(recent_countries),
        )

    last_slot: dt.datetime | None = None

    while True:
        now = dt.datetime.now(tz)
        search_from = now
        if last_slot is not None and search_from <= last_slot:
            search_from = last_slot + dt.timedelta(seconds=1)

        next_slot = _next_slot_at(search_from, slots)
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
            await asyncio.sleep(60)
