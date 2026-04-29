import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

from telethon.errors import RPCError

from .config import get_config, log
from .daily_artifact import build_daily_artifact, artifact_claims_as_dicts
from .db import (
    get_relevant_messages_for_country_last_24h,
    get_relevant_messages_last_24h,
)
from .knowledge_db import get_knowledge_for_country, mark_outdated
from .knowledge_loader import load_daily_claims
from .llm import llm_summarize, llm_summarize_brief
from .message_utils import send_message_long
from .telegram_client import get_bot_client
from .telegraph import post_to_telegraph


async def _post_digest(bot_client, target: str, day: dt.date, messages, country: str = ""):
    """Generate and post a digest to the target chat/channel."""
    cfg = get_config()

    # Fetch knowledge context for this country
    knowledge = []
    if country:
        knowledge = get_knowledge_for_country(country, limit=50)

    if messages:
        summary = llm_summarize(day, messages, knowledge=knowledge, country=country)
    else:
        summary = (
            f"No messages to summarize for the last 24 hours"
            f" (labelled as {day.isoformat()})."
        )

    try:
        if cfg.bot.summary_brief and messages:
            telegraph_url = post_to_telegraph(
                title=f"Digest {day.isoformat()}", html=summary
            )
            brief = llm_summarize_brief(day, summary)
            outgoing = (
                f"{brief}\n\n"
                f'<a href="{telegraph_url}">Full digest on Telegraph</a>'
            )
        else:
            outgoing = summary

        await send_message_long(
            bot_client,
            target,
            outgoing,
            parse_mode="html",
        )
        log.info("Digest sent to %s (country=%s)", target, country or "default")
    except RPCError as e:
        log.exception("Failed to send digest to %s: %s", target, e)


def _run_daily_understanding(day: dt.date, country: str) -> int:
    """
    Daily understanding pipeline (per country):
    1. Build artifact from day's messages — scoped to this country's channels
    2. Extract claims from artifact
    3. Load claims into knowledge table via merger

    Returns number of new claims loaded.
    """
    from .sources_db import get_channel_values_for_country

    channel_values = get_channel_values_for_country(country)
    if not channel_values:
        log.warning(
            "Daily understanding skipped for country=%s: no channels in sources DB.",
            country,
        )
        return 0

    artifact = build_daily_artifact(day, channels=list(channel_values))

    if artifact["claims_count"] == 0:
        log.info(
            "No claims extracted for %s (country=%s, %d messages)",
            day.isoformat(), country, artifact["messages_count"],
        )
        return 0

    claims = artifact_claims_as_dicts(artifact)
    loaded = load_daily_claims(claims, country)

    log.info(
        "Daily understanding for %s (country=%s): %d messages -> %d spans -> %d claims -> %d loaded",
        day.isoformat(), country,
        artifact["messages_count"], artifact["spans_count"],
        artifact["claims_count"], loaded,
    )
    return loaded


async def summary_scheduler():
    cfg = get_config()

    bot_client = get_bot_client()
    summary_hour = cfg.bot.summary_hour
    summary_minute = cfg.bot.summary_minute
    tz = ZoneInfo(cfg.bot.time_zone)

    log.info(
        "Scheduler started - daily summary at %02d:%02d (%s)",
        summary_hour,
        summary_minute,
        cfg.bot.time_zone,
    )
    last_run_for = None

    while True:
        now = dt.datetime.now(tz)
        today = now.date()

        if now.hour == summary_hour and now.minute == summary_minute:
            if last_run_for == today:
                await asyncio.sleep(60)
                continue

            log.info(
                "Daily pipeline starting: %s (ending %s)",
                today.isoformat(), now.isoformat(),
            )

            # Mark outdated knowledge entries
            outdated = mark_outdated(days=90)
            if outdated:
                log.info("Marked %d knowledge entries as outdated.", outdated)

            # --- STEP 1: Daily understanding (heuristic, no LLM) ---
            # Build artifact from yesterday's messages and load claims.
            # Country list comes from the sources DB — that is the authoritative
            # source for active countries. Static config is only bootstrap.
            from .sources_db import get_active_countries
            yesterday = today - dt.timedelta(days=1)
            countries = get_active_countries()

            if countries:
                for country in countries:
                    try:
                        loaded = _run_daily_understanding(yesterday, country)
                        log.info("Memory increment for %s: %d claims", country, loaded)
                    except Exception as e:
                        log.error("Daily understanding failed for %s: %s", country, e)
            else:
                try:
                    _run_daily_understanding(yesterday, "default")
                except Exception as e:
                    log.error("Daily understanding failed: %s", e)

            # --- STEP 2: Digest (LLM call via DeepSeek), one per country ---
            if countries:
                # Resolve digest targets from sources DB (dynamic) with fallback
                # to config (legacy). DB is the source of truth.
                from .sources_db import get_digest_target

                for country in countries:
                    target = get_digest_target(country) or cfg.sources.digest_targets.get(country)
                    if not target:
                        log.warning("No digest_target for country '%s', skipping.", country)
                        continue
                    messages = get_relevant_messages_for_country_last_24h(
                        country, max_docs=cfg.llm.max_messages,
                    )
                    log.info(
                        "Digest country=%s: %d messages retrieved", country, len(messages),
                    )
                    await _post_digest(bot_client, target, today, messages, country=country)
            else:
                # Legacy mode: single digest to summary_target, no country split
                messages = get_relevant_messages_last_24h(max_docs=cfg.llm.max_messages)
                await _post_digest(bot_client, cfg.bot.summary_target, today, messages)

            last_run_for = today
            await asyncio.sleep(65)
        else:
            await asyncio.sleep(30)
