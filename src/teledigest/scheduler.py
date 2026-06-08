import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

from telethon.errors import RPCError

from .config import get_config, log
from .daily_artifact import artifact_claims_as_dicts, build_daily_artifact
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

# Embedding is gated by Gemini's free-tier RPD quota, which resets at midnight
# America/Los_Angeles (Pacific). We run one pass per Pacific day, keyed on the
# date — so it fires immediately on container start (catch-up: a mid-day
# redeploy must NOT waste the rest of the day's quota) and again at each 00:00 PT
# rollover with fresh quota.
_EMBED_TZ = ZoneInfo("America/Los_Angeles")


async def _post_digest(
    bot_client, target: str, day: dt.date, messages, country: str = ""
):
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
                f"{brief}\n\n" f'<a href="{telegraph_url}">Full digest on Telegraph</a>'
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
            day.isoformat(),
            country,
            artifact["messages_count"],
        )
        return 0

    claims = artifact_claims_as_dicts(artifact)
    loaded = load_daily_claims(claims, country)

    log.info(
        "Daily understanding for %s (country=%s): %d messages -> %d spans -> %d claims -> %d loaded",
        day.isoformat(),
        country,
        artifact["messages_count"],
        artifact["spans_count"],
        artifact["claims_count"],
        loaded,
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
                today.isoformat(),
                now.isoformat(),
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
                    target = get_digest_target(
                        country
                    ) or cfg.sources.digest_targets.get(country)
                    if not target:
                        log.warning(
                            "No digest_target for country '%s', skipping.", country
                        )
                        continue
                    messages = get_relevant_messages_for_country_last_24h(
                        country,
                        max_docs=cfg.llm.max_messages,
                    )
                    log.info(
                        "Digest country=%s: %d messages retrieved",
                        country,
                        len(messages),
                    )
                    await _post_digest(
                        bot_client, target, today, messages, country=country
                    )
            else:
                # Legacy mode: single digest to summary_target, no country split
                messages = get_relevant_messages_last_24h(max_docs=cfg.llm.max_messages)
                await _post_digest(bot_client, cfg.bot.summary_target, today, messages)

            # --- STEP 3: Daily samples dump (raw chat by country/channel) ---
            # Writes plain-text files alongside the DB for human/LLM review.
            # Same `yesterday` window as STEP 1 so artifact and sample agree.
            try:
                from .daily_samples import dump_all_targets

                dump_all_targets(yesterday)
            except Exception as e:
                log.error("Daily samples dump failed (non-fatal): %s", e)

            # --- STEP 4: Push samples directory to Google Drive (optional) ---
            # If [google] is configured, every .txt under samples/ is uploaded
            # (idempotent — same name overwrites). Never fatal: a Drive outage
            # leaves the bot pipeline intact, files stay locally for retry.
            try:
                from .drive_uploader import upload_samples_dir

                upload_samples_dir()
            except Exception as e:
                log.error("Drive upload failed (non-fatal): %s", e)

            # --- STEP 5: Mine extracted patterns from new samples ---
            # Replaces the old Apps Script timer that read samples from Drive
            # and pushed patterns to Firestore. Now everything happens in this
            # container: read samples/, call Gemini extraction, write to SQLite
            # extracted_patterns (pending queue). Idempotent — already-processed
            # files are skipped via the .processed sidecar marker.
            #
            # run_extraction_pass blocks on time.sleep internally (model rotation
            # pacing). Wrap in asyncio.to_thread so the event loop keeps serving
            # Telegram bot commands while extraction crunches in a worker thread.
            try:
                from .extraction import run_extraction_pass

                fp, sv, at = await asyncio.to_thread(run_extraction_pass)
                log.info("Extraction pass: files=%d saved=%d attempted=%d", fp, sv, at)
            except Exception as e:
                log.error("Extraction pass failed (non-fatal): %s", e)

            # NOTE: embedding (formerly STEP 6 here) now runs in its own
            # independent embed_scheduler() task, triggered just after the
            # Pacific midnight RPD reset. Keeping it inline blocked this loop
            # for the full multi-hour pass (10s/text), which both delayed the
            # next day's extraction/digest and made a long pass straddle the
            # quota reset. See embed_scheduler below.
            last_run_for = today
            await asyncio.sleep(65)
        else:
            await asyncio.sleep(30)


async def embed_scheduler() -> None:
    """Independent embed pass — one per Pacific quota-day, with catch-up on start.

    Runs in its own task (see main.asyncio.gather) so a multi-hour pass never
    blocks the summary/extraction loop. Fires whenever the current Pacific day
    hasn't been processed yet: immediately on container start (so a mid-day
    redeploy doesn't strand the rest of the day's RPD quota until next midnight)
    and again at each 00:00 PT rollover with fresh quota. run_embed_pass drains
    pending wisdom + wiki into Qdrant until every key hits its soft cap, then
    returns (the pump breaks on exhaustion — see embed_pump); if keys are already
    spent it returns fast. Body guarded so a transient failure can't kill it.
    """
    log.info(
        "Embed scheduler started - one pass per Pacific day + catch-up on start (%s)",
        _EMBED_TZ.key,
    )
    last_embed_for: dt.date | None = None

    while True:
        try:
            today_pt = dt.datetime.now(_EMBED_TZ).date()
            if last_embed_for != today_pt:
                last_embed_for = today_pt
                log.info("Embed pass starting (PT-day %s)", today_pt.isoformat())
                try:
                    from .embed_pump import run_embed_pass

                    res = await asyncio.to_thread(run_embed_pass, 10)
                    log.info("Embed pass: %s", res)
                except Exception as e:
                    log.error("Embed pass failed (non-fatal): %s", e)
                await asyncio.sleep(65)
            else:
                # Idle until the Pacific day rolls over (checked ~every minute).
                await asyncio.sleep(60)
        except Exception as e:  # never let the scheduler loop die
            log.exception("embed_scheduler loop error (continuing): %s", e)
            await asyncio.sleep(30)
