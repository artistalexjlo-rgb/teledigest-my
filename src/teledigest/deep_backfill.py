"""
deep_backfill.py — Safe batched history pull from Telegram sources.

Pulls up to 1 year of chat history in batches of 200, with 3-second
delays between batches.  Progress is persisted after each batch so
the process can resume after interruption.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from telethon.errors import FloodWaitError

from .config import log
from .db import save_message
from .knowledge_db import get_source_meta, update_source_meta, upsert_source_meta


# Maximum history depth: 1 year
MAX_HISTORY_DAYS = 365
BATCH_SIZE = 200
BATCH_DELAY = 3  # seconds between batches


async def deep_backfill(
    user_client,
    chat_id: int,
    chat_name: str,
    country: str,
    language: str = "ru",
    *,
    progress_callback=None,
) -> int:
    """
    Pull full history (up to 1 year) from a single chat in safe batches.

    Args:
        user_client: Telethon user client
        chat_id: Telegram chat/channel peer ID
        chat_name: Human-readable name for logging and message IDs
        country: Country code for source metadata
        language: Language code (default 'ru')
        progress_callback: Optional async callable(total, oldest_id) for status updates

    Returns:
        Total number of messages saved.
    """
    # Ensure source_meta row exists
    upsert_source_meta(chat_id, chat_name, country, language)

    meta = get_source_meta(chat_id)
    if meta and meta["backfill_done"]:
        log.info("Backfill already done for %s, skipping.", chat_name)
        return 0

    offset_id = meta["backfill_oldest_id"] if meta else 0
    total = meta["total_messages"] if meta else 0
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MAX_HISTORY_DAYS)

    log.info(
        "Starting deep backfill for %s (country=%s, offset_id=%d, cutoff=%s)",
        chat_name, country, offset_id, cutoff.isoformat(),
    )

    while True:
        try:
            messages = []
            async for msg in user_client.iter_messages(
                chat_id,
                limit=BATCH_SIZE,
                offset_id=offset_id,
                reverse=False,  # newest to oldest
            ):
                # Stop if we've gone past 1 year
                if msg.date and msg.date < cutoff:
                    log.info(
                        "Backfill %s: reached 1-year cutoff at msg %d (%s)",
                        chat_name, msg.id, msg.date.isoformat(),
                    )
                    # Save what we got in this batch, then finish
                    messages.append(msg)
                    break
                messages.append(msg)

            if not messages:
                break  # reached the beginning of chat

            reached_cutoff = False
            for msg in messages:
                # Skip bot messages
                sender = getattr(msg, "sender", None)
                if sender and getattr(sender, "bot", False):
                    continue

                text = msg.message or ""
                if text.strip():
                    reply_to = None
                    if msg.reply_to and hasattr(msg.reply_to, "reply_to_msg_id"):
                        reply_to = f"{chat_name}_{msg.reply_to.reply_to_msg_id}"
                    sid = getattr(msg, "sender_id", None)
                    save_message(f"{chat_name}_{msg.id}", chat_name, msg.date, text,
                                 reply_to_msg_id=reply_to, sender_id=sid, is_bot=False)
                # Check cutoff
                if msg.date and msg.date < cutoff:
                    reached_cutoff = True

            offset_id = messages[-1].id
            total += len(messages)

            # Persist progress
            update_source_meta(
                chat_id,
                backfill_oldest_id=offset_id,
                total_messages=total,
            )

            log.info(
                "Backfill %s: %d messages so far (oldest_id=%d)",
                chat_name, total, offset_id,
            )

            if progress_callback:
                await progress_callback(total, offset_id)

            if reached_cutoff:
                break

            # Rate limit protection
            await asyncio.sleep(BATCH_DELAY)

        except FloodWaitError as e:
            log.warning(
                "Backfill %s: FloodWait %d seconds, sleeping...",
                chat_name, e.seconds,
            )
            await asyncio.sleep(e.seconds + 1)
            continue
        except Exception as e:
            log.error("Backfill %s: error at offset_id=%d: %s", chat_name, offset_id, e)
            # Progress is already saved — safe to stop
            raise

    update_source_meta(chat_id, backfill_done=1, total_messages=total)
    log.info("Backfill complete for %s: %d total messages.", chat_name, total)
    return total


async def relink_replies(
    user_client,
    chat_id: int,
    chat_name: str,
    *,
    progress_callback=None,
) -> int:
    """
    Update reply_to_msg_id for existing messages without re-downloading text.
    Fast pass — only fetches message metadata to fill in reply links.
    """
    from .db import get_db_connection

    log.info("Relinking replies for %s...", chat_name)
    offset_id = 0
    total = 0
    updated = 0
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MAX_HISTORY_DAYS)

    while True:
        try:
            messages = []
            async for msg in user_client.iter_messages(
                chat_id, limit=BATCH_SIZE, offset_id=offset_id, reverse=False,
            ):
                if msg.date and msg.date < cutoff:
                    messages.append(msg)
                    break
                messages.append(msg)

            if not messages:
                break

            reached_cutoff = False
            with get_db_connection() as conn:
                cur = conn.cursor()
                for msg in messages:
                    if msg.reply_to and hasattr(msg.reply_to, "reply_to_msg_id"):
                        reply_to = f"{chat_name}_{msg.reply_to.reply_to_msg_id}"
                        msg_id = f"{chat_name}_{msg.id}"
                        cur.execute(
                            "UPDATE messages SET reply_to_msg_id = ? WHERE id = ? AND reply_to_msg_id IS NULL",
                            (reply_to, msg_id),
                        )
                        if cur.rowcount:
                            updated += 1
                    if msg.date and msg.date < cutoff:
                        reached_cutoff = True

            offset_id = messages[-1].id
            total += len(messages)

            if total % 2000 == 0:
                log.info("Relink %s: %d messages scanned, %d links updated", chat_name, total, updated)
                if progress_callback:
                    await progress_callback(total, updated)

            if reached_cutoff:
                break

            await asyncio.sleep(BATCH_DELAY)

        except FloodWaitError as e:
            log.warning("Relink %s: FloodWait %ds", chat_name, e.seconds)
            await asyncio.sleep(e.seconds + 1)
        except Exception as e:
            log.error("Relink %s: error: %s", chat_name, e)
            raise

    log.info("Relink complete for %s: %d scanned, %d links updated.", chat_name, total, updated)
    return updated


async def backfill_country(
    user_client,
    country: str,
    channels: list[dict],
    resolved_chats: dict[int, str],
    *,
    progress_callback=None,
) -> dict[str, int]:
    """
    Run deep backfill for all channels of a given country.

    Args:
        user_client: Telethon user client
        country: Country code (e.g. 'br')
        channels: List of channel config dicts with 'url', 'name', 'language'
        resolved_chats: Mapping of peer_id -> chat_name (already resolved)
        progress_callback: Optional async callable for status updates

    Returns:
        Dict of {chat_name: messages_count}
    """
    results = {}
    for ch in channels:
        # Find the resolved peer_id for this channel
        peer_id = None
        ch_name = None
        for pid, name in resolved_chats.items():
            if name == ch.get("name") or name == ch.get("url", "").lstrip("@"):
                peer_id = pid
                ch_name = name
                break

        if peer_id is None:
            log.warning("Backfill: channel %s not resolved, skipping.", ch.get("url"))
            continue

        count = await deep_backfill(
            user_client,
            peer_id,
            ch_name,
            country,
            ch.get("language", "ru"),
            progress_callback=progress_callback,
        )
        results[ch_name] = count

    return results
