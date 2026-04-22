"""
qa_extractor.py — LLM-based Q&A extraction from chat messages.

Takes batches of messages, groups them into conversation threads,
and asks the LLM to extract structured Q&A pairs.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

from .config import get_config, log
from .knowledge_db import get_last_processed_msg_id, log_extraction_run
from .knowledge_merger import merge_qa_pair
from .text_sanitize import strip_markdown_fence

# Thread grouping: messages within this window are considered part of
# the same conversation thread.
THREAD_WINDOW_MINUTES = 10
MIN_THREAD_SIZE = 3
EXTRACTION_BATCH = 100


EXTRACTION_SYSTEM = """
You are a knowledge extractor for an expat travel knowledge base.
You receive a batch of Telegram chat messages.

Your job: find USEFUL question-answer pairs where someone asked
a practical question and got a helpful answer from the community.

Rules:
- Extract ONLY pairs where there is a clear question AND a clear answer
- Ignore: greetings, thanks, jokes, arguments, spam, ads, bot messages
- Answer should reflect CONSENSUS if multiple people answered
- Note disagreements only if significant
- Assign exactly ONE category from the list: visa, documents, finance,
  housing, transport, health, telecom, safety, food, language, work,
  culture, other
- Confidence: high (3+ people agree), medium (1-2 answers, no
  contradictions), low (single unverified claim)
- Tags: 2-5 relevant keywords in Russian

CRITICAL — answer quality:
- Be SPECIFIC and CONCRETE: exact addresses, prices, timelines, steps
- BAD: "следует обратиться в Receita Federal"
- GOOD: "Идёшь в Receita Federal с паспортом и comprovante de residência,
  берёшь талон, CPF делают за 15 минут, бесплатно"
- Include NUMBERS: стоимость, сроки, количество дней
- Include NAMES: конкретные учреждения, районы, сервисы
- Write as if explaining to a friend — прямо, по делу, без воды
- Question should be SHORT and конкретный
- Answer should be ACTIONABLE — человек прочитал и знает что делать

Respond ONLY with a JSON array:
[
  {
    "question": "...",
    "answer": "...",
    "category": "...",
    "confidence": "high|medium|low",
    "tags": ["..."]
  }
]

If no useful Q&A pairs found, respond with: []
"""

EXTRACTION_USER = """
Country: {COUNTRY}
Source: {SOURCE_NAME}
Messages ({MSG_COUNT} messages):

{MESSAGES}
"""


def _get_extraction_client():
    """Get the LLM client for extraction — uses [llm.extraction] if configured, else main [llm]."""
    from openai import OpenAI
    cfg = get_config()
    ext = cfg.llm.extraction
    if ext.api_key:
        log.info("Using extraction LLM: %s @ %s", ext.model, ext.base_url)
        return OpenAI(api_key=ext.api_key, base_url=ext.base_url), ext.model, ext.temperature
    return OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url), cfg.llm.model, 0.2


def _call_extraction_llm(country: str, source_name: str, messages_text: str, msg_count: int) -> list[dict]:
    """Call LLM with extraction prompt and parse JSON response."""
    client, model, temperature = _get_extraction_client()
    cfg = get_config()

    user_prompt = EXTRACTION_USER.format(
        COUNTRY=country,
        SOURCE_NAME=source_name,
        MSG_COUNT=msg_count,
        MESSAGES=messages_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            return []

        content = strip_markdown_fence(content.strip())

        # Parse JSON response
        result = json.loads(content)
        if not isinstance(result, list):
            log.warning("Extraction LLM returned non-list: %s", type(result))
            return []
        return result

    except json.JSONDecodeError as e:
        log.warning("Extraction LLM returned invalid JSON: %s", e)
        return []
    except Exception as e:
        log.error("Extraction LLM call failed: %s", e)
        return []


def _group_into_threads(messages: list[dict]) -> list[list[dict]]:
    """
    Group messages into conversation threads.

    Strategy:
    1. First, build reply chains using reply_to_msg_id
    2. Then, group remaining messages by time proximity
    """
    if not messages:
        return []

    # Sort by date
    messages = sorted(messages, key=lambda m: m["date"])

    # Build a lookup: msg_id -> message
    by_id: dict[str, dict] = {}
    for msg in messages:
        by_id[str(msg["id"])] = msg

    # Step 1: Build reply chains
    # Track which messages are already assigned to a thread
    assigned: set[int] = set()
    reply_threads: list[list[dict]] = []

    # Find root messages (those that others reply to)
    roots: dict[str, list[dict]] = {}  # root_msg_id -> [replies]
    for msg in messages:
        reply_to = msg.get("reply_to")
        if reply_to and reply_to in by_id:
            root_id = reply_to
            # Walk up the reply chain to find the root
            seen = {str(msg["id"])}
            while True:
                parent = by_id.get(root_id)
                if not parent or root_id in seen:
                    break
                parent_reply = parent.get("reply_to")
                if parent_reply and parent_reply in by_id:
                    seen.add(root_id)
                    root_id = parent_reply
                else:
                    break
            if root_id not in roots:
                roots[root_id] = [by_id[root_id]] if root_id in by_id else []
            roots[root_id].append(msg)

    for root_id, thread_msgs in roots.items():
        # Deduplicate and sort by date
        seen_ids = set()
        unique = []
        for m in thread_msgs:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                unique.append(m)
                assigned.add(m["id"])
        unique.sort(key=lambda m: m["date"])
        if unique:
            reply_threads.append(unique)

    # Step 2: Group remaining unassigned messages by time proximity
    remaining = [m for m in messages if m["id"] not in assigned]
    time_threads: list[list[dict]] = []

    if remaining:
        current_thread: list[dict] = [remaining[0]]
        for msg in remaining[1:]:
            prev = current_thread[-1]
            time_diff = (msg["date"] - prev["date"]).total_seconds() / 60
            if time_diff <= THREAD_WINDOW_MINUTES:
                current_thread.append(msg)
            else:
                time_threads.append(current_thread)
                current_thread = [msg]
        if current_thread:
            time_threads.append(current_thread)

    return reply_threads + time_threads


def _format_thread_messages(thread: list[dict]) -> str:
    """Format a thread of messages for the LLM prompt."""
    lines = []
    for msg in thread:
        date_str = msg["date"].strftime("%H:%M")
        lines.append(f"[{date_str}] {msg['text']}")
    return "\n".join(lines)


async def extract_from_chat(
    user_client,
    chat_id: int,
    chat_name: str,
    country: str,
) -> int:
    """
    Extract Q&A pairs from unprocessed messages in a chat.

    Args:
        user_client: Telethon user client (for fetching messages)
        chat_id: Telegram chat peer ID
        chat_name: Human-readable name
        country: Country code

    Returns:
        Number of Q&A pairs extracted.
    """
    last_id = get_last_processed_msg_id(chat_id)
    total_extracted = 0
    total_processed = 0
    highest_msg_id = last_id

    log.info(
        "Q&A extraction for %s (country=%s, after msg_id=%d)",
        chat_name, country, last_id,
    )

    while True:
        # Fetch batch of messages newer than last_processed
        raw_messages = []
        try:
            async for msg in user_client.iter_messages(
                chat_id,
                limit=EXTRACTION_BATCH,
                min_id=last_id,
                reverse=True,  # oldest first
            ):
                text = msg.message or ""
                if text.strip():
                    reply_to = None
                    if msg.reply_to and hasattr(msg.reply_to, "reply_to_msg_id"):
                        reply_to = str(msg.reply_to.reply_to_msg_id)
                    raw_messages.append({
                        "id": msg.id,
                        "date": msg.date,
                        "text": text.strip(),
                        "reply_to": reply_to,
                    })
                if msg.id > highest_msg_id:
                    highest_msg_id = msg.id
        except Exception as e:
            log.error("Failed to fetch messages from %s: %s", chat_name, e)
            break

        if not raw_messages:
            break

        total_processed += len(raw_messages)

        # Group into threads
        threads = _group_into_threads(raw_messages)

        # Process threads with >= MIN_THREAD_SIZE messages
        for thread in threads:
            if len(thread) < MIN_THREAD_SIZE:
                continue

            messages_text = _format_thread_messages(thread)
            msg_ids = [str(m["id"]) for m in thread]

            qa_pairs = _call_extraction_llm(
                country, chat_name, messages_text, len(thread),
            )

            for qa in qa_pairs:
                try:
                    merge_qa_pair(
                        country=country,
                        category=qa.get("category", "other"),
                        question=qa.get("question", ""),
                        answer=qa.get("answer", ""),
                        source_msgs=msg_ids,
                        confidence=qa.get("confidence", "medium"),
                        tags=qa.get("tags", []),
                    )
                    total_extracted += 1
                except Exception as e:
                    log.warning("Failed to merge Q&A pair: %s", e)

        # Update last_id for next batch
        last_id = highest_msg_id

    # Log this extraction run
    if total_processed > 0:
        log_extraction_run(
            chat_id=chat_id,
            last_processed_msg_id=highest_msg_id,
            messages_processed=total_processed,
            facts_extracted=total_extracted,
        )

    log.info(
        "Extraction complete for %s: %d messages processed, %d Q&A pairs extracted.",
        chat_name, total_processed, total_extracted,
    )
    return total_extracted
