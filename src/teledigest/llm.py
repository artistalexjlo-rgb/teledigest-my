from __future__ import annotations

import datetime as dt
import json
from typing import Any

from openai import OpenAI

from .config import get_config, log
from .text_sanitize import strip_markdown_fence

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        cfg = get_config()
        _openai_client = OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)
    return _openai_client


def _format_messages_corpus(messages, max_items: int = 500, max_chars: int = 500) -> str:
    lines = []
    for channel, text in messages[:max_items]:
        t = " ".join(text.split())
        if not t:
            continue
        if len(t) > max_chars:
            t = t[:max_chars] + " ..."
        lines.append(f"[{channel}] {t}")
    return "\n".join(lines)


def _format_knowledge_context(knowledge: list[dict[str, Any]]) -> str:
    """Format knowledge entries as context for the digest prompt."""
    if not knowledge:
        return ""
    lines = []
    for k in knowledge[:30]:  # top 30 entries max
        tags = json.loads(k["tags"]) if isinstance(k["tags"], str) else k["tags"]
        lines.append(
            f"[{k['category']}] Q: {k['question']}\n"
            f"A: {k['answer']} (confidence: {k['confidence']})"
        )
    return "\n\n".join(lines)


# Knowledge-aware digest prompts
_DIGEST_SYSTEM = """
Ты — редактор русскоязычного Telegram-канала для экспатов в {COUNTRY}.

У тебя два входа:
1. EXISTING KNOWLEDGE: что мы уже знаем из базы знаний
2. TODAY'S MESSAGES: сырые сообщения из чатов за последние 24 часа

Твоя задача: дайджест с КОНКРЕТИКОЙ. Не пересказ, а полезные ответы.

КРИТИЧЕСКИ ВАЖНО — сохраняй конкретику из сообщений:
- Точные цены: "R$204,77", "150 реалов/месяц", "$50"
- Коды и номера: "код 140120", "CPF", "CRNM"
- Адреса и места: "Receita Federal на Rua da Consolação"
- Сроки: "делают за 15 минут", "ждать 3 недели"
- Пошаговые инструкции: "1. Идёшь туда 2. Берёшь то 3. Платишь столько"
- Имена сервисов: "Wise", "Remessa Online", "iFood"

ПЛОХО: "Спрашивают, нужно ли оплачивать квитанции..."
ХОРОШО: "Для VITEM XIV в PF Кампинаса нужны 2 квитанции: CRNM (код 140120 — R$204,77) и residência (код 140082 — R$168). Платить обе."

Правила:
- Если сегодня повторяют известное — не включай
- Если новая инфа — дай конкретный ответ, как будто объясняешь другу
- Группируй по темам, хештеги в конце секции
- Игнорируй рекламу, спам, приветствия
- Формат: <b>жирный</b>, <i>курсив</i> — только HTML для Telegram
- Никакого Markdown (**, ###, ```)
- Пиши на русском, неформально, по делу
"""

_DIGEST_USER = """
Country: {COUNTRY}

EXISTING KNOWLEDGE (top relevant facts):
{KNOWLEDGE}

TODAY'S MESSAGES ({MSG_COUNT} messages):
{MESSAGES}
"""


def build_prompt(day: dt.date, messages, knowledge: list[dict[str, Any]] | None = None, country: str = ""):
    if not messages:
        return (
            "You are a helpful assistant.",
            f"No messages to summarize for {day.isoformat()}.",
        )

    corpus = _format_messages_corpus(messages)

    # If knowledge context is available, use the knowledge-aware prompt
    if knowledge:
        knowledge_text = _format_knowledge_context(knowledge)
        system = _DIGEST_SYSTEM.format(COUNTRY=country or "unknown")
        user = _DIGEST_USER.format(
            COUNTRY=country or "unknown",
            KNOWLEDGE=knowledge_text,
            MSG_COUNT=len(messages),
            MESSAGES=corpus,
        )
        return system, user

    # Fallback to config-based prompts (backward compatible)
    cfg = get_config()
    system = cfg.llm.system_prompt
    user = cfg.llm.user_prompt.format(
        DAY=day.isoformat(),
        MESSAGES=corpus,
        TIMEZONE=cfg.bot.time_zone,
    )

    return system, user


def llm_summarize_brief(day: dt.date, digest: str) -> str:
    """Call the LLM to produce a compact brief from an already-generated digest."""
    client = _get_client()
    cfg = get_config()
    system = cfg.llm.system_brief_prompt
    user = cfg.llm.user_brief_prompt.format(DAY=day.isoformat(), DIGEST=digest)
    log.info("Calling OpenAI for brief summary...")

    try:
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=cfg.llm.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned no content")
        brief = strip_markdown_fence(content.strip())
        log.info("Received brief summary from OpenAI (%d chars).", len(brief))
        return brief

    except Exception as e:
        log.exception("OpenAI API error for brief: %s", e)
        return f"Failed to generate brief summary.\n\nError: {e}"


def llm_summarize(
    day: dt.date,
    messages,
    knowledge: list[dict[str, Any]] | None = None,
    country: str = "",
) -> str:
    client = _get_client()
    system, user = build_prompt(day, messages, knowledge=knowledge, country=country)
    log.info("Calling OpenAI for summary (%d messages, knowledge=%d)...",
             len(messages), len(knowledge) if knowledge else 0)

    try:
        response = client.chat.completions.create(
            model=get_config().llm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=get_config().llm.temperature,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned no content")
        summary = content.strip()
        summary = strip_markdown_fence(summary)

        log.info("Received summary from OpenAI (%d chars).", len(summary))
        return summary

    except Exception as e:
        log.exception("OpenAI API error: %s", e)
        return f"Failed to generate AI summary for {day.isoformat()}.\n\n" f"Error: {e}"
