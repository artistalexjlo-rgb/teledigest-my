"""
knowledge_search.py — МОЗГ bot: knowledge base search + LLM synthesis.

Flow:
1. User asks "МОЗГ <question>"
2. Search knowledge DB for relevant entries (wider net, up to 10)
3. Feed entries as context to DeepSeek
4. DeepSeek synthesizes a concrete answer
5. Return to user
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from .config import get_config, log
from .knowledge_db import search_knowledge


_BRAIN_SYSTEM = """
Ты — бот «МОЗГ» в чате экспатов. Отвечаешь на вопросы по базе знаний сообщества.

Тебе дают вопрос и набор фактов. Многие факты нерелевантны — игнорируй их.
Собери ответ ТОЛЬКО из тех фактов, которые реально отвечают на вопрос.

Правила:
- Конкретика: цены, названия, адреса, сроки
- Кратко: 3-7 предложений максимум
- Если есть разные мнения — укажи оба
- Не ссылайся на номера фактов ("Факт 3") — пользователь их не видит
- Не пиши "точной информации нет" если можешь собрать ответ из фактов
- Если реально ничего полезного нет — скажи "в базе пока нет информации по этому вопросу"
- Простой текст, без Markdown, без форматирования
- Русский язык, неформально
- НЕ выдумывай
"""

_BRAIN_USER = """
Вопрос пользователя: {QUESTION}

Факты из базы знаний ({COUNT} записей):

{FACTS}
"""


def _format_facts_for_llm(results: list[dict]) -> str:
    """Format knowledge entries as context for LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        question = (r.get("question") or "").strip()
        answer = (r.get("answer") or "").strip()
        confidence = r.get("confidence", "medium")
        tags = json.loads(r["tags"]) if isinstance(r["tags"], str) else r.get("tags", [])

        header = f"[Факт {i}, {confidence}]"
        if question:
            parts.append(f"{header}\nВопрос: {question}\nОтвет: {answer}")
        else:
            parts.append(f"{header}\n{answer}")

    return "\n\n".join(parts)


def _synthesize_answer(query: str, results: list[dict]) -> str:
    """Call LLM to synthesize an answer from knowledge entries."""
    cfg = get_config()
    try:
        client = OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)
        facts_text = _format_facts_for_llm(results)

        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": _BRAIN_SYSTEM},
                {"role": "user", "content": _BRAIN_USER.format(
                    QUESTION=query,
                    COUNT=len(results),
                    FACTS=facts_text,
                )},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        content = response.choices[0].message.content
        if not content:
            return ""
        return content.strip()

    except Exception as e:
        log.error("МОЗГ LLM synthesis failed: %s", e)
        return ""


def get_bot_name() -> str:
    """Get the configurable bot trigger name."""
    cfg = get_config()
    return getattr(cfg.bot, "bot_name", "МОЗГ").upper()


def is_brain_query(text: str) -> str | None:
    """
    Check if the message is a МОЗГ query.

    Returns the query text (without the bot name) if it matches,
    or None if it doesn't.
    """
    if not text:
        return None

    bot_name = get_bot_name()
    pattern = re.compile(
        rf"^\s*{re.escape(bot_name)}\s+(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(text)
    if m:
        return m.group(1).strip()
    return None


def search_and_format(country: str, query: str) -> str:
    """
    Search knowledge base, synthesize answer via LLM.

    1. Search for up to 10 relevant entries (wide net for context)
    2. If entries found — synthesize via DeepSeek
    3. If nothing found — return "не знаю"
    """
    results = search_knowledge(country, query, limit=20)

    if not results:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате — "
            "кто-нибудь точно подскажет!"
        )

    log.info("МОЗГ: found %d entries for '%s', synthesizing...", len(results), query[:50])

    answer = _synthesize_answer(query, results)

    if not answer:
        # LLM failed — fallback to raw results
        parts = []
        for r in results[:3]:
            a = (r.get("answer") or "")[:300]
            parts.append(f"• {a}")
        return f"🧠 По запросу «{query}»:\n\n" + "\n\n".join(parts)

    sources = len(results)
    return f"🧠 {answer}\n\n<i>На основе {sources} записей из базы знаний</i>"
