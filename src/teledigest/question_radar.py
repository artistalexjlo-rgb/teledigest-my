"""
question_radar.py — detect translator-related questions in incoming messages
and notify the admin channel with a deep link, so the user can manually jump
into the source chat and offer help.

Pure detection + notify, no auto-reply. False positives are fine — the admin
filters by hand.
"""

from __future__ import annotations

import re

import requests as _req

from .config import get_config, log

# Translator-related keywords (Russian + English). Loose by design — we want
# recall over precision, since the admin reviews manually before answering.
_KEYWORDS = re.compile(
    r"\b("
    r"переводчик\w*|переводчиц\w*|перевод\w*|переведит\w*|переведи|"
    r"translator\w*|translat\w*|interpret\w*"
    r")\b",
    re.IGNORECASE,
)

# Interrogative markers — a keyword alone is not enough (people just say
# "пользуюсь переводчиком" without asking). Need a question signal.
_QUESTION_HINTS = re.compile(
    r"\?|"
    r"\b(как|какой|каким|какую|какие|где|чем|кто|посоветуй\w*|"
    r"подскаж\w*|нужен|нужна|нужно|посоветуете|кому|"
    r"recommend\w*|suggest\w*|best|which|where|how)\b",
    re.IGNORECASE,
)


def is_translator_question(text: str) -> bool:
    """True if `text` looks like a question about translators."""
    if not text or len(text) < 5:
        return False
    if not _KEYWORDS.search(text):
        return False
    return bool(_QUESTION_HINTS.search(text))


def _message_link(chat_id: int, chat_name: str, msg_id: int) -> str:
    """Build a deep link to a Telegram message.

    Public chat (chat_name is a username): https://t.me/<username>/<msg_id>
    Private supergroup (chat_id like -100xxxxx): https://t.me/c/<stripped>/<msg_id>
    """
    # chat_name is numeric string only when we couldn't resolve a username.
    if chat_name and not chat_name.lstrip("-").isdigit():
        return f"https://t.me/{chat_name}/{msg_id}"
    # Strip -100 prefix for supergroup deep-links.
    stripped = str(chat_id).lstrip("-")
    if stripped.startswith("100"):
        stripped = stripped[3:]
    return f"https://t.me/c/{stripped}/{msg_id}"


def notify_translator_question(
    chat_id: int, chat_name: str, msg_id: int, text: str, country: str | None
) -> None:
    """Best-effort send to summary_target with a link + the message text."""
    try:
        cfg = get_config()
        token = cfg.telegram.bot_token
        target = cfg.bot.summary_target
        if not token or not target:
            return
        link = _message_link(chat_id, chat_name, msg_id)
        snippet = text if len(text) <= 400 else text[:400] + "…"
        body = (
            f"🎯 <b>Вопрос про переводчик</b>"
            f"{' · ' + country.upper() if country else ''}\n"
            f"<i>{chat_name}</i>\n\n"
            f"{snippet}\n\n"
            f'<a href="{link}">→ перейти в чат</a>'
        )
        _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": target,
                "text": body,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception as e:
        log.warning("notify_translator_question failed: %s", e)
