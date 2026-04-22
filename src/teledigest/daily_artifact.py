"""
daily_artifact.py — Build a daily understanding artifact from messages.

Reads messages for a given day from the DB, builds reply chains,
extracts conversation spans, and produces provisional claims.
All heuristic — no LLM calls.

Based on Codex's build_daily_memory_artifact.py, ported into the bot.
"""

from __future__ import annotations

import re
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .config import log
from .db import get_db_connection

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

BOT_RE = re.compile(
    r"капч|нажмите на|добро пожаловать в чат|в течение 120 секунд|"
    r"у меня есть информация по данной теме|если ответ неполный|онлайн обмен валюты|"
    r"промокод|подключай интернет|бот-помощник",
    re.I,
)
QUESTION_RE = re.compile(
    r"\?|(?:^|\s)(?:подскаж|как |где |сколько |нужно ли|можно ли|есть ли|"
    r"кто[- ]?нибудь|принимают ли)",
    re.I,
)
USEFUL_RE = re.compile(
    r"(?:смог|смогла|получил|получила|получилось|открыли|не открыли|дают|не дают|"
    r"нужен|не нужен|нужно|не нужно|принимают|не принимают|"
    r"лимит|комис|штраф|действует|можно|нельзя|работает|не работает|"
    r"\b\d+\b|r\$|реал|дней|месяц|час|минут|"
    r"внж|rnm|cpf|спф|апостил|справк|банк|карта|pix|багаж|билет|консуль|"
    r"карторио|такси|uber|автобус)",
    re.I,
)
NOISE_RE = re.compile(r"^(?:спасибо|благодарю|ясно|понял|поняла|ок|угу|ага)$", re.I)

# Subject classification
BANK_RE = re.compile(
    r"\b(?:банк|карт[аеу]|сч[её]т|pix|wise|itau|nubank|bradesco|bb|"
    r"banco do brasil|inter|c6|santander|99)\b", re.I,
)
DOC_RE = re.compile(
    r"\b(?:внж|rnm|cpf|спф|апостил|справк|карторио|консуль|свидетельств|перевод)\b",
    re.I,
)
VISA_RE = re.compile(
    r"\b(?:виза|безвиз|90\s*дн|180\s*дн|турист|гражданств|натурализац|бежен)\b",
    re.I,
)
FLIGHT_RE = re.compile(
    r"\b(?:багаж|рейс|пересад|turkish|ethiopian|qatar|royal air maroc|билет|авиак)\b",
    re.I,
)
TRANSPORT_RE = re.compile(
    r"\b(?:такси|uber|автобус|метро|поезд|машин|аренда авто)\b", re.I,
)
HOUSING_RE = re.compile(
    r"\b(?:квартир|аренд|кондо|condo|condominio|quinto andar|жиль)\b", re.I,
)
FOOD_RE = re.compile(
    r"\b(?:супермаркет|магазин|продукт|еда|кухн|ресторан|кафе|доставк|ifood|"
    r"rappi|mercado|feira|хлеб|молок|мяс|рыб|фрукт|овощ|готови|рецепт|"
    r"pão de açúcar|assaí|carrefour|extra|atacadão)\b", re.I,
)
SHOPPING_RE = re.compile(
    r"\b(?:магазин|шоппинг|торгов|mall|shopping|купить|покупк|одежд|обувь|"
    r"техник|электрон|aliexpress|mercado livre|shopee|amazon)\b", re.I,
)

# Subject -> bot category mapping
SUBJECT_TO_CATEGORY = {
    "banking_access": "finance",
    "document_requirements": "documents",
    "tourist_stay_rules": "visa",
    "flight_rules": "transport",
    "transport_rules": "transport",
    "housing_rules": "housing",
    "food_grocery": "food",
    "shopping": "shopping",
    "general_practical_info": "other",
}

# Auto-tag patterns
TAG_PATTERNS = [
    (re.compile(r"\b(?:банк|nubank|itau|bradesco|inter|c6|santander)\b", re.I), "банк"),
    (re.compile(r"\bpix\b", re.I), "pix"),
    (re.compile(r"\bwise\b", re.I), "wise"),
    (re.compile(r"\b(?:карт[аеу]|картой|карты)\b", re.I), "карта"),
    (re.compile(r"\b(?:cpf|спф)\b", re.I), "cpf"),
    (re.compile(r"\b(?:внж|rnm|crnm)\b", re.I), "внж"),
    (re.compile(r"\b(?:апостил\w*)\b", re.I), "апостиль"),
    (re.compile(r"\b(?:справк\w*)\b", re.I), "справка"),
    (re.compile(r"\b(?:консуль\w*)\b", re.I), "консульство"),
    (re.compile(r"\b(?:карторио)\b", re.I), "карторио"),
    (re.compile(r"\b(?:паспорт\w*)\b", re.I), "паспорт"),
    (re.compile(r"\b(?:виза|визу|визой|визы)\b", re.I), "виза"),
    (re.compile(r"\b(?:гражданств\w*|натурализац\w*)\b", re.I), "гражданство"),
    (re.compile(r"\b(?:рейс\w*|перел[её]т\w*)\b", re.I), "рейс"),
    (re.compile(r"\b(?:билет\w*)\b", re.I), "билет"),
    (re.compile(r"\b(?:багаж\w*)\b", re.I), "багаж"),
    (re.compile(r"\b(?:пересад\w*)\b", re.I), "пересадка"),
    (re.compile(r"\b(?:такси|uber|убер)\b", re.I), "такси"),
    (re.compile(r"\b(?:автобус\w*|метро)\b", re.I), "транспорт"),
    (re.compile(r"\b(?:аренд\w*|квартир\w*|жиль\w*)\b", re.I), "жильё"),
    (re.compile(r"\b(?:сим|vivo|tim|claro)\b", re.I), "связь"),
    (re.compile(r"\b(?:супермаркет\w*|mercado|feira)\b", re.I), "супермаркет"),
    (re.compile(r"\b(?:pão de açúcar|assaí|carrefour|extra|atacadão)\b", re.I), "супермаркет"),
    (re.compile(r"\b(?:ресторан\w*|кафе|ifood|rappi)\b", re.I), "еда"),
    (re.compile(r"\b(?:продукт\w*|еда|готовк\w*)\b", re.I), "продукты"),
    (re.compile(r"\b(?:mercado livre|shopee|aliexpress|amazon)\b", re.I), "маркетплейс"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Msg:
    id: str
    channel: str
    date: str
    text: str
    reply_to: str | None


@dataclass
class Chain:
    root_id: str
    messages: list[Msg] = field(default_factory=list)


@dataclass
class Claim:
    """A single extracted claim — either qa or fact."""
    claim_type: str  # "qa" or "fact"
    subject: str
    category: str
    question: str
    answer: str
    source_msgs: list[str]
    tags: list[str]
    channel: str
    first_seen: str
    last_seen: str
    chain_size: int


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").replace("\r", " ").split())


def _fetch_day_messages(day: dt.date, channels: list[str] | None = None) -> list[Msg]:
    """Fetch messages for a day from the DB, filtering bot noise."""
    day_str = day.isoformat()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, channel, date, text, reply_to_msg_id
            FROM messages
            WHERE substr(date, 1, 10) = ?
              AND (is_bot = 0 OR is_bot IS NULL)
            ORDER BY date ASC
            """,
            (day_str,),
        )
        rows = cur.fetchall()

    result = []
    for row in rows:
        text = _normalize(row[3] or "")
        if not text or BOT_RE.search(text):
            continue
        if channels and row[1] not in channels:
            continue
        result.append(Msg(row[0], row[1], row[2], text, row[4]))
    return result


def _build_chains(messages: list[Msg]) -> list[Chain]:
    """Group messages into reply chains."""
    chains: dict[str, Chain] = {}
    msg_to_root: dict[str, str] = {}
    for msg in messages:
        parent_root = msg_to_root.get(msg.reply_to) if msg.reply_to else None
        root_id = parent_root or msg.id
        chains.setdefault(root_id, Chain(root_id=root_id)).messages.append(msg)
        msg_to_root[msg.id] = root_id
    return list(chains.values())


def _looks_question(text: str) -> bool:
    return bool(QUESTION_RE.search(text))


def _looks_useful(text: str) -> bool:
    return bool(USEFUL_RE.search(text))


def _classify_subject(text: str) -> str:
    if BANK_RE.search(text):
        return "banking_access"
    if DOC_RE.search(text):
        return "document_requirements"
    if VISA_RE.search(text):
        return "tourist_stay_rules"
    if FLIGHT_RE.search(text):
        return "flight_rules"
    if TRANSPORT_RE.search(text):
        return "transport_rules"
    if HOUSING_RE.search(text):
        return "housing_rules"
    if FOOD_RE.search(text):
        return "food_grocery"
    if SHOPPING_RE.search(text):
        return "shopping"
    return "general_practical_info"


def _auto_tags(text: str) -> list[str]:
    """Extract tags from text using keyword patterns."""
    tags = set()
    for pattern, tag in TAG_PATTERNS:
        if pattern.search(text):
            tags.add(tag)
    return sorted(tags)


def _extract_spans(chain: Chain) -> list[list[Msg]]:
    """Split a chain into conversation spans — sub-conversations with useful replies."""
    msgs = chain.messages
    if len(msgs) < 2:
        return []

    spans: list[list[Msg]] = []
    current = [msgs[0]]
    useful_count = 0
    trailing_noise = 0

    for msg in msgs[1:]:
        text = msg.text
        current.append(msg)

        if _looks_useful(text):
            useful_count += 1
            trailing_noise = 0
        elif NOISE_RE.search(text):
            trailing_noise += 1
        elif _looks_question(text) and useful_count >= 1 and len(current) >= 3:
            spans.append(current[:-1])
            current = [msg]
            useful_count = 0
            trailing_noise = 0
            continue
        else:
            trailing_noise += 1

        if trailing_noise >= 2 and useful_count >= 1:
            spans.append(current[:-1])
            current = [msgs[0]]
            useful_count = 0
            trailing_noise = 0

    if len(current) >= 2 and useful_count >= 1:
        spans.append(current)

    # Filter: require at least one useful reply
    cleaned = []
    for span in spans:
        if len(span) < 2:
            continue
        if sum(1 for m in span[1:] if _looks_useful(m.text)) == 0:
            continue
        cleaned.append(span)
    return cleaned


def _claims_from_span(span: list[Msg]) -> list[Claim]:
    """Extract provisional claims from a conversation span."""
    claims: list[Claim] = []
    root = span[0]
    full_text = " ".join(m.text for m in span)
    subject = _classify_subject(full_text)
    category = SUBJECT_TO_CATEGORY.get(subject, "other")
    tags = _auto_tags(full_text)

    useful_answers = [m for m in span[1:] if _looks_useful(m.text)]
    if not useful_answers:
        return claims

    source_ids = [m.id for m in span[:5]]
    first_seen = span[0].date
    last_seen = span[-1].date
    channel = root.channel
    chain_size = len(span)

    # Q&A claim (if root is a question)
    if _looks_question(root.text):
        best_answers = [m.text for m in useful_answers[:2]]
        claims.append(Claim(
            claim_type="qa",
            subject=subject,
            category=category,
            question=root.text,
            answer=" ".join(best_answers),
            source_msgs=source_ids,
            tags=tags,
            channel=channel,
            first_seen=first_seen,
            last_seen=last_seen,
            chain_size=chain_size,
        ))

    # Fact claims (standalone useful answers)
    for answer_msg in useful_answers[:3]:
        claims.append(Claim(
            claim_type="fact",
            subject=subject,
            category=category,
            question=root.text if _looks_question(root.text) else "",
            answer=answer_msg.text,
            source_msgs=source_ids,
            tags=tags,
            channel=channel,
            first_seen=first_seen,
            last_seen=last_seen,
            chain_size=chain_size,
        ))

    return claims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_daily_artifact(
    day: dt.date,
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a daily understanding artifact for a given day.

    Returns dict with keys:
        day, messages_count, chains_count, spans_count, claims_count, claims
    """
    messages = _fetch_day_messages(day, channels)
    chains = _build_chains(messages)

    all_claims: list[Claim] = []
    spans_count = 0

    for chain in chains:
        spans = _extract_spans(chain)
        spans_count += len(spans)
        for span in spans:
            all_claims.extend(_claims_from_span(span))

    log.info(
        "Daily artifact for %s: %d messages, %d chains, %d spans, %d claims",
        day.isoformat(), len(messages), len(chains), spans_count, len(all_claims),
    )

    return {
        "day": day.isoformat(),
        "messages_count": len(messages),
        "chains_count": len(chains),
        "spans_count": spans_count,
        "claims_count": len(all_claims),
        "claims": all_claims,
    }


def artifact_claims_as_dicts(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert claims from artifact to dicts suitable for merge_qa_pair()."""
    result = []
    for claim in artifact.get("claims", []):
        if isinstance(claim, Claim):
            result.append({
                "country": "",  # filled by caller
                "category": claim.category,
                "question": claim.question,
                "answer": claim.answer,
                "source_msgs": claim.source_msgs,
                "tags": claim.tags,
                "confidence": "high" if claim.chain_size >= 3 else "medium",
            })
        else:
            result.append(claim)
    return result
