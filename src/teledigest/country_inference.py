"""
country_inference.py — Infer a country from a free-text user query.

Why:
    МОЗГ retrieval depends on `country` to filter wisdom_base. But chats
    are not always 1-1 with one country (e.g. luky_channel is the general
    chat — mapped to one country, but users ask about anything). When a
    user types "как получить CPF" in the AR-tagged chat, the retrieval
    sends country=ar to Firestore and finds nothing — CPF is Brazilian.

How:
    Hardcoded map of high-confidence terms → country. Scan the query
    case-insensitively, count term hits per country, pick the winner.
    Ties resolved by the order of the terms (rare in practice). If no
    term matches, return None and caller keeps the chat-derived country.

Scope:
    - Bureaucratic / financial / agency acronyms that are unambiguously
      country-specific (CPF, CUIT, NIE, etc.)
    - Brand names of payment systems / apps tightly tied to one country
      (PIX = Brazil, iFood = Brazil, Mercado Libre = Argentina, Zalo =
      Vietnam, Vivo = Brazil)
    - Major place names where ambiguity is low (Buenos Aires = ar, Bali = id,
      Colombo = lk, Nha Trang = vn, Curitiba = br)
    - Russian transliterations of the above ("СПФ", "ПИКС", "БУЭНОС-АЙРЕС")

Out of scope (handled differently):
    - DNI (used in both ES and AR — needs disambiguation by other terms)
    - "виза", "паспорт", "банк" — generic, country comes from chat
    - Anything that needs LLM-level understanding — that's wiki-fallback
      territory, not this stage.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable

from .config import log


# Map of country (iso2 lowercase) → set of high-confidence terms.
# Terms are matched as whole-word, case-insensitive. Russian and English
# variants are listed side by side for the common ones.
#
# Adding new terms:
# - Only if they're UNAMBIGUOUS for one country across our active list.
# - For terms used in multiple countries (DNI = ES & AR), leave them out
#   and let the chat-derived country dominate. Better silent than wrong.
COUNTRY_TERMS: Dict[str, set[str]] = {
    "br": {
        # Tax IDs / documents
        "cpf", "спф", "цпф",
        "cnh", "цнх",
        "rg", "ргб",
        # Payment systems / brands unique to Brazil
        "pix", "пикс", "пих",
        "boleto", "болето",
        "ifood", "айфуд", "ифуд",
        "mercadolivre", "mercado livre",  # Brazilian variant
        "rappi",  # also Colombia/Mexico, but very common in BR
        # Government services
        "sus", "сус",   # healthcare system, distinctive
        "encceja",
        "receita federal",
        # Cities / regions
        "rio de janeiro", "сан-паулу", "são paulo", "salvador",
        "fortaleza", "curitiba", "florianópolis", "куритиба",
        # Currency / words
        "real",  # tricky — "real" is generic English, only count if context
        "reais",
        # Companies
        "vivo",  # carrier
        "tim brasil",
    },
    "ar": {
        # Tax IDs / documents
        "cuit", "куит",
        "cuil", "куил",
        # Payment systems / brands unique to Argentina
        "mercadopago", "mercado pago",
        "mercadolibre", "mercado libre",
        # Government
        "anses", "afip",
        # Cities / regions
        "buenos aires", "буэнос-айрес", "буэнос айрес",
        "bariloche", "барилоче",
        "mendoza", "мендоса",
        "ushuaia", "ушуая",
        "rosario", "росарио",
        "córdoba",  # also Spain, but more often AR in expat context
        # Words
        "porteño", "porteno",
        "colectivo",  # bus
        "kiosco",  # shop
    },
    "id": {
        # Government
        "kitas", "китас",
        "kitap", "китап",
        "nik",  # national ID
        "evoa", "e-voa", "voa",
        # Companies
        "gojek", "годжек",
        "grab",   # but also other countries — keep, common
        "tokopedia",
        "shopee",  # also other SEA — but tightly Indonesian context
        "telkomsel",
        # Cities / regions
        "bali", "бали",
        "jakarta", "джакарта",
        "yogyakarta", "yogya",
        "ubud", "убуд",
        "canggu", "чангу",
        "seminyak",
        "kuta",
        "denpasar",
        "lombok",
        "java",  # tricky — programming language, but in travel context fine
    },
    "vn": {
        # Companies
        "zalo", "зало",
        "viettel",
        "vietcombank",
        "agribank",
        "vingroup", "vinpearl",
        "vinfast",
        "vnd",   # currency code
        # Cities
        "hanoi", "ханой",
        "ho chi minh", "сайгон", "saigon",
        "danang", "da nang", "дананг",
        "nha trang", "нячанг", "нья чанг",
        "hoi an", "хой ан",
        "phu quoc", "фукуок",
        "halong", "ha long", "халонг",
        "cat ba", "кат ба",
        "cam ranh",
        "dalat", "da lat", "далат",
    },
    "lk": {
        "lkr",   # currency
        # Cities / regions
        "colombo", "коломбо",
        "kandy", "канди",
        "galle", "галле",
        "ella",  # tricky — name, also Spanish word; in travel context fine
        "negombo",
        "mirissa",
        "unawatuna",
        "weligama",
        "hikkaduwa",
        "sigiriya",
        "trincomalee",
        "nuwara eliya",
        "bentota",
        "anuradhapura",
        "polonnaruwa",
        # Visa
        "eta",  # widely used for Sri Lanka travel — but also Canada eTA. Keep
                # — most chat context for us is LK
    },
    "mu": {
        "mauritius", "маврикий",
        "port louis", "порт-луи",
        "flic-en-flac", "flic en flac", "флик-ан-флак",
        "grand baie", "гранд-бэ", "grand-baie",
        "trou aux biches",
        "casela",
        "mcb",  # Mauritius Commercial Bank
        "mur",  # currency
    },
    "at": {
        "вена", "vienna",
        "salzburg", "зальцбург",
        "innsbruck", "инсбрук",
        "graz", "грац",
        "linz", "линц",
        "hofer",  # supermarket chain
        "billa",  # also Czech/Slovak — but in Austrian expat context fine
        "anmeldung",  # also Germany — but rare in conv
        "aufenthaltsbewilligung",
        "evisitor",
    },
    "be": {
        "брюссель", "brussels",
        "antwerp", "антверпен",
        "brugge", "брюгге", "bruges",
        "ghent", "гент",
        "liege", "liège",
        "single permit",   # Belgian specific in expat context
        "nrn",   # national register number
        "ej",    # Belgian language certs sometimes
    },
    "bg": {
        "софия", "sofia",
        "пловдив", "plovdiv",
        "varna", "варна",
        "burgas", "бургас",
        "tokuda",  # Tokuda Bank
        "lev",     # currency
        "leva",
    },
    "de": {
        "берлин", "berlin",
        "мюнхен", "munich", "münchen",
        "frankfurt", "франкфурт",
        "hamburg", "гамбург",
        "schufa",
        "anmeldung",
        "kindergeld",
        "elterngeld",
        "krankenkasse",
        "kleinanzeigen",  # eBay Kleinanzeigen
        "packstation",
    },
    "fr": {
        "париж", "paris",
        "lyon", "лион",
        "marseille", "марсель",
        "nice", "ницца",
        "carte vitale",
        "cpam",
        "ofii",
        "préfecture", "prefecture",
        "ribourg", "ribourgeoise",
    },
    "ph": {
        "manila", "манила",
        "cebu", "себу",
        "boracay", "боракай",
        "palawan", "палаван",
        "tin id", "tin number",
        "sss",   # social security
        "philhealth",
        "globe", "smart",  # carriers — tricky, also generic
    },
}


# Pre-compile regex per country to a single pattern for fast scan.
# Word boundary is \b, but in Russian \b doesn't behave well — we use
# lookaround for non-letter characters.
_TERM_PATTERNS: Dict[str, re.Pattern[str]] = {}


def _is_cyrillic(text: str) -> bool:
    """True if string contains any Cyrillic letter."""
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def _build_patterns() -> None:
    """Build regex patterns once. Called lazily.

    Cyrillic-containing terms get a permissive trailing suffix `[а-яё]*`
    to absorb Russian grammatical endings (Буэнос-Айрес → Буэнос-Айресе).
    Latin terms keep strict word-boundary on both sides to avoid false
    positives like "cuit" matching "circuit".
    """
    if _TERM_PATTERNS:
        return
    for country, terms in COUNTRY_TERMS.items():
        # Sort by length DESC so multi-word terms match before partial substrings.
        sorted_terms = sorted(terms, key=len, reverse=True)
        alternatives = []
        for t in sorted_terms:
            esc = re.escape(t)
            if _is_cyrillic(t):
                # Leading boundary + term + optional Russian case suffix.
                alternatives.append(rf"(?<![\w]){esc}[а-яё]*(?![\w])")
            else:
                # Strict boundary on both sides — Latin terms.
                alternatives.append(rf"(?<![\w]){esc}(?![\w])")
        pattern = "|".join(alternatives)
        _TERM_PATTERNS[country] = re.compile(pattern, re.IGNORECASE | re.UNICODE)


def infer_country(query: str) -> str | None:
    """
    Scan a user query and return the country code (iso2 lowercase) of the
    country whose terms appear most often. Returns None if no terms match.

    Examples:
        "как получить CPF" -> "br"
        "пикс на 100 реалов" -> "br" (pix + реалов both br)
        "виза в Австрию"   -> "at" (Австрию translit-? no, Austria not in
                                    set — only Vienna etc. — returns None.
                                    Generic word "виза" — out of scope.)
        "DNI в Аргентине"  -> "ar" (Аргентине? no — but "Аргентине" not in
                                    terms. Just "Buenos Aires" / "куит".)

    Note: this is intentionally conservative. If no STRONG signal, return
    None and let the chat-derived country win.
    """
    if not query or not query.strip():
        return None

    _build_patterns()
    hits: Dict[str, int] = {}
    for country, pat in _TERM_PATTERNS.items():
        matches = pat.findall(query)
        if matches:
            hits[country] = len(matches)

    if not hits:
        return None

    # Pick country with most matches; tie-break by first-found in iteration order.
    best = max(hits.items(), key=lambda kv: kv[1])
    return best[0]


def infer_country_with_log(query: str, chat_country: str) -> tuple[str, bool]:
    """
    Same as infer_country, but returns (final_country, was_overridden) and
    logs the decision. Use this from gemini_brain to make audit trail clear.
    """
    inferred = infer_country(query)
    if inferred and inferred != chat_country:
        log.info(
            "МОЗГ country override: chat=%s -> inferred=%s (from query: %r)",
            chat_country, inferred, query[:60],
        )
        return inferred, True
    return chat_country, False
