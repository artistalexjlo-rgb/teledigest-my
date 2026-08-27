# -*- coding: utf-8 -*-
"""ЧТЕНИЕ КОРПУСА: мухи гео из базы бота. Ключей не тратит, ртов не содержит.

⭐ ЗАЧЕМ ОТДЕЛЬНЫМ МОДУЛЕМ (2026-08-24, заказ юзера). Новому тракту (§0.19) из старого кода
нужны ровно три вещи, и все они — чтение: мухи гео, фильтр junk и разбор колонки `country`.
Пока они жили внутри `facet.py` на 1 292 строки, каждое обращение к ним открывало файл, где
рядом лежат промпты и приёмы ОТМЕНЁННОЙ схемы — и я их оттуда переносил в новый код по
привычке (пачки по 90 вместо одного вызова, вторая транслитерация рядом со `slugs.py`).

Разведение по файлам — не украшение: пока живое и мёртвое в одном файле, «не смотреть на
мёртвое» держится на моей памяти, а она не держит.

`facet.py` теперь берёт эти же имена отсюда — второй копии правила нет.
"""

import re
import sqlite3

from country_codes import COUNTRIES  # справочник кодов — единственный источник правды

DB = "/home/teledigest/data/messages_fts.db"
MIN_LEN = 140
ANY_GEO = "any"  # легальное значение колонки: совет не привязан к стране

# Защита билдера: тот же junk-инвариант, что в гейте extraction.py (мухолов пересказал
# ЗАПРОС «user хочет X» / «Information on X» вместо факта). Qdrant почищен пёржем, но
# билдер тег-читает СЫРОЙ SQLite → фильтруем на входе, иначе шелфим наррации-мусор.
_JUNK = re.compile(
    r"\b(?:"
    r"User (?:is asking|is looking (?:for|to)|wants to know|needs to know|is seeking|"
    r"is inquiring|is requesting|asks|inquired|wants information)|"
    r"A user (?:is asking|asks|wants|is looking|inquired)|"
    r"Inquir(?:y|ies) (?:about|is|are)|Clarification (?:needed|is needed)|"
    r"not (?:explicitly )?provided in the log|is not provided in the log|"
    r"not specified in the log|A request for assistance|request for assistance in|"
    r"is available for rent|(?:room|apartment|flat) is available|"
    r"consult with .{0,40}Telegram|via their Telegram channel|should be researched"
    r")\b",
    re.I,
)
_OPENER = re.compile(
    r"^\s*(?:Information (?:on|about|regarding)|"
    r"Provide (?:information|instructions|details|guidance|an overview)|"
    r"Details (?:on|about|regarding)|Inquir(?:y|ies|ing)\b|Seeking\b|Looking for\b|"
    r"Request(?:ing)? (?:for|information)|Question(?:s)? (?:about|regarding|on)|"
    r"(?:The |A )?[Uu]ser (?:is|wants|needs|seeks|asks)|"
    r"Guidance (?:on|is)|Advice (?:is )?(?:sought|requested))"
)


def is_junk(t):
    return bool(t and (_JUNK.search(t) or _OPENER.match(t)))


def geo_codes(raw):
    """Значение колонки `country` → МНОЖЕСТВО кодов. Схему не меняем: колонка одна, но
    значение может быть перечислением («de, ru»).

    ⭐ Корень (замер по базе): у мухи ОДНА колонка и сравнение на равенство, а промпт давал
    бинарный выбор «одна страна или any». Совет про две страны выразить было нечем: он падал
    либо в `any` (3 077 мух), либо в мусорный ключ с запятой — таких 29 (`de, ru` 5, `ru, kg`
    4, `kg, kz, ru` 2, `au, nz` 2 …), и эти мухи не попадали НИ В ОДНУ страну. Оба исхода
    видели живьём.

    Нормализация по единственному источнику правды — справочнику стран: нижний регистр,
    срезать пробелы, неизвестное ОТБРОСИТЬ. Отсюда же исчезают гео-призраки: код, которого нет
    в справочнике, страницей больше не станет.
    """
    out = set()
    for part in (raw or "").replace(";", ",").replace("/", ",").split(","):
        c = part.strip().lower()
        if c == ANY_GEO or c in COUNTRIES:
            out.add(c)
    return out


def load_flies(geo, limit=None, exclude=None):
    """Мухи гео (junk отсеян), ИСКЛЮЧАЯ уже тегнутые (exclude=set id) → следующая порция.
    limit = сколько НОВЫХ мух за прогон (бережное потребление, не весь корпус разом).

    Гео берётся по ВХОЖДЕНИЮ кода в список: `de, ru` попадает и в Германию, и в Россию.
    `LIKE` — только грубый пред-отбор, чтобы не тащить 23 924 текста на каждый вызов; точное
    решение принимает `geo_codes`, иначе `an` цеплял бы `any`, а `ru` — любое значение с
    этими буквами.
    """
    exclude = exclude or set()
    geo = (geo or "").strip().lower()
    m = sqlite3.connect(DB)
    rows = m.execute(
        "SELECT id, ai_lesson, country FROM extracted_patterns "
        "WHERE ai_lesson IS NOT NULL AND length(ai_lesson)>? "
        "AND (country = ? OR country LIKE '%' || ? || '%') ORDER BY id",
        (MIN_LEN, geo, geo),
    ).fetchall()
    m.close()
    rows = [
        (r[0], r[1])
        for r in rows
        if geo in geo_codes(r[2]) and not is_junk(r[1]) and r[0] not in exclude
    ]
    return rows[:limit] if limit else rows
