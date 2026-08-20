"""
facet.py — билдер по ФАСЕТАМ (BUILDER_RULES §0.1/§0.2 + роли=мн.ч. = мульти-лейбл).

Муха = готовый продукт. Роли у неё НЕСКОЛЬКО (фасеты), не одна. Билдер:
  1. КАЖДУЮ муху переводит дословно (LLM = переводчик, не автор);
  2. вычитывает из её текста МУЛЬТИ-ФАСЕТЫ: задачи[] · сущности[{имя, роль}] · место · условие
     (фасеты латентны в ai_lesson — мухолов их пока не отдаёт отдельным полем);
  3. строит ВИДЫ по фасету (инвертированный индекс): страница = все мухи, у кого есть этот
     фасет; одна муха живёт во ВСЕХ своих видах, в своей роли там. Не один ящик → мис-сорт
     как класс исчезает (Turkish-посадка = {въезд, перелёт} — в обоих видах).

НЕТ cosine-argmax, НЕТ single-bucket, НЕТ MIN_PAGE, НЕТ synth, НЕТ грунт-гейта, НЕТ человека.
Ошибка тега грациозна: пропущен — нет в одном виде; лишний — лишний вид. Не катастрофа
single-bucket'а (там один неверный выбор = не та страница), поэтому проверяющий не нужен.

Роль сущности ∈ {цель, требование, обход, обстоятельство}.
Задачи именуются из текста мухи → пасс консолидации сводит ярлыки к одному грайну (не косинус).

Запуск (VPS): cd /root/pseo_builder && /root/embed_ab/venv/bin/python facet.py <geo> [--limit N]
Плуминг Gemini (пейсинг/квота/429/IPv4) — внутри keybroker.call (сосок мозга). build.py снесён.
"""

import json
import os
import re
import sqlite3
import sys

import tail_taxonomy as tax
from country_codes import COUNTRIES  # справочник кодов — единственный источник правды
from keybroker import call

DB = "/home/teledigest/data/messages_fts.db"
MIN_LEN = 140
ROLES = ("цель", "требование", "обход", "обстоятельство")

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


# Окно экстракции УБРАНО: коэкзистенцию с экстрактором держит мозг (резерв 60/ключ +
# per-ключ шаг + abuse-пауза), временнОе разделение не нужно. Мозг — единственный раздатчик.


FACET_SYS = (
    "Ты РАЗМЕТЧИК готового совета (мухи) по фасетам, НЕ автор. Муху НЕ переписывай, НЕ дополняй, "
    "НЕ сокращай, НЕ обобщай.\n"
    "Верни СТРОГО JSON с полями:\n"
    '  "perevod"  — дословный перевод мухи на русский: ВСЕ факты/числа/названия/условия/оговорки '
    "как есть, ничего не добавить и не выкинуть, естественный русский (не калька).\n"
    '  "zadachi"  — СПИСОК задач/тем, которых касается совет (МОЖЕТ БЫТЬ НЕСКОЛЬКО, это ключ). '
    'Коротко, из текста мухи. Пример: ["получение CPF"] или ["покупка билета на автобус", '
    '"обход требования CPF"]. Не выдумывай задач, которых в мухе нет.\n'
    '  "sushnosti" — СПИСОК [{"imya","rol"}]: конкретные сущности (CPF, ВНЖ, Vivo, Correios, Busbud, '
    "Рио…) и роль каждой в совете. rol ∈ {цель, требование, обход, обстоятельство}: цель — за ней "
    "пришли; требование — без неё не сделать задачу; обход — способ без неё; обстоятельство — просто "
    'упомянута. Пример: [{"imya":"CPF","rol":"требование"},{"imya":"Busbud","rol":"обход"}].\n'
    '  "mesto"    — город/страна, если совет привязан к месту, иначе null (пример: "Рио-де-Жанейро").\n'
    '  "uslovie"  — для кого совет, если сказано (турист/резидент/…), иначе null.\n'
    "Только JSON, без пояснений."
)


# ── БАТЧ РАЗМЕТКИ (2026-07-26, решение юзера: пачка 25).
# Зачем: facet шлёт ПО ОДНОЙ мухе и потому разгоняется до 43 запросов/мин — по замерам это
# втрое выше границы, за которой источник начинает отдавать 429 (translate на 11/мин: 2372
# запроса, 14 отказов; facet на 43/мин: 2949 запросов, 861 отказ). Пачка режет число
# запросов, а не темп каждого — единственный рычаг, который не требует растягивать прогон.
#
# ПОЧЕМУ 25, А НЕ 50 КАК У translate: у facet выход ≈ входу (перевод по длине равен мухе)
# ПЛЮС теги, а у translate возвращается только перевод. При 25 мухах по ~150 токенов выход
# ~5К — с запасом под потолок генерации. Число выбрал юзер.
#
# ФОРМА ОТВЕТА — ТАБЛИЦА, не объекты на каждую муху:
#   строка = [индекс, перевод, задачи, сущности, место, условие]
# Имена пяти полей, повторённые 25 раз, — чистый расход выходных токенов; а число колонок
# и совпадение индекса с позицией ПРОВЕРЯЕМЫ, в отличие от имён, которые модель переименует.
# Ключ входа — ПОРЯДКОВЫЙ номер, не id: на 24-символьном хэше модель врала при копировании
# (факт 07-22, см. facet_lang.translate_texts). Настоящий id живёт снаружи.
FACET_BATCH = 25

# dead-letter: мухи, которые facet_one провалил как "bad" >=DEAD_AT раз (непереваримый
# контент). Иначе одна битая муха в хвосте держит зрелость гео вечно (remaining застревает
# на 1). Инфра-сбои (бюджет/429) НЕ считаем — их отсеивает status="infra".
#
# ⛔ БЫЛА ЛОКАЛЬНОЙ ВНУТРИ ФУНКЦИИ (поднята 2026-08-07). Из-за этого `facet.DEAD_AT` не
# существовал, а тройку переписывали руками: в `bot.py` (фильтр дед-леттера пульта) и в
# гейте зрелости `ship`. Правило про порог мёртвой мухи обязано жить в ОДНОМ месте —
# иначе пульт зовёт на мухи, которых facet не берёт (так и было: 26 мух вечно в меню).
DEAD_AT = 3

FACET_BATCH_SYS = (
    "Ты РАЗМЕТЧИК готовых советов (мух) по фасетам, НЕ автор. Мух НЕ переписывай, НЕ дополняй, "
    "НЕ сокращай, НЕ обобщай.\n"
    'На вход JSON {"0": "<муха>", "1": "<муха>", ...}. Разметь КАЖДУЮ.\n'
    'Верни СТРОГО JSON: {"rows": [[индекс, perevod, zadachi, sushnosti, mesto, uslovie], ...]}\n'
    "Ровно 6 элементов в строке, ровно по одной строке на каждую муху входа, индекс — та же "
    "строка-ключ, что на входе. Порядок строк любой, но индекс обязателен и уникален.\n"
    '  индекс     — строка, ключ мухи из входа (пример: "7").\n'
    "  perevod    — дословный перевод мухи на русский: ВСЕ факты/числа/названия/условия/оговорки "
    "как есть, ничего не добавить и не выкинуть, естественный русский (не калька).\n"
    "  zadachi    — СПИСОК задач/тем, которых касается совет (МОЖЕТ БЫТЬ НЕСКОЛЬКО, это ключ). "
    'Коротко, из текста мухи. Пример: ["получение CPF"]. Не выдумывай задач, которых в мухе нет.\n'
    '  sushnosti  — СПИСОК ПАР [["имя","роль"], ...]: конкретные сущности (CPF, ВНЖ, Vivo, '
    "Correios, Рио…) и роль каждой. Роль ∈ {цель, требование, обход, обстоятельство}: цель — за "
    "ней пришли; требование — без неё не сделать задачу; обход — способ без неё; обстоятельство "
    '— просто упомянута. Пример: [["CPF","требование"],["Busbud","обход"]]. Нет сущностей — [].\n'
    '  mesto      — город/страна, если совет привязан к месту, иначе null (пример: "Рио").\n'
    "  uslovie    — для кого совет, если сказано (турист/резидент/…), иначе null.\n"
    "Только JSON, без пояснений."
)


def _row_to_rec(fid, row):
    """Строка таблицы → запись мухи. None, если строка не годится (муха идёт в дед-леттер).

    Проверяем СТРОГО: длину строки и наличие обязательных полей. Мягкая склейка «как есть»
    тут запрещена — именно она в carve давала откат, который снаружи выглядел успехом.
    """
    if not isinstance(row, (list, tuple)) or len(row) != 6:
        return None
    _, perevod, zadachi, sushnosti, mesto, uslovie = row
    if not isinstance(perevod, str) or not perevod.strip():
        return None
    zad = [z.strip() for z in (zadachi or []) if isinstance(z, str) and z.strip()]
    if not zad:
        return None  # без задачи муху не к чему привязать как вид
    ent = []
    for pair in sushnosti or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0]).strip():
            rol = pair[1] if pair[1] in ROLES else "обстоятельство"
            ent.append({"imya": str(pair[0]).strip(), "rol": rol})
    return {
        "id": fid,
        "perevod": perevod.strip(),
        "zadachi": zad,
        "sushnosti": ent,
        "mesto": (mesto or None),
        "uslovie": (uslovie or None),
    }


def facet_many(chunk):
    """Пачка [(fid, lesson), ...] → (recs, bad_fids, reason).

    recs      — разобранные записи (status "ok" поштучно);
    bad_fids  — мухи, чью строку модель вернула негодной → дед-леттер, как "bad" у facet_one;
    reason    — не None, если ВСЯ пачка не получилась (инфра): мух не винить, отложить.

    Ретрай пачки до 3 раз, как в translate: разовый флаки-парс на повторе обычно проходит,
    а если пул реально не отдаёт — call вернёт None быстро, без похода в Google.
    """
    idx = {str(j): lesson for j, (_fid, lesson) in enumerate(chunk)}
    out = None
    for _ in range(3):
        out = call(
            json.dumps(idx, ensure_ascii=False), FACET_BATCH_SYS, consumer="facet"
        )
        if out is not None:
            break
    if out is None:
        return [], [], "пул не отдал (после 3 попыток пачки)"
    rows = out.get("rows")
    if not isinstance(rows, list) or not rows:
        return [], [], "модель не вернула rows"
    by_i = {}
    for row in rows:
        if isinstance(row, (list, tuple)) and row:
            by_i[str(row[0]).strip()] = row  # сшивка ПО ИНДЕКСУ, не по порядку строк
    recs, bad = [], []
    for j, (fid, _lesson) in enumerate(chunk):
        rec = _row_to_rec(fid, by_i.get(str(j)))
        (recs if rec else bad).append(rec or fid)
    return recs, bad, None


def _atomic_json(path, obj):
    """Запись через temp+rename: kill в любой момент не оставит недописанный/битый файл."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)  # атомарно на том же fs


def done_and_dead(geo):
    """(протегованные, мёртвые) id для гео. Одно место, где читаются tags/<geo>*.json."""
    done, fails = set(), {}
    try:
        with open(f"tags/{geo}.json", encoding="utf-8") as fh:
            done = {r["id"] for r in json.load(fh)}
    except Exception:
        pass
    try:
        with open(f"tags/{geo}_fails.json", encoding="utf-8") as fh:
            fails = json.load(fh)
    except Exception:
        pass
    return done, {fid for fid, c in fails.items() if c >= DEAD_AT}


def mature_geos():
    """Гео, у которых НЕ ОСТАЛОСЬ живых мух в очереди → блок можно публиковать.

    ⭐ ЗАМЕНА МЁРТВЫМ ШТАМПАМ (2026-08-07). Гейт `ship` пускал гео только по
    `runner_stamps.json`, а писал его `pseo-runner`, снесённый 20.07 за то, что жил
    невидимкой и жёг ключи. Файл замёрз на 36 гео из 90: всё, собранное позже, не поехало
    бы НИКОГДА — не из-за сырости, а потому что штамповать стало некому.

    Зрелость теперь ВЫЧИСЛЯЕТСЯ по тому же правилу, по которому facet берёт работу:
    зрелое = `load_flies` не отдаёт ничего. Мёртвые мухи (>=DEAD_AT провалов) живыми не
    считаются — иначе одна непереваримая муха держит гео вечно (ровно так 26 мух висели
    в меню пульта, и все 26 были мёртвыми).
    """
    import glob as _glob

    out = {}
    for f in sorted(_glob.glob("out_facet/*.json")):
        geo = os.path.basename(f)[:-5]
        done, dead = done_and_dead(geo)
        out[geo] = not load_flies(geo, limit=1, exclude=done | dead)
    return out


ANY_GEO = "any"  # легальное значение колонки: совет не привязан к стране


def geo_codes(raw):
    """Значение колонки `country` → МНОЖЕСТВО кодов. Схему не меняем: колонка одна, но
    значение может быть перечислением («de, ru»).

    ⭐ ШАГ 8, корень (замер по базе): у мухи ОДНА колонка и сравнение на равенство, а
    промпт давал бинарный выбор «одна страна или any». Совет про две страны выразить
    было нечем: он падал либо в `any` (3 077 мух), либо в мусорный ключ с запятой —
    таких 29 (`de, ru` 5, `ru, kg` 4, `kg, kz, ru` 2, `au, nz` 2 …), и эти мухи не
    попадали НИ В ОДНУ страну. Оба исхода видели живьём.

    Нормализация здесь же и по единственному источнику правды — справочнику стран:
    нижний регистр, срезать пробелы, неизвестное ОТБРОСИТЬ. Отсюда же исчезают
    гео-призраки: код, которого нет в справочнике, страницей больше не станет.
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

    Гео берётся по ВХОЖДЕНИЮ кода в список (шаг 8): `de, ru` попадает и в Германию, и в
    Россию. `LIKE` — только грубый пред-отбор, чтобы не тащить 23 924 текста на каждый
    вызов; точное решение принимает `geo_codes`, иначе `an` цеплял бы `any`, а `ru` —
    любое значение с этими буквами.
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


def facet_one(fid, lesson):
    """Возвращает (status, rec): "ok"+запись | "bad"+None (муха непереваримая — к дед-леттеру) |
    "infra"+None (gemini_json ничего не отдал — бюджет/429/таймаут, НЕ вина мухи, не пенализировать).
    """
    out = call(lesson, FACET_SYS, consumer="facet")  # сосок мозга, рот=facet
    if out is None:
        return ("infra", None)  # инфра-сбой — муху не виним
    if "perevod" not in out or "zadachi" not in out:
        return ("bad", None)  # модель ответила, но контент невалиден → битая муха
    zad = [z.strip() for z in (out.get("zadachi") or []) if z and z.strip()]
    ent = []
    for e in out.get("sushnosti") or []:
        if isinstance(e, dict) and e.get("imya"):
            rol = e.get("rol") if e.get("rol") in ROLES else "обстоятельство"
            ent.append({"imya": e["imya"].strip(), "rol": rol})
    if not zad:
        return ("bad", None)  # без задачи муху не к чему привязать как вид (редко)
    return (
        "ok",
        {
            "id": fid,
            "perevod": out["perevod"].strip(),
            "zadachi": zad,
            "sushnosti": ent,
            "mesto": (out.get("mesto") or None),
            "uslovie": (out.get("uslovie") or None),
        },
    )


# ── CARVE (джоб1): ЗАМЕНА consolidate. Читаем ТЕКСТЫ мух пачкой по фасет-семье → LLM вычленяет
# подпункты и присваивает мух. Правильнее дедупа меток (доказано br/vn 2026-07-18): метка =
# degraded-артефакт, текст = исходник — carve находит нюансы, которых в метке нет, держит ВНЖ≠гражданство.
# ГРАНИЦА: carve ТОЛЬКО на плотной когерентной семье; тонкий хвост уходит в джоб2 (assign_tail —
# раскладка по таксономии полки×тип, НЕ сырые синглтоны). split-по-числу мёртв: режем по семье+плотности.
MIN_CARVE = 6  # ГРАНИЦА джоб1/джоб2: семья ≥6 → carve (уплотнить), мельче → в хвост-раскладку (assign_tail)
CARVE_BATCH = 90  # мух в пачку (окно ~16.6К ток; 90×~110 ≈ 10К, проверено 200)

CARVE_SYS = (
    "Ниже связанные советы (id: текст) — близкая тема. Вычлени СВЯЗНЫЕ подпункты (каждый = "
    "отдельная страница-гайд). Присвой каждый совет к подпункту(ам) — совет МОЖЕТ быть в "
    "нескольких. Правила: дубли/переформулировки ОДНОЙ задачи — в ОДИН подпункт; РАЗНЫЕ "
    "задачи/объекты/места — РАЗДЕЛЬНО (студенческая≠рабочая виза; CPF≠гражданство≠ВНЖ); НЕ "
    "укрупняй в широкие категории; НЕ выдумывай тем, которых в советах нет; охвати ВСЕ.\n"
    "⛔ ИМЯ ПОДПУНКТА = ЗАГОЛОВОК СТРАНИЦЫ, значит это ОДИН запрос человека, а не рубрика "
    "и не перечень. ЗАПРЕЩЕНЫ имена «Прочее», «Разное», «Общие советы…», «Полезная "
    "информация…», «Адаптация…», «Особенности» без предмета, а также имя, совпадающее с "
    "названием широкой темы (например «Транспорт и логистика»). Если подпункт выходит "
    "сборным — раздели его или НЕ ВЫДЕЛЯЙ вовсе.\n"
    'СТРОГО JSON: {"intents":[{"name":"<конкретная тема>","ids":["0",...]}]}'
)


# ── ОСЬ НАРЕЗКИ = РАЗДЕЛ (канон §0.15) ─────────────────────────────────────────────
# ⛔ ЧТО БЫЛО НЕ ТАК. Семьи для нарезки собирались по ПЕРВОМУ СЛОВУ метки задачи
# (`_first_word`). Первое слово — грамматическая случайность, а не тема: «Сроки оформления
# шенгенской визы» и «Требования к документам» уезжали в РАЗНЫЕ семьи, и рот `carve`
# физически не мог их сравнить — он видит только свою семью. Отсюда 41 разбор при девяти
# делах (замер по визам Греции): рот честно резал то, что ему дали.
#
# ⭐ КАК СТАЛО. Ось — РАЗДЕЛ, он же ось сайта:
#   1) раздел ставится КАЖДОЙ мухе (тот же рот `assign`, тот же закрытый список 13 —
#      механизм уже работал на хвосте, тут он применён ко всем мухам);
#   2) семья = «страна × раздел», значит близкие дела гарантированно в одной пачке;
#   3) нарезка идёт ДВУМЯ проходами: А составляет ЗАКРЫТЫЙ список дел раздела, Б
#      присваивает мухам дела ИЗ ЭТОГО списка. Закрытый список и есть защита от
#      размножения формулировок: рот не может выдумать шестнадцатую.
FLY_SHELF_BATCH = 90  # мух в пачку (тексты; та же пачка, что у раскладки хвоста)

FLY_SHELF_SYS = (
    "Ниже советы (id: текст). Отнеси КАЖДЫЙ ровно к ОДНОМУ разделу закрытой таксономии. "
    "Отвечай КЛЮЧОМ раздела.\n"
    + "\n".join(f"{k} — {name}: {desc}" for k, name, desc in tax.SHELVES)
    + "\nНи один не подходит → ключ 'prochee' (сигнал дырки, не злоупотребляй).\n"
    'СТРОГО JSON: {"map":{"0":"<ключ раздела>",...}}'
)

DEALS_MAX = 15  # дел в разделе максимум: больше — это уже не список, а простыня
DEALS_LABELS = 90  # меток в проход А (они коротки, окно держит с запасом)

DEALS_SYS = (
    "Ниже метки задач из ОДНОГО раздела гида по стране (id: метка), в скобках число "
    "советов. Составь ЗАКРЫТЫЙ список ДЕЛ этого раздела — то, с чем человек реально "
    f"приходит. Дел от 5 до {DEALS_MAX}.\n"
    "ПРАВИЛА ИМЕНИ ДЕЛА: имя = ОДИН запрос человека, оно станет ЗАГОЛОВКОМ страницы. "
    "Разные формулировки одного дела — ОДНО дело. Разные дела не смешивать "
    "(студенческая≠рабочая виза; CPF≠гражданство≠ВНЖ). ЗАПРЕЩЕНЫ имена «Прочее», "
    "«Разное», «Общие советы…», «Полезная информация…», «Адаптация…», «Особенности» без "
    "предмета, а также имя, совпадающее с названием самого раздела.\n"
    'СТРОГО JSON: {"deals":["<имя дела>", ...]}'
)

DEAL_ASSIGN_SYS = (
    "Ниже ЗАКРЫТЫЙ список дел раздела (номер: имя), затем советы (id: текст). Отнеси "
    "КАЖДЫЙ совет ровно к ОДНОМУ делу ИЗ СПИСКА — новых дел не придумывать. Если ни одно "
    'дело не подходит, ставь "0".\n'
    'СТРОГО JSON: {"map":{"<id совета>":"<номер дела или 0>",...}}'
)


def deals_for_pair(geo, shelf_key, fails=None):
    """КОНТРОЛЬ: список дел ОДНОГО раздела одного гео — ровно ОДИН вызов рта.

    Зачем отдельный вход: проход А в боевом прогоне идёт по всем разделам гео (замер сухого
    прогона 19.08: 5–13 вызовов на гео), и «проверить метод на четырёх парах» без этого
    входа означало бы 87 вызовов вместо четырёх.

    Раздел мухам может быть ещё не проставлен (шаг `--assign-flies` не прогонялся), поэтому
    метки берём из ВИДОВ этого раздела — того же материала, что увидит проход А в бою.
    """
    fn = f"out_facet/{geo}.json"
    if not os.path.exists(fn):
        print(f"{geo}: корпуса нет", flush=True)
        return []
    page = json.load(open(fn, encoding="utf-8"))
    by_name = {n: k for k, n, _ in tax.SHELVES}
    mass = {}
    for v in page.get("views_by_task") or []:
        if len(v.get("items") or []) < tax.PAGE_MIN:
            continue
        if by_name.get(v.get("shelf") or "") != shelf_key:
            continue
        z = (v.get("zadacha") or "").strip()
        if z:
            mass[z] = mass.get(z, 0) + len(v["items"])
    if not mass:
        print(f"{geo}/{shelf_key}: меток нет — контролировать нечего", flush=True)
        return []
    print(
        f"{geo}/{shelf_key}: меток {len(mass)}, абзацев {sum(mass.values())}",
        flush=True,
    )
    deals, _answered = carve_deals(mass, fails, f"{geo}/{shelf_key}")
    print(f"{geo}/{shelf_key}: ДЕЛ {len(deals)}", flush=True)
    for d in deals:
        print(f"    - {d}", flush=True)
    return deals


def assign_fly_shelves(geo, fails=None):
    """Раздел КАЖДОЙ размеченной мухе гео. Пишет `shelf_key` в `tags/<geo>.json`.

    Работой считаются только мухи без раздела — повторный запуск ключей не тратит.
    Муха, на которую рот не ответил или ответил неизвестным ключом, остаётся без раздела:
    она уйдёт в хвост, а не в выдуманный раздел.
    """
    fn = f"tags/{geo}.json"
    if not os.path.exists(fn):
        print(f"{geo}: разметки нет — раздел мухам ставить нечему", flush=True)
        return 0
    tagged = json.load(open(fn, encoding="utf-8"))
    todo = [r for r in tagged if not r.get("shelf_key")]
    if not todo:
        print(f"{geo}: раздел у всех {len(tagged)} мух — пропуск", flush=True)
        return 0
    keys = {k for k, _n, _d in tax.SHELVES} | {"prochee"}
    done = unknown = 0
    unknown_keys = []
    for st in range(0, len(todo), FLY_SHELF_BATCH):
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп между пачками на {st}/{len(todo)}", flush=True)
            break
        chunk = todo[st : st + FLY_SHELF_BATCH]
        idx = {str(j): r["perevod"] for j, r in enumerate(chunk)}
        res = call(
            json.dumps(idx, ensure_ascii=False), FLY_SHELF_SYS, consumer="assign"
        )
        m = (res or {}).get("map") or {}
        if not m:
            if fails is not None:
                fails.append(
                    {"step": "fly_shelf", "geo": geo, "batch": st // FLY_SHELF_BATCH}
                )
            continue
        for j, r in enumerate(chunk):
            k = (m.get(str(j)) or "").strip()
            if k in keys:
                r["shelf_key"] = k
                done += 1
            else:
                unknown += 1
                if k and k not in unknown_keys:
                    unknown_keys.append(k)
    if done:
        _atomic_json(fn, tagged)
    print(
        f"{geo}: мух {len(tagged)}, без раздела было {len(todo)} -> размечено {done}"
        + (
            f", неопознанный ключ {unknown} ({','.join(unknown_keys[:5])})"
            if unknown
            else ""
        ),
        flush=True,
    )
    return done


def _queue(fails, step, family, flies):
    """Записать НЕСДЕЛАННОЕ в файл гео — это и есть очередь (канон §0.17).

    ⛔ Не отчёт: эту запись читает пульт (`pipeline_state` → шаг 0, «сперва брак») и ставит
    такое гео первым на перепрогон. Форма ОДНА для всех звеньев — `step`, `family`, `flies`:
    читатели берут `flies` и `family`, и на чужой форме молча показывали «0 мух» и «?».
    """
    if fails is not None:
        fails.append({"step": step, "family": family, "flies": flies})


def carve_deals(labels_with_mass, fails=None, family=None):
    """Проход А: метки раздела → ЗАКРЫТЫЙ список дел.

    Возвращает (имена, ответил_ли_рот). Канон §0.17: пустой список при `answered=True` —
    рот ответил, годных имён нет, раздел законно уходит в хвост. При `answered=False` пачка
    не доехала (429, транспорт, СТОП), и раздел обязан остаться РАБОТОЙ ШАГА, а не уехать в
    хвост: хвост — это размещение, то есть заявление «с мухой разобрались».
    """
    top = sorted(labels_with_mass.items(), key=lambda kv: -kv[1])[:DEALS_LABELS]
    left = len(labels_with_mass) - len(top)
    if left:
        print(f"    проход А: меток сверх пачки роту не показано: {left}", flush=True)
    idx = {str(j): f"{z} ({n})" for j, (z, n) in enumerate(top)}
    res = call(json.dumps(idx, ensure_ascii=False), DEALS_SYS, consumer="carve")
    raw = [(x or "").strip() for x in ((res or {}).get("deals") or [])]
    good = []
    for d in [x for x in raw if x][:DEALS_MAX]:
        why = tax.bad_label(d)  # правило имени то же, что у меток (канон §0.13)
        if why:
            print(f"    проход А: имя дела отвергнуто ({why}): {d!r}", flush=True)
            continue
        if d not in good:
            good.append(d)
    return good, res is not None


def assign_to_deals(fids, by_id, deals, fails=None, family=None):
    """Проход Б: мухи раздела → дела ИЗ ЗАКРЫТОГО списка.

    Возвращает ({имя дела: [id мух]}, застрявшие). Канон §0.17:
    - муха, которой дело не подошло (ответ "0") или чей номер вне списка, остаётся
      неприсвоенной и уйдёт в хвост — рот её видел и не взял;
    - мухи ПАЧКИ, которая не доехала (или оборвана СТОПом), попадают в «застрявшие»: они не
      размещаются нигде и остаются работой шага до следующего прохода.
    """
    out, stalled = {}, []
    for st in range(0, len(fids), CARVE_BATCH):
        if os.path.exists("RUNNER_STOP"):
            print(f"    стоп между пачками на {st}/{len(fids)}", flush=True)
            stalled.extend(
                fids[st:]
            )  # необработанный остаток — работа, а не размещение
            break
        chunk = fids[st : st + CARVE_BATCH]
        head = "\n".join(f"{i + 1}: {d}" for i, d in enumerate(deals))
        body = {str(j): by_id[fid]["perevod"] for j, fid in enumerate(chunk)}
        res = call(
            head + "\n---\n" + json.dumps(body, ensure_ascii=False),
            DEAL_ASSIGN_SYS,
            consumer="carve",
        )
        m = (res or {}).get("map") or {}
        if not m:
            stalled.extend(chunk)  # пачка не доехала — мухи остаются работой шага
            continue
        for j, fid in enumerate(chunk):
            # ⛔ `str(...)`: промпт просит номер В КАВЫЧКАХ, а рты кавычки роняют и отдают
            # число. На `(… or "").strip()` это был AttributeError, то есть падение всего
            # прогона гео вместе с уже оплаченными вызовами.
            v = str(m.get(str(j)) or "").strip()
            if v.isdigit() and 1 <= int(v) <= len(deals):
                out.setdefault(deals[int(v) - 1], []).append(fid)
    return out, stalled


def _first_word(z):
    p = z.split()
    return p[0].lower() if p else z.lower()


def carve_family(fids, by_id, fails=None, family=None):
    """Пачка мух семьи (ТЕКСТЫ perevod) → LLM вычленяет подпункты + присваивает. [{name, ids}].
    Батчами по CARVE_BATCH.

    ⛔ ОТКАТ БОЛЬШЕ НЕ МАСКИРУЕТСЯ (юзер 07-23). Раньше сбой пачки молча давал каждой мухе
    имя её ИСХОДНОЙ задачи — а это общие рубрики («Транспорт», «Документы и виза»), и склейка
    по имени лепила ком на 139 пунктов. Снаружи выглядело как нормальная сборка: код 0,
    файл записан, гео «готово». Так собрались 6 гео (hu/ge/ar/kz/in/kg) — БЕЗ тематической
    нарезки вообще. Теперь провал пачки ЗАПИСЫВАЕТСЯ в fails: мухи не теряем (откат остаётся),
    но гео честно помечено недоделанным — попадёт в отчёт и встанет на перепрогон.
    """
    out = []
    for s in range(0, len(fids), CARVE_BATCH):
        chunk = fids[s : s + CARVE_BATCH]
        idx = {str(j): by_id[fid]["perevod"] for j, fid in enumerate(chunk)}
        res = call(json.dumps(idx, ensure_ascii=False), CARVE_SYS, consumer="carve")
        if not res or not res.get("intents"):
            if (
                fails is not None
            ):  # ТОЧКА ОТКАЗА: где именно обломилось (для перепрогона)
                fails.append(
                    {
                        "step": "carve",
                        "family": family,  # ключ семьи (первое слово метки) — как её собирали
                        "batch": f"{s // CARVE_BATCH + 1}/{(len(fids) - 1) // CARVE_BATCH + 1}",
                        "flies": len(chunk),
                        "ids": chunk,  # ИМЕННО эти мухи не разобраны — можно добить точечно
                        "why": "carve не вернул intents",
                    }
                )
            for fid in chunk:  # мух НЕ теряем: откат как был, но теперь он виден
                out.append({"name": by_id[fid]["zadachi"][0], "ids": [fid]})
            continue
        for g in res["intents"]:
            ids = [
                chunk[int(i)]
                for i in (g.get("ids") or [])
                if isinstance(i, str) and i.isdigit() and int(i) < len(chunk)
            ]
            if ids:
                name = (g.get("name") or by_id[ids[0]]["zadachi"][0]).strip()
                out.append({"name": name, "ids": ids})
    return out


# ── ХВОСТ-РАСКЛАДКА (джоб2): тонкий хвост — НЕ сырые синглтоны и НЕ мёрж, а раскладка по
# ГЛОБАЛЬНОЙ таксономии полки×тип (tail_taxonomy.py). Каждый сингл = самостоятельный абзац,
# ложится на полку(и) как антология + получает тип. Метод open→lock→assign доказан на ru
# 2026-07-19 (192/192, ~94% чисто). Непокрытое → prochee (park-ведро, сигнал роста таксономии).
ASSIGN_SYS = (
    "Ниже разрозненные советы людей из чатов (id: текст) — каждый самостоятелен, "
    "НЕ схлопывай и НЕ выкидывай. Разложи КАЖДЫЙ по ЗАКРЫТОЙ таксономии:\n"
    "ПОЛКИ (можно НЕСКОЛЬКО, минимум 1): " + " | ".join(tax.SHELF_NAMES) + "\n"
    "ТИП (РОВНО один): " + " | ".join(tax.TYPE_NAMES) + "\n"
    "Не лезет ни на одну полку → полка '"
    + tax.PROCHEE
    + "' (сигнал дырки, не злоупотребляй).\n"
    'СТРОГО JSON: {"assign":{"0":{"shelves":["..."],"type":"..."},...}}'
)


def assign_tail(tail_fids, by_id, fails=None):
    """Тонкий хвост → раскладка по таксономии. Возвращает (shelves {полка: [items]},
    prochee [items]). Потерь НЕТ: сбой пачки / непокрытие / не-возврат мухи → в prochee.

    РАЗЛИЧАЕМ (07-23): муха в prochee «не подошла ни под одну полку» = НОРМА (парк-ведро,
    сигнал роста таксономии); а «пачка не ответила 3 раза» = СБОЙ → в fails, чтобы знали,
    что раскладка на этой пачке не состоялась (раньше молчало, как ветвление полок)."""
    fids = list(tail_fids)
    shelves, prochee = {}, []
    stop_at = None  # индекс, на котором нажали стоп (остаток честно уходит в prochee)

    def item(fid, typ):
        r = by_id[fid]
        return {
            "id": r["id"],
            "text": r["perevod"],
            "sushnosti": r["sushnosti"],
            "mesto": r["mesto"],
            "uslovie": r["uslovie"],
            "type": typ,
        }

    for s in range(0, len(fids), CARVE_BATCH):
        # ДУБЛЬ КОМБАЙНА: стоп МЕЖДУ ПАЧКАМИ — у facet такая проверка есть, у хвоста
        # не было, и стоп посреди гео сжигал уже сделанные вызовы впустую.
        if os.path.exists("RUNNER_STOP"):
            stop_at = s
            print(f"  стоп между пачками на {s}/{len(fids)}", flush=True)
            break
        chunk = fids[s : s + CARVE_BATCH]
        idx = {str(j): by_id[fid]["perevod"] for j, fid in enumerate(chunk)}
        res = None
        for _ in range(3):
            res = call(
                json.dumps(idx, ensure_ascii=False), ASSIGN_SYS, consumer="assign"
            )
            if res and res.get("assign"):
                break
        if not (
            res and res.get("assign")
        ):  # пачка молчит 3 раза → СБОЙ (не «не покрыто»)
            if fails is not None:
                fails.append(
                    {
                        "step": "assign_tail",
                        "batch": f"{s // CARVE_BATCH + 1}/{(len(fids) - 1) // CARVE_BATCH + 1}",
                        "flies": len(chunk),
                        "ids": chunk,
                        "why": "раскладка хвоста не ответила (3 попытки) → мухи в prochee",
                    }
                )
        a = (res or {}).get("assign") or {}
        for j, fid in enumerate(chunk):
            rec = a.get(str(j))
            if not isinstance(rec, dict):  # пачка сдохла / муху не вернули → не теряем
                prochee.append(item(fid, ""))
                continue
            typ = rec.get("type") if rec.get("type") in tax.TYPE_NAMES else ""
            shs = [x for x in (rec.get("shelves") or []) if x in tax.SHELF_NAMES]
            if not shs:  # не покрыто таксономией → park
                prochee.append(item(fid, typ))
                continue
            for sh in shs:
                shelves.setdefault(sh, []).append(item(fid, typ))
    if stop_at is not None:  # инвариант «потерь НЕТ»: недоразобранный хвост → prochee
        prochee.extend(item(fid, "") for fid in fids[stop_at:])
        print(
            f"  остаток {len(fids) - stop_at} мух → prochee (следующий запуск разложит)",
            flush=True,
        )
    return shelves, prochee


def build_views_by_carve(tagged, fails=None):
    """Джоб1: плотные фасет-семьи (≥MIN_CARVE) → carve по ТЕКСТАМ (тематические страницы,
    уплотняет дубли). Джоб2: тонкий хвост (мухи, не попавшие в carve) → assign_tail по глобальной
    таксономии полки×тип (антологии, НЕ сырые синглтоны). Возвращает (views {интент: [items]},
    shelves {полка: [items]}, prochee [items]). views_by_task ← views (совместимо с pages.py).
    """
    by_id = {r["id"]: r for r in tagged}
    fams = {}
    for r in tagged:
        for z in r["zadachi"]:
            fams.setdefault(_first_word(z), set()).add(r["id"])

    views = {}
    dropped = []  # сборные метки: отсев обязан быть ВИДЕН, а не молчалив
    # ⛔ §0.17: мухи, чья пачка не доехала (429, транспорт, СТОП). Они НЕ идут ни на страницу,
    # ни в хвост — остаются работой шага, иначе упавшее выглядит размещённым и заново не встаёт.
    stalled = set()
    # ⭐ ОСЬ = РАЗДЕЛ (канон §0.15). Раздел стоит у мухи (`shelf_key`, рот `assign`),
    # поэтому семья — «страна × раздел», а не первое слово метки.
    # ⚠️ ПЕРЕХОД: если раздела нет НИ У ОДНОЙ мухи (гео ещё не прогонялось новым шагом),
    # падаем на старую ось по первому слову и ГОВОРИМ об этом. Иначе первый же прогон на
    # старых данных отправил бы весь корпус гео в хвост.
    by_shelf = {}
    for r in tagged:
        k = r.get("shelf_key")
        if k and k != "prochee":
            by_shelf.setdefault(k, []).append(r["id"])
    # ⛔ «Новый шаг прошёл» и «есть что резать» — РАЗНОЕ. Если все мухи гео легли в
    # парк-ведро `prochee`, раздел у них ЕСТЬ, значит старая ось уже неуместна: иначе гео
    # молча резалось бы по первому слову. Поймано собственным сторожем.
    new_axis = any(r.get("shelf_key") for r in tagged)
    if not new_axis:
        print(
            "  РАЗДЕЛА У МУХ НЕТ — семьи по первому слову (старая ось). "
            "Прогнать шаг «раздел мухам», иначе раздробленность останется",
            flush=True,
        )

    def add(name, fid):
        r = by_id[fid]
        views.setdefault(name, []).append(
            {
                "id": r["id"],
                "text": r["perevod"],
                "sushnosti": r["sushnosti"],
                "mesto": r["mesto"],
                "uslovie": r["uslovie"],
            }
        )

    if new_axis:
        # ── ДВА ПРОХОДА ПО РАЗДЕЛУ ─────────────────────────────────────────────────
        # А: метки раздела → ЗАКРЫТЫЙ список дел. Б: мухи → дела ИЗ этого списка.
        # Мухи, которым дело не досталось, идут в хвост — как и раньше.
        for skey, fids in sorted(by_shelf.items(), key=lambda kv: -len(kv[1])):
            if len(fids) < MIN_CARVE:
                continue  # тонкий раздел → целиком в хвост, дела там выделять не из чего
            mass = {}
            for fid in fids:
                for z in by_id[fid].get("zadachi") or []:
                    z = (z or "").strip()
                    if z:
                        mass[z] = mass.get(z, 0) + 1
            if not mass:
                continue
            deals, answered = carve_deals(mass, fails, f"{skey}")
            print(
                f"  раздел {skey}: мух {len(fids)}, меток {len(mass)} -> дел {len(deals)}",
                flush=True,
            )
            if not deals:
                # Рот ответил, а годных имён нет → раздел законно в хвост. Пачка не доехала
                # (§0.17) → раздел остаётся работой шага, в хвост его не размещаем.
                if not answered:
                    stalled.update(fids)
                    _queue(fails, "deals", skey, len(fids))
                continue
            assigned, stalled_here = assign_to_deals(fids, by_id, deals, fails, skey)
            if stalled_here:
                stalled.update(stalled_here)
                _queue(fails, "deal_assign", skey, len(stalled_here))
            for name, dfids in assigned.items():
                why = tax.bad_label(name)
                if why:
                    dropped.append((name, len(dfids), why))
                    continue
                for fid in dfids:
                    add(name, fid)
    else:
        for w, fset in fams.items():
            fids = list(fset)
            if len(fids) < MIN_CARVE:
                continue  # тонкие семьи → хвост-раскладка ниже (НЕ сырые синглтоны)
            for it in carve_family(
                fids, by_id, fails, w
            ):  # плотная семья: carve по текстам
                # ⛔ СБОРНАЯ МЕТКА — БРАК НАРЕЗКИ, А НЕ ТЕМА (канон §0.13). Вид НЕ пишем:
                # мухи остаются в хвосте и раскладываются по разделам обычным порядком.
                why = tax.bad_label(it["name"])
                if why:
                    dropped.append((it["name"], len(it["ids"]), why))
                    continue
                for fid in it["ids"]:
                    add(it["name"], fid)

    # страховка: дедуп мух в карв-виде по id (одна муха могла попасть дважды на стыке семей)
    for name, items in views.items():
        seen, uniq = set(), []
        for it in items:
            if it["id"] not in seen:
                seen.add(it["id"])
                uniq.append(it)
        views[name] = uniq

    # ⛔ ПРОСЕВШИЙ ВИД НЕ ОСТАЁТСЯ В КОРПУСЕ. После дедупа вид мог упасть ниже порога страницы,
    # и тогда в файле гео лежала бы «вторая правда»: страницей он не станет (её гейт тот же
    # порог), а вид есть — значит и счётчики пульта, и переводы считали бы работу, которой нет.
    # Потеря НЕ молчаливая: печатаем, а мухи уходят в хвост предикатом ниже.
    thin = [n for n, items in views.items() if len(items) < tax.PAGE_MIN]
    for n in thin:
        views.pop(n)
    if thin:
        print(
            f"  видов просело ниже порога страницы {len(thin)} -> мухи в хвост: "
            + "; ".join(repr(n) for n in thin[:5]),
            flush=True,
        )

    if dropped:
        print(
            f"  сборных меток отсеяно {len(dropped)} "
            f"(мух {sum(c for _n, c, _w in dropped)}) -> в хвост: "
            + "; ".join(f"{n!r}({c}): {w}" for n, c, w in dropped[:5]),
            flush=True,
        )

    # ⛔ ХВОСТ = мухи, не попавшие НИ НА ОДНУ СТРАНИЦУ. Раньше здесь стояло «не в ВИДЕ»
    # (`carved_fids`), и через эту щель абзацы уходили с сайта: дело из 2-3 мух видом
    # становилось, страницей — нет, а в хвост муха уже не шла. Порог тут НЕ применяем второй
    # раз: просевшие виды сняты выше, значит в `views` остались только страничные.
    on_page = {it["id"] for items in views.values() for it in items}
    tail_fids = [fid for fid in by_id if fid not in on_page and fid not in stalled]
    shelves, prochee = assign_tail(tail_fids, by_id, fails)
    return views, shelves, prochee


# Что стоило прогонов и НЕ должно исчезать при пересборке файла гео.
# `groups` — дедуп (keyless, но считается); `kratko` — запрос на страницу; `subshelves` —
# ветвление (запрос на страницу); `branch_tried` — «пробовали, вышло цельно»; `key` — АДРЕС.
CARRY_IF_SAME = ("groups", "kratko", "subshelves", "branch_tried")
# Какая доля СТАРОГО состава должна уцелеть в новом узле, чтобы считать его тем же самым
# и сохранить ему АДРЕС. Не Жаккар: страница, доросшая вдвое, по Жаккару даёт 0.5 и адрес
# бы сменился — то есть проиндексированный URL стал бы 404 просто от новых мух.
SAME_NODE_MIN = 0.7


def carry_forward(path, page, geo=""):
    """Перенести в НОВЫЙ файл гео то, что стоило прогонов. Мутирует `page` на месте.

    ⛔ ЗАЧЕМ (2026-08-08). `run()` собирает файл гео С НУЛЯ и пишет поверх. Значит в тот
    самый момент, когда гео ДОЗРЕВАЕТ (remaining==0) — то есть когда разметка удаётся —
    стираются дедуп-группы, короткие ответы, ветвление и адреса. Замер 08.08: 58 гео из 90
    стоят в очереди разметки, и в них лежит ВЕСЬ корпус — 1889 страниц, 1889 коротких
    ответов (100%), 182 ветвления. Месяц это не срабатывало только потому, что ни одно гео
    не доводили до конца: при remaining>0 функция выходит раньше записи. Машинерия рушила
    наработанное ровно на успехе. Правильный образец был всё это время в 30 строках ниже —
    `run_assign_tail` читает файл и мёржит ключи.

    ПРАВИЛО ОДНО, и оно про честность, а не про экономию:
      • состав id УЗЛА не изменился → переносим ВСЁ (`CARRY_IF_SAME` + адрес);
      • состав изменился → переносим ТОЛЬКО адрес.
    Почему не переносить остальное при изменившемся составе: `groups` — это разбиение
    ИМЕННО ЭТОГО набора мух, и новые мухи ни в одну группу не попадут; а страница
    рендерится ИЗ ГРУПП, значит новые мухи стали бы невидимы. Тихая потеря хуже
    пересчёта: дедуп keyless и дёшев. `kratko` — выжимка из содержимого; содержимое
    изменилось, значит выжимка устарела, и держать её — врать читателю. `subshelves`
    ссылаются на id репрезентантов, часть которых могла уйти.
    Адрес переносим ВСЕГДА: он опубликован и проиндексирован, и менять его от того, что
    в теме добавились мухи, — ломать живые ссылки. Совпадение узла ищем по доле
    сохранившегося СТАРОГО состава (`SAME_NODE_MIN`), а не по Жаккару.

    Молчания нет: что перенесено и что пересчитается — печатается.
    """
    try:
        old = json.load(open(path, encoding="utf-8"))
    except Exception:
        return  # первый прогон гео — переносить нечего

    def ids_of(node):
        return frozenset(i["id"] for i in node.get("items") or [])

    stat = {"full": 0, "addr": 0, "fresh": 0}
    for key in ("views_by_task", "shelves"):
        old_nodes = [(ids_of(o), o) for o in old.get(key) or []]
        old_nodes = [(i, o) for i, o in old_nodes if i]
        taken = set()  # один старый адрес не может достаться двум новым узлам
        for n in page.get(key) or []:
            nids = ids_of(n)
            same = next((o for i, o in old_nodes if i == nids), None)
            if same is not None:
                for f in CARRY_IF_SAME:
                    if f in same:
                        n[f] = same[f]
                if same.get("key"):
                    n["key"] = same["key"]
                    taken.add(same["key"])
                stat["full"] += 1
                continue
            # состав изменился → ищем, чей это узел, по доле уцелевшего СТАРОГО состава
            best, score = None, 0.0
            for oids, o in old_nodes:
                k = o.get("key")
                if not k or k in taken:
                    continue
                s = len(nids & oids) / len(oids)
                if s > score:
                    best, score = o, s
            if best is not None and score >= SAME_NODE_MIN:
                n["key"] = best["key"]
                taken.add(best["key"])
                stat["addr"] += 1
            else:
                stat["fresh"] += 1
    if any(stat.values()):
        print(
            f"{geo}: перенесено целиком {stat['full']}, только адрес {stat['addr']} "
            f"(им пересчитаются дедуп и короткий ответ), новых узлов {stat['fresh']}",
            flush=True,
        )


def run(geo, limit=None):
    """Накопительный прогон: догружает УЖЕ тегнутое (tags/<geo>.json), тегает СЛЕДУЮЩИЕ
    ≤limit мух, мёржит, пересобирает виды. Возвращает число НОВЫХ тегнутых (для темпа).
    """
    os.makedirs("tags", exist_ok=True)
    tags_fn = f"tags/{geo}.json"
    tagged = []
    if os.path.exists(tags_fn):
        try:
            tagged = json.load(open(tags_fn, encoding="utf-8"))
        except Exception:
            tagged = []
    done_ids = {r["id"] for r in tagged}
    fails_fn = f"tags/{geo}_fails.json"
    try:
        fails = json.load(open(fails_fn, encoding="utf-8"))
    except Exception:
        fails = {}
    dead = {fid for fid, c in fails.items() if c >= DEAD_AT}
    flies = load_flies(geo, limit, exclude=done_ids | dead)  # не сделанные и не мёртвые
    new_n = 0
    stopped = False
    tentative = []  # провалившиеся В ЭТОМ проходе (bad ИЛИ infra) — засчитаем в конце
    # ПАЧКАМИ по FACET_BATCH: один запрос на 25 мух вместо 25 запросов (см. FACET_BATCH_SYS).
    # Стоп-флаг читаем МЕЖДУ пачками — сделанное сохраняем, как и раньше между мухами.
    for s in range(0, len(flies), FACET_BATCH):
        if os.path.exists("RUNNER_STOP"):
            stopped = True
            break
        chunk = flies[s : s + FACET_BATCH]
        recs, bad_fids, reason = facet_many(chunk)
        for r in recs:
            tagged.append(r)
            new_n += 1
            print(
                f"  + {', '.join(r['zadachi'])[:48]:50} :: {r['perevod'][:52]}",
                flush=True,
            )
        if reason:  # ВСЯ пачка не получилась — инфра, мух не виним поштучно
            print(f"  пачка {s // FACET_BATCH + 1}: {reason}", flush=True)
        for fid in bad_fids:  # негодная строка = та же «bad», что была у facet_one
            tentative.append(fid)
        if reason:
            tentative.extend(fid for fid, _ in chunk)
            # СИСТЕМАТИКА: 0 тегнуто + >=3 провала = бюджет/инфра сдохли, НЕ вина мух →
            # откат (tentative не применяем), стоп прохода. Иначе битую муху виним честно.
            if new_n == 0 and len(tentative) >= 3:
                # маркер НЕУДАЧ → цикл сам перепройдёт фазу (транзиент пула), 3 раза не
                # вышло → стоп-машина. Текст ЧЕСТНЫЙ: это «пул не отдал», а НЕ брак сборки
                # (данные чисты, гео просто недоразмечено) — не путать с carve-комом.
                print(
                    f"⚠️ НЕУДАЧ: {geo} разметка упёрлась в пул "
                    f"({len(tentative)} провалов без единого тега) — гео недозрело, не собрано",
                    flush=True,
                )
                tentative = []
                stopped = True  # выйти чисто, не портить счётчики
                break
    _atomic_json(tags_fn, tagged)  # атомарно: temp+rename, kill не бьёт файл
    # засчитать провалы прохода в дед-леттер (только если НЕ систематический откат)
    new_dead = []
    for fid in tentative:
        fails[fid] = fails.get(fid, 0) + 1
        if fails[fid] >= DEAD_AT:
            new_dead.append(fid)
    _atomic_json(fails_fn, fails)
    if new_dead:
        print(
            f"{geo}: дед-леттер {len(new_dead)} непереваримых мух {new_dead} (>={DEAD_AT} провалов)",
            flush=True,
        )
    if stopped:
        print(f"{geo}: STOP — сохранено {new_n} новых, чисто вышел", flush=True)
        return new_n

    # ЧЕСТНЫЙ ОСТАТОК считаем ДО тяжёлой части: пока гео не дозрел (remaining>0) —
    # консолидацию/виды НЕ гоняем (нужны только зрелому гео для ship; на больших гео
    # consolidate одним запросом на тысячи ярлыков = таймауты и сожжённый пул впустую).
    dead = {
        fid for fid, c in fails.items() if c >= DEAD_AT
    }  # пересчёт после инкрементов
    remaining = len(load_flies(geo, None, exclude={r["id"] for r in tagged} | dead))
    if remaining > 0:
        print(
            f"\n{geo}: +{new_n} новых → всего {len(tagged)} мух (виды при дозревании) "
            f"remaining={remaining}",
            flush=True,
        )
        return new_n

    # ВИДЫ через CARVE (замена consolidate): группировка по фасет-семье → carve плотных семей
    # по ТЕКСТАМ мух / тонкий хвост как facet. Инвертированный индекс строится внутри.
    # ⚠ имя НЕ `fails` — так зовётся дед-леттер мух выше в этой же функции.
    # ⛔ Ниже файл гео пересобирается заново, поэтому перед записью обязателен
    # `carry_forward` — см. его докстринг. Без него дозревание гео стирает нажитое.
    run_fails = (
        []
    )  # неудачи ЭТОГО прогона (carve не разобрал) → в файл гео, в отчёт, в перепрогон
    views, shelves, prochee = build_views_by_carve(tagged, run_fails)

    # индекс по сущности (для видов-страниц вида «CPF»: всё, где CPF — цель/требование/обход)
    ent_index = {}
    for r in tagged:
        for e in r["sushnosti"]:
            ent_index.setdefault(e["imya"], []).append({"id": r["id"], "rol": e["rol"]})

    os.makedirs("out_facet", exist_ok=True)
    page = {
        "geo": geo,
        # джоб1: плотные тематические страницы (совместимо с pages.py)
        "views_by_task": [{"zadacha": z, "items": its} for z, its in views.items()],
        # джоб2: хвост-антологии по глобальной таксономии (Ф3 pages.py их рендерит)
        "shelves": [{"shelf": sh, "items": its} for sh, its in shelves.items()],
        "prochee": prochee,  # park-ведро непокрытого — сигнал роста таксономии
        "taxonomy_version": tax.VERSION,
        "entity_index": {k: v for k, v in ent_index.items() if len(v) > 1},
    }
    carry_forward(f"out_facet/{geo}.json", page, geo)
    # НЕУДАЧИ прогона — рядом с данными. Пусто = гео разобрано честно; непусто = гео
    # собрано с откатом (нарезки не было) и ЖДЁТ ПЕРЕПРОГОНА. Читают: отчёт и пульт.
    if run_fails:
        page["fails"] = run_fails
    _atomic_json(f"out_facet/{geo}.json", page)  # атомарно

    print(
        f"\n{geo}: +{new_n} новых → всего {len(tagged)} мух, видов-задач {len(views)}, "
        f"полок {len(shelves)}, прочее {len(prochee)}, "
        f"сущностей-кросс {len(page['entity_index'])} → out_facet/{geo}.json remaining=0"
        + (
            f"\n⚠️ НЕУДАЧ: {len(run_fails)} — carve не разобрал "
            f"{sum(f.get('flies', 0) for f in run_fails)} мух "
            f"({', '.join(str(f.get('family') or '?') for f in run_fails[:3])}). "
            f"Гео собрано ОТКАТОМ, нужен перепрогон."
            if run_fails
            else ""
        ),
        flush=True,
    )
    return new_n


# ── ПОЛКА ВИДУ: тот же рот `assign`, что раскладывает хвост (заказ юзера 13.08) ──
# Ось адресации у видов — формулировка метки задачи, и уровня темы у них не было: у Греции
# 62 вида при 8 настоящих темах, а хаб вываливал 63 ссылки плоским списком. На этом поле
# стоят плитки хаба и довод CTA по теме.
# ⛔ Способ ОДИН и он существующий: тот же рот, та же закрытая таксономия, тот же сосок мозга.
# Своей механики (векторы, центры полок) не заводить — заказа не было, а второй способ = вторая
# правда.
# На вход идут МЕТКИ видов, а не тексты мух: метка коротка, и полка по ней видна (замер 13.08:
# 60 меток из 62 у Греции опознавались даже словарём).
ASSIGN_VIEW_BATCH = 90  # меток в пачку: они короткие, окно соска держит с запасом

ASSIGN_VIEW_SYS = (
    "Ниже названия тем страниц гида (id: название). Отнеси КАЖДУЮ ровно к ОДНОЙ полке "
    "закрытой таксономии. Отвечай КЛЮЧОМ полки.\n"
    + "\n".join(f"{k} — {name}: {desc}" for k, name, desc in tax.SHELVES)
    + "\nНи одна не подходит → ключ 'prochee' (сигнал дырки, не злоупотребляй).\n"
    'СТРОГО JSON: {"map":{"0":"<ключ полки>",...}}'
)


def assign_views(geo, fails=None):
    """Полка каждому виду-странице гео. Пишет `view["shelf"]` в out_facet/<geo>.json.

    Работой считаются только виды БЕЗ полки — повторный запуск ключей не тратит.
    Вид, на который рот не ответил или ответил неизвестным ключом, остаётся БЕЗ полки: молча
    приписать ближайшую значило бы соврать, а пустое поле видно и в пульте, и в сборке.
    """
    out_fn = f"out_facet/{geo}.json"
    page = json.load(open(out_fn, encoding="utf-8"))
    views = [
        (i, v)
        for i, v in enumerate(page.get("views_by_task") or [])
        if len(v.get("items") or []) >= tax.PAGE_MIN
    ]
    todo = [(i, v) for i, v in views if not v.get("shelf")]
    if not todo:
        print(f"{geo}: полки у всех {len(views)} видов — пропуск", flush=True)
        return 0
    by_key = {k: name for k, name, _ in tax.SHELVES}
    done = unknown = 0
    unknown_keys = []
    for s in range(0, len(todo), ASSIGN_VIEW_BATCH):
        if os.path.exists("RUNNER_STOP"):  # стоп МЕЖДУ пачками: сделанное не теряем
            print(f"  стоп между пачками на {s}/{len(todo)}", flush=True)
            break
        chunk = todo[s : s + ASSIGN_VIEW_BATCH]
        idx = {str(j): (v.get("zadacha") or "") for j, (_i, v) in enumerate(chunk)}
        res = call(
            json.dumps(idx, ensure_ascii=False), ASSIGN_VIEW_SYS, consumer="assign"
        )
        m = (res or {}).get("map") or {}
        if not m:
            if fails is not None:  # сбой ВИДЕН, а не проглочен
                fails.append(
                    {
                        "step": "assign_views",
                        "geo": geo,
                        "batch": s // ASSIGN_VIEW_BATCH,
                    }
                )
            continue
        for j, (_i, v) in enumerate(chunk):
            k = (m.get(str(j)) or "").strip()
            if k in by_key:
                v["shelf"] = by_key[k]
                done += 1
            else:
                unknown += 1
                if k and k not in unknown_keys:
                    unknown_keys.append(k)  # ЧТО пришло, а не только сколько
    if done:
        _atomic_json(out_fn, page)
    print(
        f"{geo}: видов {len(views)}, без полки было {len(todo)} -> разложено {done}"
        + (
            f", неопознанный ключ {unknown} ({','.join(unknown_keys[:5])})"
            if unknown
            else ""
        ),
        flush=True,
    )
    return done


def run_assign_tail(geo):
    """Только джоб2 на УЖЕ построенном out_facet: хвост = тегнутые мухи, НЕ доходящие
    ни до одной страницы (все их виды <PAGE_MIN; в старом формате синглы лежат видами
    по 1 мухе — «вне видов» их не ловит). Раскладка по таксономии → shelves/prochee
    мёржатся в out_facet/<geo>.json. БЕЗ пере-карва (полный run() на дозревшем гео
    пережёвывает carve заново — дорого).
    """
    tagged = json.load(open(f"tags/{geo}.json", encoding="utf-8"))
    by_id = {r["id"]: r for r in tagged}
    out_fn = f"out_facet/{geo}.json"
    page = json.load(open(out_fn, encoding="utf-8"))
    on_page = {
        it["id"]
        for v in page.get("views_by_task", [])
        if len(v["items"]) >= tax.PAGE_MIN
        for it in v["items"]
    }
    tail = [fid for fid in by_id if fid not in on_page]
    run_fails = []  # сбои раскладки этого прогона → в файл гео, в отчёт
    shelves, prochee = assign_tail(tail, by_id, run_fails)
    page["shelves"] = [{"shelf": sh, "items": its} for sh, its in shelves.items()]
    page["prochee"] = prochee
    page["taxonomy_version"] = tax.VERSION
    if run_fails:
        page["fails"] = (page.get("fails") or []) + run_fails
    _atomic_json(out_fn, page)
    print(
        f"{geo}: хвост {len(tail)} → полок {len(shelves)} "
        f"({sum(len(v) for v in shelves.values())} членств), прочее {len(prochee)}",
        flush=True,
    )


def run_reassign_shelf(geo, shelf_name):
    """Пере-разложить ТОЛЬКО одну полку по текущей таксономии. Остальные не трогаем.

    ⭐ Зачем отдельный режим (2026-08-13). При смене набора полок смысл меняется не у всего
    хвоста, а у той полки, которую разобрали. Полный `--assign-tail` по всем гео — 120 вызовов
    и потолок 1440 обращений к Google; целевой по одной полке — 15 вызовов и потолок 180.
    Разница в восемь раз, и она в общем ресурсе ключей, а не в удобстве.

    ⚠️ ЧЕСТНОСТЬ МЕТКИ: помечаем `taxonomy_version` текущей версией, но рядом пишем
    `taxonomy_reassigned` — какие именно полки пере-разложены. Иначе файл заявлял бы полное
    соответствие новой таксономии, хотя мухи из ДРУГИХ полок под уточнённые границы не
    пересматривались (например «погода» могла осесть в транспорте, а её место — в туризме).
    """
    tagged = json.load(open(f"tags/{geo}.json", encoding="utf-8"))
    by_id = {r["id"]: r for r in tagged}
    out_fn = f"out_facet/{geo}.json"
    page = json.load(open(out_fn, encoding="utf-8"))
    shelves = {
        s["shelf"]: list(s.get("items") or []) for s in page.get("shelves") or []
    }
    target = shelves.pop(shelf_name, None)
    if target is None:
        print(f"{geo}: полки «{shelf_name}» нет — пропуск", flush=True)
        return 0
    fids = [it["id"] for it in target if it.get("id") in by_id]
    lost = len(target) - len(fids)  # мухи, которых нет в tags: считаем и говорим
    run_fails = []
    fresh, prochee = assign_tail(fids, by_id, run_fails)
    for sh, its in fresh.items():
        have = {it["id"] for it in shelves.get(sh, [])}
        shelves.setdefault(sh, []).extend(it for it in its if it["id"] not in have)
    page["shelves"] = [
        {"shelf": sh, "items": its} for sh, its in shelves.items() if its
    ]
    old_pro = {it["id"] for it in (page.get("prochee") or [])}
    page["prochee"] = (page.get("prochee") or []) + [
        it for it in prochee if it["id"] not in old_pro
    ]
    page["taxonomy_version"] = tax.VERSION
    page["taxonomy_reassigned"] = sorted(
        set(page.get("taxonomy_reassigned") or []) | {shelf_name}
    )
    if run_fails:
        page["fails"] = (page.get("fails") or []) + run_fails
    _atomic_json(out_fn, page)
    print(
        f"{geo}: «{shelf_name}» {len(fids)} мух → "
        + ", ".join(f"{sh}: {len(its)}" for sh, its in sorted(fresh.items()))
        + f"; прочее +{len(prochee)}"
        + (f"; без текста {lost}" if lost else ""),
        flush=True,
    )
    return len(fids)


if __name__ == "__main__":
    # --mature: список зрелых гео в JSON. Зовёт ship (гейт публикации) — тут, а не у себя,
    # чтобы правило «что такое зрелое» жило рядом с тем, кто по нему берёт работу.
    if "--mature" in sys.argv:
        print(json.dumps(mature_geos(), ensure_ascii=False))
        sys.exit(0)
    if len(sys.argv) < 2:
        print(
            "usage: facet.py <geo> [--limit N] [--assign-tail] [--assign-flies] "
            "[--deals-only <ключ раздела>] "
            '[--assign-views] [--reassign-shelf "<полка>"] | facet.py --mature'
        )
        sys.exit(1)
    geo = sys.argv[1]
    if "--deals-only" in sys.argv:
        # КОНТРОЛЬ метода: один вызов рта на одну пару «гео × раздел» (канон §0.15).
        deals_for_pair(geo, sys.argv[sys.argv.index("--deals-only") + 1], [])
        sys.exit(0)
    if "--assign-flies" in sys.argv:
        # шаг «раздел мухам» — ось нарезки (канон §0.15). Запускается ПУЛЬТОМ.
        assign_fly_shelves(geo, [])
        sys.exit(0)
    if "--assign-views" in sys.argv:
        assign_views(geo, [])
        sys.exit(0)
    if "--reassign-shelf" in sys.argv:
        run_reassign_shelf(geo, sys.argv[sys.argv.index("--reassign-shelf") + 1])
        sys.exit(0)
    if "--assign-tail" in sys.argv:
        run_assign_tail(geo)
        sys.exit(0)
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )
    run(geo, limit=limit)
