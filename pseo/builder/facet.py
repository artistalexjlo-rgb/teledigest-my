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

Запуск (VPS): /root/embed_ab/venv/bin/python facet.py <geo> [--limit N]
Плуминг Gemini (пейсинг/квота/429/IPv4) — внутри keybroker.call (сосок мозга). build.py снесён.
"""

import json
import os
import re
import sqlite3
import sys

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

CONSOLIDATE_SYS = (
    "Дан список ЗАДАЧ (ярлыков), собранных по одной мухе, поэтому одна задача записана разными "
    "словами и в разной дробности. Сведи к СОГЛАСОВАННОМУ набору задач ОДНОГО уровня дробности "
    "(как оглавление гайда: 'Документы и виза', 'Банк и деньги', 'SIM и интернет', 'Транспорт', "
    "'Жильё', 'Покупки', 'Здоровье', 'Билеты и развлечения', 'Работа и налоги', 'Безопасность', "
    "'Почта и посылки', 'Обмен и переводы'…). Только ОБЪЕДИНЯЙ существующие, НЕ выдумывай новых, "
    "НИ ОДИН входной ярлык не потеряй.\n"
    'Верни СТРОГО JSON: {"map": {"<вход>": "<согласованная задача>", ...}}.'
)


def _atomic_json(path, obj):
    """Запись через temp+rename: kill в любой момент не оставит недописанный/битый файл."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)  # атомарно на том же fs


def load_flies(geo, limit=None, exclude=None):
    """Мухи гео (junk отсеян), ИСКЛЮЧАЯ уже тегнутые (exclude=set id) → следующая порция.
    limit = сколько НОВЫХ мух за прогон (бережное потребление, не весь корпус разом)."""
    exclude = exclude or set()
    m = sqlite3.connect(DB)
    rows = m.execute(
        "SELECT id, ai_lesson FROM extracted_patterns "
        "WHERE country=? AND ai_lesson IS NOT NULL AND length(ai_lesson)>? ORDER BY id",
        (geo, MIN_LEN),
    ).fetchall()
    m.close()
    rows = [r for r in rows if not is_junk(r[1]) and r[0] not in exclude]
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


def _norm(s):
    return " ".join(s.split()).rstrip(".").strip()


def consolidate(labels):
    """Свод ярлыков к одному грайну. БАТЧАМИ по 120 — на больших гео ярлыков тысячи,
    один запрос на всё = таймаут + модель физически не покрывает map."""
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return {x: x for x in uniq}
    mp = {}
    for i in range(0, len(uniq), 120):
        batch = uniq[i : i + 120]
        out = call(
            json.dumps(batch, ensure_ascii=False),
            CONSOLIDATE_SYS,
            consumer="consolidate",
        )
        mp.update((out or {}).get("map") or {})
    return {x: _norm(mp.get(x, x)) for x in uniq}


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
    # dead-letter: мухи, которые facet_one провалил как "bad" >=DEAD_AT раз (непереваримый
    # контент). Иначе одна битая муха в хвосте держит зрелость гео вечно (remaining застревает
    # на 1). Инфра-сбои (бюджет/429) НЕ считаем — их отсеивает status="infra".
    DEAD_AT = 3
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
    for fid, lesson in flies:
        if os.path.exists(
            "RUNNER_STOP"
        ):  # чистый стоп МЕЖДУ мухами — сохраним что успели
            stopped = True
            break
        status, r = facet_one(fid, lesson)
        if status == "ok":
            tagged.append(r)
            new_n += 1
            print(
                f"  + {', '.join(r['zadachi'])[:48]:50} :: {r['perevod'][:52]}",
                flush=True,
            )
        else:  # bad или infra
            tentative.append(fid)
            # СИСТЕМАТИКА: 0 тегнуто + >=3 провала = бюджет/инфра сдохли, НЕ вина мух →
            # откат (tentative не применяем), стоп прохода. Иначе битую муху виним честно.
            if new_n == 0 and len(tentative) >= 3:
                print(
                    f"{geo}: {len(tentative)} провалов без единого тега — бюджет/инфра, откат+стоп",
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

    # консолидация ЯРЛЫКОВ задач (гео ДОЗРЕЛ; батчами — на больших гео тысячи ярлыков)
    cmap = consolidate([z for r in tagged for z in r["zadachi"]])
    for r in tagged:
        r["zadachi"] = sorted({cmap.get(z, z) for z in r["zadachi"]})

    # ВИДЫ по задаче (инвертированный индекс): муха с N задачами → в N видах
    views = {}
    for r in tagged:
        for z in r["zadachi"]:
            views.setdefault(z, []).append(
                {
                    "id": r["id"],
                    "text": r["perevod"],
                    "sushnosti": r["sushnosti"],
                    "mesto": r["mesto"],
                    "uslovie": r["uslovie"],
                }
            )

    # индекс по сущности (для видов-страниц вида «CPF»: всё, где CPF — цель/требование/обход)
    ent_index = {}
    for r in tagged:
        for e in r["sushnosti"]:
            ent_index.setdefault(e["imya"], []).append({"id": r["id"], "rol": e["rol"]})

    os.makedirs("out_facet", exist_ok=True)
    page = {
        "geo": geo,
        "views_by_task": [{"zadacha": z, "items": its} for z, its in views.items()],
        "entity_index": {k: v for k, v in ent_index.items() if len(v) > 1},
    }
    _atomic_json(f"out_facet/{geo}.json", page)  # атомарно

    print(
        f"\n{geo}: +{new_n} новых → всего {len(tagged)} мух, видов-задач {len(views)}, "
        f"сущностей-кросс {len(page['entity_index'])} → out_facet/{geo}.json remaining=0",
        flush=True,
    )
    return new_n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: facet.py <geo> [--limit N]")
        sys.exit(1)
    geo = sys.argv[1]
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )
    run(geo, limit=limit)
