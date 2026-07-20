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
    'СТРОГО JSON: {"intents":[{"name":"<конкретная тема>","ids":["0",...]}]}'
)


def _first_word(z):
    p = z.split()
    return p[0].lower() if p else z.lower()


def carve_family(fids, by_id):
    """Пачка мух семьи (ТЕКСТЫ perevod) → LLM вычленяет подпункты + присваивает. [{name, ids}].
    Батчами по CARVE_BATCH. None/сбой пачки → каждая её муха как свой вид (fallback, НЕ теряем).
    """
    out = []
    for s in range(0, len(fids), CARVE_BATCH):
        chunk = fids[s : s + CARVE_BATCH]
        idx = {str(j): by_id[fid]["perevod"] for j, fid in enumerate(chunk)}
        res = call(json.dumps(idx, ensure_ascii=False), CARVE_SYS, consumer="carve")
        if not res or not res.get("intents"):
            for fid in chunk:  # fallback: муха как есть (её первая задача = имя)
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
    "Ниже разрозненные советы путешественников/экспатов (id: текст) — каждый самостоятелен, "
    "НЕ схлопывай и НЕ выкидывай. Разложи КАЖДЫЙ по ЗАКРЫТОЙ таксономии:\n"
    "ПОЛКИ (можно НЕСКОЛЬКО, минимум 1): " + " | ".join(tax.SHELF_NAMES) + "\n"
    "ТИП (РОВНО один): " + " | ".join(tax.TYPE_NAMES) + "\n"
    "Не лезет ни на одну полку → полка '"
    + tax.PROCHEE
    + "' (сигнал дырки, не злоупотребляй).\n"
    'СТРОГО JSON: {"assign":{"0":{"shelves":["..."],"type":"..."},...}}'
)


def assign_tail(tail_fids, by_id):
    """Тонкий хвост → раскладка по таксономии. Возвращает (shelves {полка: [items]},
    prochee [items]). Потерь НЕТ: сбой пачки / непокрытие / не-возврат мухи → в prochee.
    """
    fids = list(tail_fids)
    shelves, prochee = {}, []

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
        chunk = fids[s : s + CARVE_BATCH]
        idx = {str(j): by_id[fid]["perevod"] for j, fid in enumerate(chunk)}
        res = None
        for _ in range(3):
            res = call(
                json.dumps(idx, ensure_ascii=False), ASSIGN_SYS, consumer="assign"
            )
            if res and res.get("assign"):
                break
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
    return shelves, prochee


def build_views_by_carve(tagged):
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
    carved_fids = set()

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

    for w, fset in fams.items():
        fids = list(fset)
        if len(fids) < MIN_CARVE:
            continue  # тонкие семьи → хвост-раскладка ниже (НЕ сырые синглтоны)
        for it in carve_family(fids, by_id):  # плотная семья: carve по текстам
            if len(it["ids"]) < 2:
                continue  # одиночный интент carve = тонкий абзац → в хвост-раскладку, не 1-мушь-страница
            for fid in it["ids"]:
                add(it["name"], fid)
                carved_fids.add(fid)

    # страховка: дедуп мух в карв-виде по id (одна муха могла попасть дважды на стыке семей)
    for name, items in views.items():
        seen, uniq = set(), []
        for it in items:
            if it["id"] not in seen:
                seen.add(it["id"])
                uniq.append(it)
        views[name] = uniq

    # ХВОСТ = мухи, не попавшие ни в один плотный карв-вид → раскладка по таксономии
    tail_fids = [fid for fid in by_id if fid not in carved_fids]
    shelves, prochee = assign_tail(tail_fids, by_id)
    return views, shelves, prochee


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

    # ВИДЫ через CARVE (замена consolidate): группировка по фасет-семье → carve плотных семей
    # по ТЕКСТАМ мух / тонкий хвост как facet. Инвертированный индекс строится внутри.
    views, shelves, prochee = build_views_by_carve(tagged)

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
    _atomic_json(f"out_facet/{geo}.json", page)  # атомарно

    print(
        f"\n{geo}: +{new_n} новых → всего {len(tagged)} мух, видов-задач {len(views)}, "
        f"полок {len(shelves)}, прочее {len(prochee)}, "
        f"сущностей-кросс {len(page['entity_index'])} → out_facet/{geo}.json remaining=0",
        flush=True,
    )
    return new_n


PAGE_MIN = 4  # гейт страницы (зеркало pages.py): вид <4 мух страницей не станет


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
        if len(v["items"]) >= PAGE_MIN
        for it in v["items"]
    }
    tail = [fid for fid in by_id if fid not in on_page]
    shelves, prochee = assign_tail(tail, by_id)
    page["shelves"] = [{"shelf": sh, "items": its} for sh, its in shelves.items()]
    page["prochee"] = prochee
    page["taxonomy_version"] = tax.VERSION
    _atomic_json(out_fn, page)
    print(
        f"{geo}: хвост {len(tail)} → полок {len(shelves)} "
        f"({sum(len(v) for v in shelves.values())} членств), прочее {len(prochee)}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: facet.py <geo> [--limit N] [--assign-tail]")
        sys.exit(1)
    geo = sys.argv[1]
    if "--assign-tail" in sys.argv:
        run_assign_tail(geo)
        sys.exit(0)
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )
    run(geo, limit=limit)
