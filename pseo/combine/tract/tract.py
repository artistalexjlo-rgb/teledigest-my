"""ТРАКТ «ТЕМА И ПОДТЕМА» (канон §0.19) — шаги 3 и 4 взамен старой нарезки.

Шаг 3 РАЗМЕТКА: муха → перевод + тема (одна из 13) + подтема. Пачка 25.
Шаг 4 СПИСКИ:   мухи темы → списки с именами. Пачка 90; корпус в том виде, который уже
                понимает сборка: `views_by_task` и `shelves` (остаток).

⛔ ВСЁ ПИШЕТСЯ В `tests/` И ТОЛЬКО ТУДА (заказ юзера 20.08: прогоны новой схемы — в тестовую
папку). Боевые `tags/` и `out_facet/` не трогаются, пока схема не принята: 21.08 я направил
разметку в боевую папку, там смешались 146 старых записей с 42 новыми, а свод переписал боевой
корпус Чехии. Каталог задан ОДНОЙ константой ниже — второго места, где это решается, нет.

⛔ Числа списков у рта НЕ спрашиваем и вилок не задаём — их место в коде. Пороги страницы
живут ЗДЕСЬ же, ниже: `PAGE_MIN` и `PAGE_MAX`.

Оба шага ДОБИРАЮТ работу: размеченные мухи пропускаются, корпус пишется чекпоинтами, поэтому
повторный запуск ключей не тратит, а СТОП между пачками не теряет сделанного.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⛔ Тракт НЕ импортирует отменённую схему (`facet`, `dedup`): чтение корпуса и
# вектора живут своими модулями, чтобы старый код не попадался под руку (рамка 24.08).
import vectors  # noqa: E402
from corpus import load_flies  # noqa: E402
from keybroker import call  # noqa: E402

# ⛔ ЕДИНСТВЕННОЕ место, где решается «куда пишем». Боевые каталоги рядом (`tags/`,
# `out_facet/`) — их новый тракт не касается.
TESTS = "tests"

# Порог схлопывания почти-копий. Замер 20.08 на боевом корпусе: 0.86 прячет за счётчиком
# до четверти содержимого (в одной группе из 44 советов Черногории слиплись налог,
# регистрация и ответственность владельца жилья), 0.93 берёт 3,5% — настоящие почти-копии.
# Судить порог по протоколу схлопывания (`tests/dedup/<гео>.txt`), а не по счётчику.
COLLAPSE_THR = 0.93

# Верх страницы: толще пятнадцати абзацев страниц не бывает (правило юзера). Низ —
# `PAGE_MIN` ниже, в справочнике тракта.
PAGE_MAX = 15

# ── СПРАВОЧНИК ТРАКТА: темы, пороги, запреты. Всё своё, из старой таксономии сюда
# перенесены ровно три вещи, которыми пользовался тракт (27.08).
#
# ТЕМЫ — тринадцать плиток хаба. Здесь только КЛЮЧИ: они идут в адрес и роту. Имена по
# языкам (включая русский) живут ОДНИМ файлом — `themes.json`, рядом с сайтом, а не в коде:
# исключения для русского языка в коде тракта больше нет ни в одном месте (27.08).
THEME_KEYS = [
    "border",
    "visa",
    "finance",
    "transport",
    "docs",
    "safety",
    "customs",
    "digital",
    "tourism",
    "housing",
    "shopping",
    "work",
    "health",
]

# Порог «есть ли ветка»: пачка меньше — идёт в остаток. К делению уже названной ветки
# отношения НЕ имеет (PLAN.md, «Ветвление одной темы»).
PAGE_MIN = 4

# Сборные имена: ими рот прикрывает лень нарезки, и под таким именем собирается корзина.
# ⛔ Список АНГЛИЙСКИЙ — имена рождаются по-английски с 25.08. Русский список остался в
# архиве вместе с русской эпохой.
JUNK_NAMES = (
    "other",
    "others",
    "miscellaneous",
    "misc",
    "general",
    "general information",
    "general info",
    "general tips",
    "general advice",
    "useful information",
    "useful tips",
    "additional information",
    "additional info",
    "various",
    "various questions",
    "everything else",
    "life abroad",
    "living abroad",
    "tips for travellers",
    "tips for travelers",
    "advice for tourists",
)


def bad_name(z):
    """Имя — брак нарезки? Возвращает причину строкой или None.

    Причину возвращаем, а не True: она уходит в лог. Молчаливый отсев на этом проекте уже
    прятал 33 пустых хаба целый год.
    """
    t = (z or "").strip().lower()
    if not t:
        return "пустое имя"
    core = t.strip(" .:;!?-—()[]«»\"'")
    for w in JUNK_NAMES:
        if core == w or core.startswith(w + " ") or core.startswith(w + ","):
            return f"сборное слово: {w}"
    if core in {k.replace("_", " ") for k in THEME_KEYS}:
        return "имя повторяет тему"
    return None


def slug(t):
    """Адрес из английского имени. Пусто — значит адреса НЕТ, и страницу не публикуем.

    ⛔ Фолбэка вроде «tema» тут не будет: он давал всем страницам гео ОДИН адрес, и они
    молча затирали друг друга — 90 страниц на язык вместо 1843 (случай 08.08).
    """
    return re.sub(r"[^a-z0-9]+", "-", (t or "").lower()).strip("-")[:40]


# Остаток темы, набравшийся на ветку, выходит служебной веткой.
# ⛔ Имя АНГЛИЙСКОЕ, как и все имена прохода А: русское «Разное» уехало бы в корпус и
# осталось бы единственной строкой сайта, которую звено 6 не переводит. «Разное» —
# это перевод, и появляется он там же, где остальные заголовки.
# ⛔ Запрет сборных имён (`bad_label`) — про имена ОТ РТА: ими он прикрывал лень нарезки.
# Здесь имя ставит код и честно называет то, что не сложилось, — это не то же самое.
MISC_TITLE, MISC_SLUG = "Other", "misc"

MARK_BATCH = 25  # мух в запрос разметки: 25 переводов ≈ 10К символов ответа


def mark_sys():
    """Промпт звена 3: тема и подтема, ПО-АНГЛИЙСКИ. Перевода не просим.

    Темы — закрытый список ключей из таксономии: рот выбирает ключ, а не выдумывает
    название. Ключи латинские и говорят сами за себя (`visa`, `transport`, `housing`),
    поэтому список идёт роту как есть.
    """
    names_map = ", ".join(THEME_KEYS)
    return (
        "You are a MARKER of a ready advice, NOT an author. Do not rewrite, shorten "
        "or summarise the advice.\n"
        'Input is JSON {"0": "<advice>", ...}. Mark EVERY one.\n'
        'Return STRICT JSON: {"rows": [{"i": "<index>", '
        '"theme": "…", "subtheme": "…"}, ...]}\n'
        "  Each advice is a SEPARATE object: a broken record costs one advice, "
        "not the whole batch.\n"
        "  i       — the key of the advice from the input.\n"
        f"  tema    — EXACTLY ONE key from this list: {names_map}.\n"
        "  podtema — 2-6 words IN ENGLISH: what a person comes for. Like a human "
        'query: "car rental", "paying for a taxi", "airport transfer". '
        "FORBIDDEN: repeating the theme name and broad rubric words "
        "(transport, money, documents).\n"
        "JSON only, no explanations."
    )


# Имён в теме: границы из старого промпта (`DEALS_MAX = 15`, «дел от 5 до 15») — число
# не выдумано заново, оно бегало в проде. Больше 15 — это уже не список, а простыня.
NAMES_MIN, NAMES_MAX = 5, 15
ASSIGN_BATCH = 90  # советов в запрос прохода Б: тексты в один запрос не влезают


def names_sys():
    """Проход А: закрытый список имён темы. Дословно `DEALS_SYS`, по-английски."""
    return (
        "Below are task labels from ONE theme of a country guide (number: label), with "
        "the number of advices in brackets. Compose a CLOSED list of what people really "
        f"come with. From {NAMES_MIN} to {NAMES_MAX} items.\n"
        "NAME RULES: a name is ONE human request, it becomes the PAGE TITLE. Different "
        "wordings of the same thing are ONE name. Do NOT mix different things "
        "(student visa != work visa; tax id != citizenship != residence permit). "
        "FORBIDDEN: 'Other', 'Miscellaneous', 'General tips', 'Useful information', "
        "'Specifics' without a subject, and a name equal to the theme name itself. "
        "English, 2-6 words.\n"
        'STRICT JSON: {"names": ["<name>", ...]}'
    )


def assign_sys():
    """Проход Б: присваивание из закрытого списка. Дословно `DEAL_ASSIGN_SYS`."""
    return (
        "Below is a CLOSED list of names (number: name), then advices (number: text). "
        "Assign EVERY advice to exactly ONE name FROM THE LIST — do not invent new ones. "
        'If none of them fits, put "0".\n'
        'STRICT JSON: {"map": {"<advice number>": "<name number or 0>"}}'
    )


def collapsed_path(geo):
    return f"{TESTS}/dedup/{geo}.json"


def collapse(geo):
    """Шаг 2: схлопнуть почти-копии гео ДО разметки — размечать будем представителей.

    Ключей НЕ тратит: вектора готовые, из `local_vec.db` свипера.

    ⭐ ПИШЕТ ПРОТОКОЛ, а не счётчик. `tests/dedup/<гео>.txt` — по каждой склейке текст
    представителя, который остаётся, и полные тексты тех, кого он проглотил. Иначе шаг
    неизмерим: «схлопнуто 27» не отличает настоящие повторы от съеденного содержимого, а
    ровно на этом 0.86 и прокололся.
    """
    flies = load_flies(geo)
    texts = {i: t for i, t in flies}
    ids = [i for i, _ in flies]
    vv = vectors.load_vecs(ids)
    no_vec = [i for i in ids if i not in vv]
    groups = vectors.groups_all(ids, vv, COLLAPSE_THR)
    multi = [g for g in groups if len(g) > 1]
    swallowed = sum(len(g) - 1 for g in multi)
    recs = []
    for g in sorted(groups, key=lambda g: -len(g)):
        rep = max(g, key=lambda i: len(texts[i]))
        recs.append({"rep": rep, "ids": g})

    os.makedirs(f"{TESTS}/dedup", exist_ok=True)
    with open(collapsed_path(geo), "w", encoding="utf-8") as fh:
        json.dump(
            {"geo": geo, "thr": COLLAPSE_THR, "no_vec": len(no_vec), "groups": recs},
            fh,
            ensure_ascii=False,
        )
    proto = f"{TESTS}/dedup/{geo}.txt"
    with open(proto, "w", encoding="utf-8") as fh:
        print(f"# {geo}: схлопывание почти-копий, порог {COLLAPSE_THR}", file=fh)
        print(
            f"# мух {len(ids)}, без вектора {len(no_vec)}, "
            f"склеек {len(multi)}, схлопнуто {swallowed}",
            file=fh,
        )
        for r in recs:
            if len(r["ids"]) < 2:
                continue
            print("", file=fh)
            print(f"=== группа из {len(r['ids'])} ===", file=fh)
            print(f"[ОСТАЁТСЯ {r['rep']}] {texts[r['rep']]}", file=fh)
            for i in r["ids"]:
                if i != r["rep"]:
                    print(f"  [СХЛОПНУТА {i}] {texts[i]}", file=fh)
    sizes = sorted((len(g) for g in multi), reverse=True)[:8]
    print(
        f"{geo}: мух {len(ids)}, без вектора {len(no_vec)}, склеек {len(multi)}, "
        f"схлопнуто {swallowed} ({100.0 * swallowed / max(len(ids), 1):.1f}%), "
        f"крупнейшие {sizes}",
        flush=True,
    )
    print(f"  к разметке пойдут {len(recs)} представителей", flush=True)
    print(f"  протокол глазами -> {proto}", flush=True)


def undone(geo, ids, base=""):
    """Из мух гео — те, которые шаг разметки РЕАЛЬНО возьмёт. ОДНО место, где это решается.

    ⛔ ЗАЧЕМ ОТДЕЛЬНОЙ ФУНКЦИЕЙ. Правило «кого берёт разметка» жило дважды: здесь и своей
    арифметикой в пульте («мухи гео минус размеченные»). Пока правила совпадали, копия не
    мешала; 22.08 сюда добавился фильтр представителей — и пульт про него не узнал: Греция
    после полной разметки 751 из 751 висела с «14 мух» и звала шаг вхолостую. Теперь пульт
    зовёт ЭТУ функцию, и разойтись им нечем.

    `ids` даёт вызывающий (у пульта они уже прочитаны одним проходом по базе) — второго
    похода в базу функция не делает. `base` — корень данных: рот бежит с cwd=BRAIN и не
    передаёт ничего, пульт зовёт из своего процесса и передаёт BRAIN явно.
    """
    tagged = set()
    fn = os.path.join(base, TESTS, "tags", f"{geo}.json")
    if os.path.exists(fn):
        try:
            tagged = {r["id"] for r in json.load(open(fn, encoding="utf-8"))}
        except Exception:
            tagged = set()
    sp = os.path.join(base, TESTS, "dedup", f"{geo}.json")
    if os.path.exists(sp):
        try:
            sg = json.load(open(sp, encoding="utf-8"))
            reps = {g["rep"] for g in sg.get("groups") or []}
            ids = [
                i for i in ids if i in reps
            ]  # проглоченных разметка не увидит НИКОГДА
        except Exception:
            pass
    return [i for i in ids if i not in tagged]


def mark(geo, limit=None):
    """Шаг 3: разметить мух гео. Добирает только неразмеченных.

    Если схлопывание (шаг 2) прошло — размечаем ТОЛЬКО представителей, а сколько мух за
    каждым, несём числом `n`: платить рту за пересказ одного и того же незачем.
    """
    fn = f"{TESTS}/tags/{geo}.json"
    done = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else []
    za_rep = {}
    sp = collapsed_path(geo)
    if os.path.exists(sp):
        sg = json.load(open(sp, encoding="utf-8"))
        za_rep = {g["rep"]: len(g["ids"]) for g in sg.get("groups") or []}
        print(
            f"{geo}: схлопывание есть (порог {sg.get('thr')}), "
            f"представителей {len(za_rep)}",
            flush=True,
        )
    else:
        print(f"{geo}: схлопывания нет — размечаем все мухи как есть", flush=True)
    all_flies = load_flies(geo)
    beru = set(undone(geo, [i for i, _t in all_flies]))  # ТА ЖЕ функция, что у пульта
    flies = [f for f in all_flies if f[0] in beru]
    if limit:
        flies = flies[: int(limit)]
    if not flies:
        print(f"{geo}: размечать нечего (уже есть {len(done)})", flush=True)
        return
    print(f"{geo}: к разметке {len(flies)} мух, пачка {MARK_BATCH}", flush=True)
    keys = set(THEME_KEYS)
    for st in range(0, len(flies), MARK_BATCH):
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп на {st}/{len(flies)}", flush=True)
            break
        chunk = flies[st : st + MARK_BATCH]
        idx = {str(j): lesson for j, (_fid, lesson) in enumerate(chunk)}
        res = call(
            json.dumps(idx, ensure_ascii=False),
            mark_sys(),
            consumer="mark",  # звено 3, канон §0.4
            salvage=("rows", "subtheme"),
        )
        rows = (res or {}).get("rows") or []
        by_i = {str(r.get("i")).strip(): r for r in rows if isinstance(r, dict)}
        for j, (fid, _lesson) in enumerate(chunk):
            r = by_i.get(str(j))
            if not r or not str(r.get("subtheme") or "").strip():
                continue
            tema = str(r.get("theme") or "").strip()
            done.append(
                {
                    "id": fid,
                    # ⛔ Тема вне закрытых 13 — не выдумка рта, а парковка: пусть лежит
                    # видимой кучей, а не растворяется по чужим темам.
                    "theme": tema if tema in keys else "prochee",
                    "subtheme": str(r.get("subtheme")).strip(),
                    # сколько мух стоит за представителем (1 — если схлопывания не было)
                    "n": za_rep.get(fid, 1),
                }
            )
        os.makedirs(f"{TESTS}/tags", exist_ok=True)
        with open(fn, "w", encoding="utf-8") as fh:  # чекпоинт: СТОП не съест сделанное
            json.dump(done, fh, ensure_ascii=False)
        print(f"  {min(st + MARK_BATCH, len(flies))}/{len(flies)}", flush=True)
    print(f"{geo}: размечено всего {len(done)} -> {fn}", flush=True)


def names_path(base=""):
    """Справочник имён: «имя подтемы → канон + латинский адрес».

    Канон §0.19 держит его в git как `pseo/site/canon.json` и правит РУКАМИ; прогон в
    контейнере в git писать не может, поэтому пишет сюда, а в git файл переносится руками.
    Место ОДНО: два места чтения — это два разных ответа на один вопрос.
    """
    return os.path.join(base, TESTS, "canon.json")


def load_names(base=""):
    return _load_json(names_path(base), {})


def summarize(geo):
    """Звено 4 ОБОБЩЕНИЕ: два прохода на тему (канон §0.19, PLAN.md).

    А — из меток темы с массами закрытый СПИСОК ИМЁН (это корзины страниц).
    Б — каждому совету имя ТОЛЬКО из списка; не подошло ни одно — совет уходит в остаток.

    ⛔ Почему два прохода, а не один. Один вызов «сведи названия» (22.08) слипал по широкому
    слову: в «документы на визу» ушли 33 разные подтемы — биометрия, ч/б печать, выписки,
    скрепки. Имя подтемы ставится в звене 3 ПО ОДНОЙ мухе, и свод по сходству имён честно
    слил всё, где есть частое слово. Здесь имя рождается из списка, а список ЗАКРЫТ.

    ⛔ Рту идут НОМЕРА, не id (сквозное правило PLAN.md): и метки, и советы, и имена.
    """
    fn = f"{TESTS}/tags/{geo}.json"
    tagged = _load_json(fn, None)
    if not tagged:
        print(f"{geo}: разметки нет", flush=True)
        return
    # ⛔ Текст берём ИЗ БАЗЫ по id: в тегах его больше нет (звено 3 не переводит и не
    # дублирует). Читать корпус бесплатно, а хранить второй экземпляр текста незачем.
    texts = dict(load_flies(geo))
    by_tema = {}
    for r in tagged:
        if r["id"] in texts:
            by_tema.setdefault(r.get("theme") or "prochee", []).append(r)

    directory = load_names()
    for tema, group in sorted(by_tema.items(), key=lambda kv: -len(kv[1])):
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп на теме {tema}", flush=True)
            break

        # ── ПРОХОД А: закрытый список имён темы ────────────────────────────────────
        masses = {}
        for r in group:
            masses[r["subtheme"]] = masses.get(r["subtheme"], 0) + 1
        labels = sorted(masses.items(), key=lambda kv: -kv[1])
        idx = {str(j): f"{m} ({n})" for j, (m, n) in enumerate(labels)}
        res = call(
            json.dumps(idx, ensure_ascii=False),
            names_sys(),
            consumer="canon",
            salvage=("names", "names"),
        )
        names = []
        for raw in (res or {}).get("names") or []:
            nm = str(raw or "").strip()
            if nm and not bad_name(nm) and nm not in names:
                names.append(nm)
        names = names[:NAMES_MAX]
        if not names:
            print(f"  тема {tema}: список имён не получен, пропуск", flush=True)
            continue
        for nm in names:
            directory.setdefault(nm, {"slug": slug(nm)})

        # ── ПРОХОД Б: присваивание из закрытого списка ─────────────────────────────
        # ⛔ Нумерация имён С ЕДИНИЦЫ: «0» занят под «ни одно не подошло» (так в
        # исходном промпте). С нуля ноль значил бы и первое имя, и отказ.
        names_map = {str(k): nm for k, nm in enumerate(names, start=1)}
        vzyato = 0
        for st in range(0, len(group), ASSIGN_BATCH):
            if os.path.exists("RUNNER_STOP"):
                print(f"  стоп на теме {tema}, пачка {st}", flush=True)
                break
            chunk = group[st : st + ASSIGN_BATCH]
            user = json.dumps(
                {
                    "names": names_map,
                    "advices": {str(j): texts[r["id"]] for j, r in enumerate(chunk)},
                },
                ensure_ascii=False,
            )
            res = call(user, assign_sys(), consumer="assign", salvage=("map", "map"))
            mp = (res or {}).get("map") or {}
            for j, r in enumerate(chunk):
                nom = str(mp.get(str(j), "0")).strip()
                # ⛔ Номер вне списка — не «новое имя», а промах: совет остаётся без имени
                # и уйдёт в остаток. Список закрыт, придумать шестнадцатое нельзя.
                if nom in names_map:
                    r["name"] = names_map[nom]
                    vzyato += 1
            _save_json(fn, tagged)  # чекпоинт: СТОП не съедает уплаченное
            _save_json(names_path(), directory)
        print(
            f"  тема {tema}: советов {len(group)}, меток {len(labels)} -> имён "
            f"{len(names)}, разложено {vzyato}",
            flush=True,
        )
    print(
        f"обобщение {geo}: справочник {len(directory)} имён -> {names_path()}",
        flush=True,
    )


def page_name(r, directory=None):
    """Имя страницы для совета. ОДНО место правила.

    Имя ставит проход Б звена 4 и кладёт его совету полем `kanon`. Нет поля — значит рот
    ответил «0» (ни одно имя из закрытого списка не подошло) или пачка не доехала: совет
    идёт в ОСТАТОК темы, а не заводит собственную корзину из своей подтемы. Иначе корзины
    начнут плодиться по одной на совет — ровно то, из-за чего звено 4 переписано.
    """
    return r.get("name") or ""


def split(items, limit):
    """Пачка больше `limit` → куски по `limit` подряд, последний неполный.

    38 при пороге 15 → 15/15/8. Ровнять части (13/13/12) юзер не просил, и выдумывать это
    не надо: правило простое — на странице не больше порога.
    """
    return [items[i : i + limit] for i in range(0, len(items), limit)] or [items]


def branch(title, base, items, theme, extra=None):
    """Пачка → страницы ОДНОЙ ВЕТКИ. Единственное место нарезки: и имена, и остаток.

    Ветвление одной темы (PLAN.md, слова юзера 27.08): тема — это список ВЕТОК, а ветка —
    одно имя. Не влезла в `PAGE_MAX` — продолжается страницами 1, 2, 3…; **последняя часть
    сколько осталось, хоть один абзац** (16 = 15 + 1, «не будем мы бегать и прибираться»).

    ⛔ Хвост НЕ сбрасывается в остаток: у этих советов имя УЖЕ есть, и сброс терял бы
    принадлежность. `PAGE_MIN` решает другое — заводить ли ветку с нуля.

    ⛔ Номер части живёт ПОЛЕМ (`part`), а не в имени: имя переводится в звене 6, и «(2)»
    внутри заголовка размножилось бы по четырнадцати языкам.
    """
    parts = split(items, PAGE_MAX)
    return [
        {
            "title": title,
            # ⛔ ТОЛЬКО ключ темы. Человеческое имя тут лежало русским, и сборщик переводил
            # его обратно в ключ — круг «ключ → русское → ключ». Имя темы на языке страницы
            # ставит сборщик: русское из справочника тракта, прочие из `themes.json`.
            "theme": theme,
            "items": part,
            "slug": base if nom == 1 else f"{base}-{nom}",
            # ⛔ Сборщик группирует части по ЭТОМУ полю, а не по имени и не разбором
            # суффикса: имя на 14 языках разное, суффикс — следствие, а не признак.
            "branch": base,
            "part": nom,
            "parts": len(parts),
            **(extra or {}),
        }
        for nom, part in enumerate(parts, start=1)
    ]


def build_corpus(geo):
    """Звено 5 СБОРКА: код, ключей НЕ тратит.

    Группируем советы по канону; подтема от PAGE_MIN пунктов — страница, остальное в
    остаток своей темы. Правила вывода проверяет КОД (канон, звено 4): каждый id ровно в
    одном месте, на странице от 4 до 15 пунктов. Нарушение печатается и правится рукой
    в справочнике — не перепрогоном рта.
    """
    tagged = _load_json(f"{TESTS}/tags/{geo}.json", None)
    if not tagged:
        print(f"{geo}: разметки нет", flush=True)
        return
    texts = dict(load_flies(geo))
    flies = [r for r in tagged if r["id"] in texts]
    directory = load_names()
    kratko = _load_json(f"{TESTS}/kratko/{geo}.json", {})

    po_kanonu = {}
    for r in flies:
        po_kanonu.setdefault((r.get("theme") or "prochee", page_name(r)), []).append(r)

    views, leftovers = [], {}
    for (theme, name), group in sorted(po_kanonu.items(), key=lambda kv: -len(kv[1])):
        items = [
            {"id": r["id"], "text": texts[r["id"]], "n": r.get("n", 1)} for r in group
        ]
        if not (name and len(items) >= PAGE_MIN and not bad_name(name)):
            leftovers.setdefault(theme, []).extend(items)
            continue
        base = (directory.get(name) or {}).get("slug") or slug(name)
        parts = branch(
            name,
            base,
            items,
            theme,
            {"kratko": kratko[name]} if kratko.get(name) else None,
        )
        views.extend(parts)
        if len(parts) > 1:
            print(
                f"  «{name}»: {len(items)} пунктов -> {len(parts)} страниц "
                f"{[len(c['items']) for c in parts]}",
                flush=True,
            )

    # ── ОСТАТОК: набралось на ветку — делаем служебную, нет — оставляем на теме ────────
    # ⛔ Тем же вызовом, что и имена: отдельного правила для остатка НЕТ, иначе назавтра
    # они разойдутся. «Разное» — не исключение, а такая же ветка (PLAN.md, 27.08).
    for theme, items in list(leftovers.items()):
        if len(items) < PAGE_MIN:
            continue  # мелочь веткой не становится — остаётся списком на странице темы
        views.extend(branch(MISC_TITLE, MISC_SLUG, items, theme))
        leftovers.pop(theme)

    vse = [it["id"] for v in views for it in v["items"]]
    vse += [it["id"] for x in leftovers.values() for it in x]
    tolstye = [
        (v["title"], len(v["items"])) for v in views if len(v["items"]) > PAGE_MAX
    ]
    print(
        f"сборка {geo}: страниц {len(views)}, тем с остатком {len(leftovers)}, "
        f"пунктов {len(vse)} из {len(flies)} размеченных",
        flush=True,
    )
    if len(vse) != len(set(vse)):
        print(f"  ⛔ id в двух местах: {len(vse) - len(set(vse))}", flush=True)
    if len(vse) != len(flies):
        print(f"  ⛔ потеряно советов: {len(flies) - len(vse)}", flush=True)
    if tolstye:
        print(
            f"  ⛔ СЛОМАНО ДЕЛЕНИЕ: страниц толще {PAGE_MAX} — {len(tolstye)}: "
            f"{tolstye[:5]}",
            flush=True,
        )
    os.makedirs(f"{TESTS}/out_facet", exist_ok=True)
    out = {
        "geo": geo,
        "views_by_task": views,
        # ⛔ Остаток адресуется КЛЮЧОМ темы: сборщику не нужно узнавать тему по имени.
        "shelves": [{"theme": k, "items": v} for k, v in leftovers.items()],
        "prochee": [],
    }
    with open(f"{TESTS}/out_facet/{geo}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"  -> {TESTS}/out_facet/{geo}.json", flush=True)


def _load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "нужно: tract.py <гео> --collapse | --mark [сколько] | --summarize | --build"
        )
    _geo = sys.argv[1]
    if "--collapse" in sys.argv:
        collapse(_geo)
    elif "--mark" in sys.argv:
        _i = sys.argv.index("--mark")
        mark(_geo, sys.argv[_i + 1] if len(sys.argv) > _i + 1 else None)
    elif "--summarize" in sys.argv:
        summarize(_geo)
    elif "--build" in sys.argv:
        build_corpus(_geo)
    else:
        raise SystemExit("неизвестный шаг")
