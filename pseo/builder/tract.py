"""ТРАКТ «ТЕМА И ПОДТЕМА» (канон §0.19) — шаги 3 и 4 взамен старой нарезки.

Шаг 3 РАЗМЕТКА: муха → перевод + тема (одна из 13) + подтема. Пачка 25.
Шаг 4 СПИСКИ:   мухи темы → списки с именами. Пачка 90; корпус в том виде, который уже
                понимает сборка: `views_by_task` и `shelves` (остаток).

⛔ ВСЁ ПИШЕТСЯ В `tests/` И ТОЛЬКО ТУДА (заказ юзера 20.08: прогоны новой схемы — в тестовую
папку). Боевые `tags/` и `out_facet/` не трогаются, пока схема не принята: 21.08 я направил
разметку в боевую папку, там смешались 146 старых записей с 42 новыми, а свод переписал боевой
корпус Чехии. Каталог задан ОДНОЙ константой ниже — второго места, где это решается, нет.

⛔ Числа списков у рта НЕ спрашиваем и вилок не задаём — их место в коде. Порог страницы один
на весь тракт (`tail_taxonomy.PAGE_MIN`).

Оба шага ДОБИРАЮТ работу: размеченные мухи пропускаются, корпус пишется чекпоинтами, поэтому
повторный запуск ключей не тратит, а СТОП между пачками не теряет сделанного.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⛔ Тракт НЕ импортирует отменённую схему (`facet`, `dedup`): чтение корпуса и
# вектора живут своими модулями, чтобы старый код не попадался под руку (рамка 24.08).
import slugs  # noqa: E402  адрес — существующей транслитерацией, второй копии нет
import tail_taxonomy as tax  # noqa: E402
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
SGUSTOK_THR = 0.93

# Верх страницы — канон §0.19 («на странице от 4 до 15 пунктов»). Низ живёт в
# `tail_taxonomy.PAGE_MIN` и здесь НЕ дублируется.
PAGE_MAX = 15

MARK_BATCH = 25  # мух в запрос разметки: 25 переводов ≈ 10К символов ответа


def mark_sys():
    """Промпт шага 3: три поля одним ответом, записи объектами."""
    spisok = "; ".join(f"{k} — {n}" for k, n, _d in tax.SHELVES)
    return (
        "Ты РАЗМЕТЧИК готового совета, НЕ автор. Совет не переписывай и не сокращай.\n"
        'На вход JSON {"0": "<совет>", ...}. Разметь КАЖДЫЙ.\n'
        'Верни СТРОГО JSON: {"rows": [{"i": "<индекс>", "perevod": "…", '
        '"tema": "…", "podtema": "…"}, ...]}\n'
        "  Каждая муха — ОТДЕЛЬНЫЙ объект: битая запись стоит одной мухи, а не всей пачки.\n"
        "  i       — ключ совета из входа.\n"
        "  perevod — дословный перевод на русский: все факты, числа, названия, условия как есть.\n"
        f"  tema    — РОВНО ОДИН ключ из списка: {spisok}.\n"
        "  podtema — 2–6 слов: зачем человек придёт за этим советом. Как запрос человека: "
        '"аренда автомобиля", "оплата такси", "поездка из аэропорта Тиват". '
        "⛔ ЗАПРЕЩЕНО повторять название темы и давать широкое слово-рубрику.\n"
        "Только JSON, без пояснений."
    )


# Имён в теме: границы из старого промпта (`DEALS_MAX = 15`, «дел от 5 до 15») — число
# не выдумано заново, оно бегало в проде. Больше 15 — это уже не список, а простыня.
NAMES_MIN, NAMES_MAX = 5, 15
RASKLAD_BATCH = 90  # советов в запрос прохода Б: тексты в один запрос не влезают


def spisok_sys():
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


def rasklad_sys():
    """Проход Б: присваивание из закрытого списка. Дословно `DEAL_ASSIGN_SYS`."""
    return (
        "Below is a CLOSED list of names (number: name), then advices (number: text). "
        "Assign EVERY advice to exactly ONE name FROM THE LIST — do not invent new ones. "
        'If none of them fits, put "0".\n'
        'STRICT JSON: {"map": {"<advice number>": "<name number or 0>"}}'
    )


def sgustok_path(geo):
    return f"{TESTS}/dedup/{geo}.json"


def sgusti(geo):
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
    groups = vectors.groups_all(ids, vv, SGUSTOK_THR)
    multi = [g for g in groups if len(g) > 1]
    swallowed = sum(len(g) - 1 for g in multi)
    recs = []
    for g in sorted(groups, key=lambda g: -len(g)):
        rep = max(g, key=lambda i: len(texts[i]))
        recs.append({"rep": rep, "ids": g})

    os.makedirs(f"{TESTS}/dedup", exist_ok=True)
    with open(sgustok_path(geo), "w", encoding="utf-8") as fh:
        json.dump(
            {"geo": geo, "thr": SGUSTOK_THR, "no_vec": len(no_vec), "groups": recs},
            fh,
            ensure_ascii=False,
        )
    proto = f"{TESTS}/dedup/{geo}.txt"
    with open(proto, "w", encoding="utf-8") as fh:
        print(f"# {geo}: схлопывание почти-копий, порог {SGUSTOK_THR}", file=fh)
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
    sp = sgustok_path(geo)
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
    keys = {k for k, _n, _d in tax.SHELVES}
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
            salvage=("rows", "podtema"),
        )
        rows = (res or {}).get("rows") or []
        by_i = {str(r.get("i")).strip(): r for r in rows if isinstance(r, dict)}
        for j, (fid, _lesson) in enumerate(chunk):
            r = by_i.get(str(j))
            if not r or not str(r.get("podtema") or "").strip():
                continue
            tema = str(r.get("tema") or "").strip()
            done.append(
                {
                    "id": fid,
                    "perevod": str(r.get("perevod") or "").strip(),
                    # ⛔ Тема вне закрытых 13 — не выдумка рта, а парковка: пусть лежит
                    # видимой кучей, а не растворяется по чужим темам.
                    "tema": tema if tema in keys else "prochee",
                    "podtema": str(r.get("podtema")).strip(),
                    # сколько мух стоит за представителем (1 — если схлопывания не было)
                    "n": za_rep.get(fid, 1),
                }
            )
        os.makedirs(f"{TESTS}/tags", exist_ok=True)
        with open(fn, "w", encoding="utf-8") as fh:  # чекпоинт: СТОП не съест сделанное
            json.dump(done, fh, ensure_ascii=False)
        print(f"  {min(st + MARK_BATCH, len(flies))}/{len(flies)}", flush=True)
    print(f"{geo}: размечено всего {len(done)} -> {fn}", flush=True)


def canon_path(base=""):
    """Справочник имён: «имя подтемы → канон + латинский адрес».

    Канон §0.19 держит его в git как `pseo/site/canon.json` и правит РУКАМИ; прогон в
    контейнере в git писать не может, поэтому пишет сюда, а в git файл переносится руками.
    Место ОДНО: два места чтения — это два разных ответа на один вопрос.
    """
    return os.path.join(base, TESTS, "canon.json")


def load_canon(base=""):
    return _load_json(canon_path(base), {})


def obobshi(geo):
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
    by_tema = {}
    for r in tagged:
        if r.get("perevod"):
            by_tema.setdefault(r.get("tema") or "prochee", []).append(r)

    imena = load_canon()
    for tema, group in sorted(by_tema.items(), key=lambda kv: -len(kv[1])):
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп на теме {tema}", flush=True)
            break

        # ── ПРОХОД А: закрытый список имён темы ────────────────────────────────────
        massy = {}
        for r in group:
            massy[r["podtema"]] = massy.get(r["podtema"], 0) + 1
        metki = sorted(massy.items(), key=lambda kv: -kv[1])
        idx = {str(j): f"{m} ({n})" for j, (m, n) in enumerate(metki)}
        res = call(
            json.dumps(idx, ensure_ascii=False),
            spisok_sys(),
            consumer="canon",
            salvage=("names", "names"),
        )
        names = []
        for raw in (res or {}).get("names") or []:
            nm = str(raw or "").strip()
            if nm and not tax.bad_label(nm) and nm not in names:
                names.append(nm)
        names = names[:NAMES_MAX]
        if not names:
            print(f"  тема {tema}: список имён не получен, пропуск", flush=True)
            continue
        for nm in names:
            imena.setdefault(nm, {"adres": slugs.slug(nm)})

        # ── ПРОХОД Б: присваивание из закрытого списка ─────────────────────────────
        # ⛔ Нумерация имён С ЕДИНИЦЫ: «0» занят под «ни одно не подошло» (так в
        # исходном промпте). С нуля ноль значил бы и первое имя, и отказ.
        spisok = {str(k): nm for k, nm in enumerate(names, start=1)}
        vzyato = 0
        for st in range(0, len(group), RASKLAD_BATCH):
            if os.path.exists("RUNNER_STOP"):
                print(f"  стоп на теме {tema}, пачка {st}", flush=True)
                break
            chunk = group[st : st + RASKLAD_BATCH]
            user = json.dumps(
                {
                    "names": spisok,
                    "advices": {str(j): r["perevod"] for j, r in enumerate(chunk)},
                },
                ensure_ascii=False,
            )
            res = call(user, rasklad_sys(), consumer="assign", salvage=("map", "map"))
            mp = (res or {}).get("map") or {}
            for j, r in enumerate(chunk):
                nom = str(mp.get(str(j), "0")).strip()
                # ⛔ Номер вне списка — не «новое имя», а промах: совет остаётся без имени
                # и уйдёт в остаток. Список закрыт, придумать шестнадцатое нельзя.
                if nom in spisok:
                    r["kanon"] = spisok[nom]
                    vzyato += 1
            _save_json(fn, tagged)  # чекпоинт: СТОП не съедает уплаченное
            _save_json(canon_path(), imena)
        print(
            f"  тема {tema}: советов {len(group)}, меток {len(metki)} -> имён "
            f"{len(names)}, разложено {vzyato}",
            flush=True,
        )
    print(
        f"обобщение {geo}: справочник {len(imena)} имён -> {canon_path()}",
        flush=True,
    )


def kanon_mukhi(r, imena=None):
    """Имя страницы для совета. ОДНО место правила.

    Имя ставит проход Б звена 4 и кладёт его совету полем `kanon`. Нет поля — значит рот
    ответил «0» (ни одно имя из закрытого списка не подошло) или пачка не доехала: совет
    идёт в ОСТАТОК темы, а не заводит собственную корзину из своей подтемы. Иначе корзины
    начнут плодиться по одной на совет — ровно то, из-за чего звено 4 переписано.
    """
    return r.get("kanon") or ""


def sborka(geo):
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
    flies = [r for r in tagged if r.get("perevod")]
    imena = load_canon()
    kratko = _load_json(f"{TESTS}/kratko/{geo}.json", {})
    names = {k: n for k, n, _d in tax.SHELVES}

    po_kanonu = {}
    for r in flies:
        po_kanonu.setdefault((r.get("tema") or "prochee", kanon_mukhi(r)), []).append(r)

    views, ostatki = [], {}
    for (tema, kan), group in sorted(po_kanonu.items(), key=lambda kv: -len(kv[1])):
        shelf = names.get(tema, tema)
        items = [
            {"id": r["id"], "text": r["perevod"], "n": r.get("n", 1)} for r in group
        ]
        if kan and len(items) >= tax.PAGE_MIN and not tax.bad_label(kan):
            views.append(
                {
                    "zadacha": kan,
                    "shelf": shelf,
                    "items": items,
                    "adres": (imena.get(kan) or {}).get("adres") or slugs.slug(kan),
                    **({"kratko": kratko[kan]} if kratko.get(kan) else {}),
                }
            )
        else:
            ostatki.setdefault(shelf, []).extend(items)

    vse = [it["id"] for v in views for it in v["items"]]
    vse += [it["id"] for x in ostatki.values() for it in x]
    tolstye = [
        (v["zadacha"], len(v["items"])) for v in views if len(v["items"]) > PAGE_MAX
    ]
    print(
        f"сборка {geo}: страниц {len(views)}, тем с остатком {len(ostatki)}, "
        f"пунктов {len(vse)} из {len(flies)} размеченных",
        flush=True,
    )
    if len(vse) != len(set(vse)):
        print(f"  ⛔ id в двух местах: {len(vse) - len(set(vse))}", flush=True)
    if len(vse) != len(flies):
        print(f"  ⛔ потеряно советов: {len(flies) - len(vse)}", flush=True)
    if tolstye:
        print(
            f"  ⚠️ толще {PAGE_MAX}: {len(tolstye)} страниц — {tolstye[:5]}; "
            "делить рукой в справочнике",
            flush=True,
        )
    os.makedirs(f"{TESTS}/out_facet", exist_ok=True)
    out = {
        "geo": geo,
        "views_by_task": views,
        "shelves": [{"shelf": k, "items": v} for k, v in ostatki.items()],
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
            "нужно: tract.py <гео> --sgusti | --mark [сколько] | --obobshi | --sborka"
        )
    _geo = sys.argv[1]
    if "--sgusti" in sys.argv:
        sgusti(_geo)
    elif "--mark" in sys.argv:
        _i = sys.argv.index("--mark")
        mark(_geo, sys.argv[_i + 1] if len(sys.argv) > _i + 1 else None)
    elif "--obobshi" in sys.argv:
        obobshi(_geo)
    elif "--sborka" in sys.argv:
        sborka(_geo)
    else:
        raise SystemExit("неизвестный шаг")
