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

import dedup  # noqa: E402
import tail_taxonomy as tax  # noqa: E402
from facet import load_flies  # noqa: E402
from keybroker import call  # noqa: E402

# ⛔ ЕДИНСТВЕННОЕ место, где решается «куда пишем». Боевые каталоги рядом (`tags/`,
# `out_facet/`) — их новый тракт не касается.
TESTS = "tests"

# Порог схлопывания почти-копий. Замер 20.08 на боевом корпусе: 0.86 прячет за счётчиком
# до четверти содержимого (в одной группе из 44 советов Черногории слиплись налог,
# регистрация и ответственность владельца жилья), 0.93 берёт 3,5% — настоящие почти-копии.
# Судить порог по протоколу схлопывания (`tests/dedup/<гео>.txt`), а не по счётчику.
SGUSTOK_THR = 0.93

MARK_BATCH = 25  # мух в запрос разметки: 25 переводов ≈ 10К символов ответа
SVOD_BATCH = 90  # мух в запрос списков: та же пачка, что у прочих ртов тракта


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


SVOD_SYS = (
    'Ниже мухи ОДНОЙ темы гида по стране: {"<id>": "<текст мухи>", ...}.\n'
    "Разложи их по СПИСКАМ: список — один запрос человека, на который эти мухи отвечают.\n"
    "ПРАВИЛА:\n"
    "  - КАЖДАЯ муха ровно в ОДНОМ списке, ни одна не потеряна;\n"
    '  - имя списка 2–6 слов, как запрос человека: "аренда автомобиля", "оплата такси";\n'
    "  - ⛔ ЗАПРЕЩЕНЫ имена-рубрики («прочее», «общие советы») и имя самой темы.\n"
    'СТРОГО JSON: {"spiski": [{"imya": "…", "ids": ["…"]}]}'
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
    vv = dedup.load_vecs(ids)
    no_vec = [i for i in ids if i not in vv]
    groups = dedup.groups_all(ids, vv, SGUSTOK_THR)
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


def mark(geo, limit=None):
    """Шаг 3: разметить мух гео. Добирает только неразмеченных.

    Если схлопывание (шаг 2) прошло — размечаем ТОЛЬКО представителей, а сколько мух за
    каждым, несём числом `n`: платить рту за пересказ одного и того же незачем.
    """
    fn = f"{TESTS}/tags/{geo}.json"
    done = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else []
    have = {r["id"] for r in done}
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
    flies = [f for f in load_flies(geo) if f[0] not in have]
    if za_rep:
        flies = [f for f in flies if f[0] in za_rep]
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
            consumer="facet",
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


def svod(geo):
    """Шаг 4: мухи темы → списки → страницы и остаток. Пишет корпус для сборки."""
    fn = f"{TESTS}/tags/{geo}.json"
    if not os.path.exists(fn):
        print(f"{geo}: разметки нет", flush=True)
        return
    flies = [r for r in json.load(open(fn, encoding="utf-8")) if r.get("perevod")]
    by_tema = {}
    for r in flies:
        by_tema.setdefault(r.get("tema") or "prochee", []).append(r)
    names = {k: n for k, n, _d in tax.SHELVES}
    views, shelves = [], []
    for tema, group in sorted(by_tema.items(), key=lambda kv: -len(kv[1])):
        spiski = {}
        for st in range(0, len(group), SVOD_BATCH):
            if os.path.exists("RUNNER_STOP"):
                print(f"  стоп на теме {tema}", flush=True)
                break
            chunk = group[st : st + SVOD_BATCH]
            idx = {r["id"]: r["perevod"] for r in chunk}
            res = call(
                json.dumps(idx, ensure_ascii=False),
                SVOD_SYS,
                consumer="carve",
                salvage=("spiski", "ids"),
            )
            for sp in (res or {}).get("spiski") or []:
                imya = str(sp.get("imya") or "").strip()
                if not imya or tax.bad_label(imya):
                    continue
                spiski.setdefault(imya, []).extend(
                    i for i in (sp.get("ids") or []) if i in idx
                )
        by_id = {r["id"]: r for r in group}
        seen, ostatok = set(), []
        shelf_name = names.get(tema, tema)
        for imya, ids in spiski.items():
            ids = [i for i in dict.fromkeys(ids) if i not in seen and i in by_id]
            seen.update(ids)
            items = [
                {"id": i, "text": by_id[i]["perevod"], "n": by_id[i].get("n", 1)}
                for i in ids
            ]
            if len(items) >= tax.PAGE_MIN:
                views.append({"zadacha": imya, "shelf": shelf_name, "items": items})
            else:
                ostatok.extend(items)
        ostatok.extend(
            {"id": r["id"], "text": r["perevod"], "n": r.get("n", 1)}
            for r in group
            if r["id"] not in seen
        )
        if ostatok:
            shelves.append({"shelf": shelf_name, "items": ostatok})
        print(
            f"  тема {tema}: мух {len(group)} -> страниц "
            f"{sum(1 for v in views if v['shelf'] == shelf_name)}, остаток {len(ostatok)}",
            flush=True,
        )
    os.makedirs(f"{TESTS}/out_facet", exist_ok=True)
    out = {"geo": geo, "views_by_task": views, "shelves": shelves, "prochee": []}
    with open(f"{TESTS}/out_facet/{geo}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(
        f"свод {geo}: страниц {len(views)}, тем с остатком {len(shelves)} "
        f"-> {TESTS}/out_facet/{geo}.json",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("нужно: tract.py <гео> --mark [сколько] | --svod")
    _geo = sys.argv[1]
    if "--sgusti" in sys.argv:
        sgusti(_geo)
    elif "--mark" in sys.argv:
        _i = sys.argv.index("--mark")
        mark(_geo, sys.argv[_i + 1] if len(sys.argv) > _i + 1 else None)
    elif "--svod" in sys.argv:
        svod(_geo)
    else:
        raise SystemExit("неизвестный шаг")
