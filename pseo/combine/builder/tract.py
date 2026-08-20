"""ТРАКТ «ТЕМА И ПОДТЕМА» (канон §0.19) — шаги 3 и 4 взамен старой нарезки.

Шаг 3 РАЗМЕТКА: муха → перевод + тема (одна из 13) + подтема. Пачка 25, пишет `tags/<гео>.json`.
Шаг 4 СПИСКИ:   мухи темы → списки с именами. Пачка 90, пишет `out_facet/<гео>.json` в том
                виде, который уже понимает сборка: `views_by_task` и `shelves` (остаток).

⛔ Числа списков у рта НЕ спрашиваем и вилок не задаём — их место в коде. Порог страницы один
на весь тракт (`tail_taxonomy.PAGE_MIN`).

Оба шага ДОБИРАЮТ работу: размеченные мухи пропускаются, корпус пишется чекпоинтами, поэтому
повторный запуск ключей не тратит, а СТОП между пачками не теряет сделанного.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tail_taxonomy as tax  # noqa: E402
from facet import load_flies  # noqa: E402
from keybroker import call  # noqa: E402

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


def mark(geo, limit=None):
    """Шаг 3: разметить мух гео. Добирает только неразмеченных."""
    fn = f"tags/{geo}.json"
    done = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else []
    have = {r["id"] for r in done}
    flies = [f for f in load_flies(geo) if f[0] not in have]
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
                }
            )
        with open(fn, "w", encoding="utf-8") as fh:  # чекпоинт: СТОП не съест сделанное
            json.dump(done, fh, ensure_ascii=False)
        print(f"  {min(st + MARK_BATCH, len(flies))}/{len(flies)}", flush=True)
    print(f"{geo}: размечено всего {len(done)} -> {fn}", flush=True)


def svod(geo):
    """Шаг 4: мухи темы → списки → страницы и остаток. Пишет корпус для сборки."""
    fn = f"tags/{geo}.json"
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
            items = [{"id": i, "text": by_id[i]["perevod"]} for i in ids]
            if len(items) >= tax.PAGE_MIN:
                views.append({"zadacha": imya, "shelf": shelf_name, "items": items})
            else:
                ostatok.extend(items)
        ostatok.extend(
            {"id": r["id"], "text": r["perevod"]} for r in group if r["id"] not in seen
        )
        if ostatok:
            shelves.append({"shelf": shelf_name, "items": ostatok})
        print(
            f"  тема {tema}: мух {len(group)} -> страниц "
            f"{sum(1 for v in views if v['shelf'] == shelf_name)}, остаток {len(ostatok)}",
            flush=True,
        )
    os.makedirs("out_facet", exist_ok=True)
    out = {"geo": geo, "views_by_task": views, "shelves": shelves, "prochee": []}
    with open(f"out_facet/{geo}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(
        f"свод {geo}: страниц {len(views)}, тем с остатком {len(shelves)} "
        f"-> out_facet/{geo}.json",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("нужно: tract.py <гео> --mark [сколько] | --svod")
    _geo = sys.argv[1]
    if "--mark" in sys.argv:
        _i = sys.argv.index("--mark")
        mark(_geo, sys.argv[_i + 1] if len(sys.argv) > _i + 1 else None)
    elif "--svod" in sys.argv:
        svod(_geo)
    else:
        raise SystemExit("неизвестный шаг")
