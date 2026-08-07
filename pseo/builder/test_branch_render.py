"""ВЕТКИ РЕНДЕРЯТСЯ — и у фактов, и у полок, одной реализацией.

Повод (2026-08-07): `pages.py` читал `subshelves` ТОЛЬКО в полочной ветке. Данные для
ветвления считались, а фактовая страница собиралась той же простынёй — 561 такая, максимум
471 пункт на одном адресе. Копировать 60 строк в факты было нельзя: одно правило в двух
копиях дало 07.08 четыре промаха подряд, поэтому построение веток вынесено в `build_branches`
и зовётся дважды.

⚠️ Зовём НАСТОЯЩИЙ `build_geo`, а не `build_branches` напрямую. Сегодня трижды зелёная
фикстура проверяла то, до чего исполнение не доходит; проверка обязана идти боевым путём.

Проверка №3 — РЕГРЕССИЯ полок: я рефакторил работающий полочный путь, и он обязан вести
себя как раньше.

Ни сети, ни ключей. Запуск:  python test_branch_render.py
"""

import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pages  # noqa: E402

GEO = "br"  # реальный код: у pages есть его имена/падежи, иначе KeyError


def mkpage(label_key, label, n, subs=True):
    items = [
        {"id": "i%d" % i, "text": "Живой совет номер %d про важное." % i}
        for i in range(n)
    ]
    groups = [{"rep": "i%d" % i, "ids": ["i%d" % i], "n": 1} for i in range(n)]
    pg = {label_key: label, "items": items, "groups": groups}
    if subs:
        pg["subshelves"] = [
            {"name": "первая ветка", "reps": ["i0", "i1", "i2", "i3"]},
            {"name": "вторая ветка", "reps": ["i4", "i5", "i6", "i7"]},
        ]
    return pg


def build(views, shelves):
    built, data = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(f"{built}/out_facet")
    pages.BUILT, pages.DATA = built, data
    json.dump(
        {"geo": GEO, "views_by_task": views, "shelves": shelves},
        open(f"{built}/out_facet/{GEO}.json", "w", encoding="utf-8"),
    )
    pages.build_geo(GEO, "ru")
    out = {}
    for f in glob.glob(f"{data}/*.json"):
        out[os.path.basename(f)] = json.load(open(f, encoding="utf-8"))
    return out


def ok(cond, what, got=""):
    print("%-56s %-26s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    good = True

    # 1. ⭐ ФАКТ-гигант → хаб с плитками + под-страницы. Раньше был простынёй.
    out = build([mkpage("zadacha", "обмен валюты", 20)], [])
    hub = out.get("ru_br_obmen-valyuty.json")
    subs = [k for k in out if k.startswith("ru_br_obmen-valyuty_")]
    good &= ok(
        bool(hub and hub.get("template") == "index.html.j2" and hub.get("tiles")),
        "1. фактовый гигант стал ХАБОМ (не простынёй)",
        (
            "шаблон %s, плиток %d" % (hub.get("template"), len(hub.get("tiles") or []))
            if hub
            else "нет"
        ),
    )
    good &= ok(len(subs) == 2, "   под-страницы веток записаны", "%d шт" % len(subs))
    if subs:
        sp = out[sorted(subs)[0]]
        good &= ok(
            bool(sp["path"].startswith("/ru/br/obmen-valyuty/") and sp.get("faqs")),
            "   у ветки свой адрес под темой и есть содержимое",
            sp["path"],
        )
    good &= ok(
        bool(hub and hub.get("faqs")),
        "   остаток вне ветвей показан аккордеоном (не потерян)",
        "пунктов в остатке %d" % len(hub.get("faqs") or []) if hub else "-",
    )

    # 2. Тема без subshelves — как раньше: обычная страница-аккордеон, не хаб.
    out = build([mkpage("zadacha", "мелкая тема", 6, subs=False)], [])
    p = out.get("ru_br_melkaya-tema.json")
    good &= ok(
        bool(p and p.get("template") == "page.html.j2"),
        "2. тема без ветвления осталась обычной страницей",
        p.get("template") if p else "нет",
    )

    # 3. ⭐ РЕГРЕССИЯ ПОЛОК: рефакторинг не должен их изменить.
    out = build([], [mkpage("shelf", "Финансы и банковские услуги", 20)])
    shub = [k for k in out if "_s_" in k and k.count("_") == 3]
    ssub = [k for k in out if "_s_" in k and k.count("_") == 4]
    good &= ok(
        bool(shub) and bool(ssub),
        "3. полка-гигант по-прежнему хаб + под-страницы",
        "хабов %d, ветвей %d" % (len(shub), len(ssub)),
    )
    if ssub:
        good &= ok(
            "/s/" in out[sorted(ssub)[0]]["path"],
            "   адрес ветви полки остался под /s/",
            out[sorted(ssub)[0]]["path"],
        )

    print("\nVERDICT:", "OK — ветки рендерятся у обоих контуров" if good else "FAIL")
    sys.exit(0 if good else 1)
