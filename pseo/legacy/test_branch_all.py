"""ВЕТВЛЕНИЕ — ПРАВИЛО СТРАНИЦЫ, а не свойство контура и не побочный эффект kratko.

Повод (2026-08-07). Замер по 4799 страницам при `BRANCH_MIN=15`:
    факты   3632 стр, нарушают 561, максимум 471 пункт на одном адресе
    полки    867 стр, нарушают  69, максимум 120
    вопросы   16 стр, нарушают   1
Механизм ветвления существовал и работал — но применялся ТОЛЬКО к полкам, и только внутри
`if kratko`. Поэтому: фактовые простыни не ветвились никогда, а полка, до которой проход
kratko не дошёл, простынёй и оставалась. Разное поведение контуров было не решением, а
недоделкой.

Проверяем три свойства:
  1. фактовый вид крупнее BRANCH_MIN ветвится (раньше — нет);
  2. полка ветвится БЕЗ kratko (раньше зависела от чужого шага);
  3. страница мельче порога не ветвится, и уже разветвлённую не пере-жжём.

Ни сети, ни ключей: `branch_page` подменён. Запуск:
  python test_branch_all.py
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
# `legacy` — отменённая схема: сторожу читать оттуда МОЖНО (правило живое, код мёртв),
# править нельзя. См. pseo/legacy/README.md
sys.path[:0] = [_HERE, os.path.join(os.path.dirname(_HERE), "legacy")]
import dedup  # noqa: E402


def page(n, label_key, label):
    """Страница на n пунктов: items + groups (по группе на пункт)."""
    items = [{"id": "i%d" % i, "text": "муха %d" % i} for i in range(n)]
    return {
        label_key: label,
        "items": items,
        "groups": [{"rep": "i%d" % i, "ids": ["i%d" % i], "n": 1} for i in range(n)],
    }


def run_on(views, shelves, kratko=False):
    """Прогнать dedup.run на своём гео, вернуть записанный файл."""
    d = tempfile.mkdtemp()
    dedup.OUT = d
    json.dump(
        {"geo": "xx", "views_by_task": views, "shelves": shelves},
        open(f"{d}/xx.json", "w", encoding="utf-8"),
    )
    dedup.group_view = lambda v, vv: v["groups"]  # дедуп не проверяем, он не про это
    dedup.load_vecs = lambda ids: {}
    called = []

    def fake_branch(pg, fails=None, kind="полка"):
        called.append((kind, pg.get("shelf") or pg.get("zadacha"), len(pg["groups"])))
        return [
            {"name": "ветка-1", "reps": [g["rep"] for g in pg["groups"][:4]]},
            {"name": "ветка-2", "reps": [g["rep"] for g in pg["groups"][4:8]]},
        ]

    dedup.branch_page = fake_branch
    dedup.run("xx", kratko=kratko)
    return json.load(open(f"{d}/xx.json", encoding="utf-8")), called


def ok(cond, what, got=""):
    print("%-58s %-24s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    good = True
    BR = dedup.BRANCH_MIN

    # 1. ⭐ ФАКТ крупнее порога — ветвится. Раньше не ветвился НИКОГДА.
    out, called = run_on([page(BR + 5, "zadacha", "большая тема")], [])
    good &= ok(
        bool(out["views_by_task"][0].get("subshelves")),
        "1. фактовый вид > BRANCH_MIN → разветвлён",
        "ветвлений: %d" % len(called),
    )
    good &= ok(
        called and called[0][0] == "тема",
        "   и помечен как тема, не как полка",
        str(called[0][:2]) if called else "-",
    )

    # 2. ⭐ ПОЛКА ветвится БЕЗ kratko. Раньше стояла под `if kratko`.
    out, called = run_on([], [page(BR + 5, "shelf", "большая полка")], kratko=False)
    good &= ok(
        bool(out["shelves"][0].get("subshelves")),
        "2. полка > BRANCH_MIN ветвится БЕЗ kratko",
        "ветвлений: %d" % len(called),
    )

    # 3. Мелкая страница не ветвится — порог работает в обе стороны.
    out, called = run_on([page(BR - 1, "zadacha", "мелкая")], [])
    good &= ok(
        not out["views_by_task"][0].get("subshelves") and not called,
        "3. страница <= BRANCH_MIN НЕ ветвится",
        "вызовов %d" % len(called),
    )

    # 4. Уже разветвлённую не трогаем — прогон идемпотентен, ключи не пере-жжём.
    v = page(BR + 5, "zadacha", "уже готова")
    v["subshelves"] = [{"name": "было", "reps": ["i0"]}]
    out, called = run_on([v], [])
    good &= ok(
        not called and out["views_by_task"][0]["subshelves"][0]["name"] == "было",
        "4. уже разветвлённую не пере-ветвляем",
        "вызовов %d" % len(called),
    )

    print(
        "\nVERDICT:", "OK — ветвление одинаково для всех контуров" if good else "FAIL"
    )
    sys.exit(0 if good else 1)
