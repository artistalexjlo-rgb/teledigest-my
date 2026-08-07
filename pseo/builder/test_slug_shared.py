"""АДРЕС ОДИН НА ВСЕ ЯЗЫКИ, ХВОСТ АНГЛИЙСКИЙ.

Правило юзера дословно: `/<язык>/<страна>/` + одинаковый английский хвост.
`/ru/br/money/` и `/zh/br/money/` — один и тот же хвост.

Повод (2026-08-08). Правило было записано в канон в ОСЛАБЛЕННОМ виде — «всегда латиницей» —
и ослабленный честно исполнялся: с 11.07 (d825245) слаг считался от ЛОКАЛИЗОВАННОЙ метки,
то есть у каждого языка выходил свой адрес. Транслит с русского под «латиницу» подходит.
Три следствия, все были живыми:
  1. свитчер языка падал в 404 → 14.07 его увели на хаб страны (ce103c9), причину не тронули;
  2. hreflang по сей день объявляет Google несуществующие адреса — проверено на живом сайте:
     `/ru/ar/bank-i-dengi/` шлёт на `/en/ar/bank-i-dengi/`, которого нет;
  3. на нелатинице (zh ja ko ar hi th) `slug()` вычищает ВСЕ символы и отдаёт "tema", а
     уникализации адресов в pages.py нет НИГДЕ — страницы молча перезаписывали бы друг
     друга, и в гео осталась бы ОДНА фактовая вместо ~20.

Сети и ключей не требует: переводчик подменён. Запуск:
  python test_slug_shared.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dedup as dd  # noqa: E402
import facet_lang as fl  # noqa: E402
import pages as pg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def ok(cond, what, got=""):
    print("%-62s %-24s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def ru_geo():
    """RU-гео: два страничных вида с НЕЛАТИНСКИМИ метками (как выглядит zh/ja/ar),
    у одного ветвление. Плюс уже нажитое добро — kratko и groups, которое штамповка
    адресов обязана сохранить."""

    def it(i):
        return {"id": "i%d" % i, "text": "t%d" % i}

    def grp(i):
        return {"rep": "i%d" % i, "ids": ["i%d" % i], "n": 1}

    return {
        "geo": "xx",
        "views_by_task": [
            {
                "zadacha": "банк и деньги",
                "items": [it(i) for i in range(8)],
                "groups": [grp(i) for i in range(8)],
                "kratko": "короткий ответ ОДИН",
                "subshelves": [
                    {"name": "переводы за рубеж", "reps": ["i0", "i1", "i2", "i3"]},
                    {"name": "карты и снятие", "reps": ["i4", "i5", "i6", "i7"]},
                ],
            },
            {
                "zadacha": "банк, и деньги!",  # ДРУГАЯ метка, тот же слаг → ловушка адресов
                "items": [it(i) for i in range(10, 15)],
                "groups": [grp(i) for i in range(10, 15)],
                "kratko": "короткий ответ ДВА",
            },
        ],
        "shelves": [],
        "prochee": [],
    }


if __name__ == "__main__":
    good = True

    # ── 1. ЛОВУШКА, из-за которой шесть языков нерабочие. Документируем поведение slug().
    good &= ok(
        pg.slug("银行和钱") == "tema" and pg.slug("البنوك والمال") == "tema",
        "1. slug() от нелатиницы вырождается в 'tema'",
        "%r / %r" % (pg.slug("银行和钱"), pg.slug("البنوك والمال")),
    )
    good &= ok(
        pg.slug("银行和钱") == pg.slug("保险和医疗"),
        "   и две РАЗНЫЕ темы дают ОДИН адрес → тихая перезапись",
    )

    # ── 2. addr() берёт ключ из данных → адрес латинский и одинаковый во всех языках.
    zh = {"zadacha": "银行和钱", "key": "money"}
    ar = {"zadacha": "البنوك والمال", "key": "money"}
    good &= ok(
        pg.addr(zh, "zadacha") == pg.addr(ar, "zadacha") == "money",
        "2. addr() с ключом: хвост общий и английский",
        pg.addr(zh, "zadacha"),
    )
    good &= ok(
        pg.addr({"zadacha": "银行和钱"}, "zadacha") == "tema",
        "   без ключа — прежний фолбэк (переходный, не адрес)",
    )

    # ── 3. ШТАМПОВКА АДРЕСОВ. Ключей не жжём: переводчик меток подменён.
    tmp = tempfile.mkdtemp()
    os.makedirs(f"{tmp}/out_facet", exist_ok=True)
    json.dump(
        ru_geo(),
        open(f"{tmp}/out_facet/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    fl.HERE = tmp
    EN = {
        "банк и деньги": "Banks and money",
        # НАМЕРЕННО схлопывается в ТОТ ЖЕ слаг, что первая: скобки и пробелы → один дефис.
        # «Banks & money» тут не годится — даёт banks-money, то есть совпадения нет, и
        # уникализация осталась бы непроверенной (моя первая версия фикстуры этим и болела).
        "банк, и деньги!": "Banks (and) money",
        "переводы за рубеж": "Transfers abroad",
        "карты и снятие": "Cards and withdrawal",
    }
    calls = []
    fl.translate_labels = lambda labels, lang: (
        calls.append((tuple(labels), lang)) or EN
    )

    n = fl.stamp_keys("xx")
    d = json.load(open(f"{tmp}/out_facet/xx.json", encoding="utf-8"))
    v0, v1 = d["views_by_task"]
    good &= ok(n == 4, "3. проштамповано узлов", "n=%d" % n)
    good &= ok(
        calls and calls[0][1] == "en",
        "   метки переводились именно в АНГЛИЙСКИЙ",
        calls[0][1] if calls else "-",
    )
    good &= ok(
        v0["key"] == "banks-and-money",
        "   ключ вида = слаг английской метки",
        repr(v0.get("key")),
    )
    good &= ok(
        v1["key"] == "banks-and-money-2",
        "   ⭐ совпавший хвост уникализирован (иначе тихая перезапись)",
        repr(v1.get("key")),
    )
    good &= ok(
        v0["key"] != v1["key"] and len({v0["key"], v1["key"]}) == 2,
        "   два вида — два РАЗНЫХ адреса",
    )
    good &= ok(
        [s["key"] for s in v0["subshelves"]]
        == ["transfers-abroad", "cards-and-withdrawal"],
        "   ветви тоже получили адреса",
        str([s.get("key") for s in v0["subshelves"]]),
    )

    # ── 4. МЁРЖ, а не пересборка: нажитое добро на месте. Это ровно та болезнь, из-за
    #    которой facet.run() стирает kratko — здесь она НЕ повторена.
    good &= ok(
        v0.get("kratko") == "короткий ответ ОДИН"
        and v1.get("kratko") == "короткий ответ ДВА",
        "4. короткие ответы уцелели (файл дописан, не пересобран)",
    )
    good &= ok(len(v0.get("groups") or []) == 8, "   дедуп-группы уцелели")

    # ── 5. Идемпотентность: второй заход не переводит и не тратит ничего.
    calls.clear()
    good &= ok(fl.stamp_keys("xx") == 0, "5. второй заход: 0 новых ключей")
    good &= ok(not calls, "   и НИ ОДНОГО обращения к модели")

    # ── 6. Перевод несёт адрес дальше: ветвь ниже порога выпадает, порог — ИЗ dedup.
    src = {
        "subshelves": [
            {
                "name": "переводы за рубеж",
                "key": "transfers-abroad",
                "reps": ["a", "b", "c", "d"],
            },
            {"name": "карты и снятие", "key": "cards", "reps": ["e", "f", "g", "h"]},
        ]
    }
    kept = {"a", "b", "c", "d", "e", "f"}  # у второй ветви осталось 2 < BRANCH_ITEM_MIN
    subs = fl.carry_subs(src, kept, {"переводы за рубеж": "Transfers abroad"})
    good &= ok(
        dd.BRANCH_ITEM_MIN == 4,
        "6. порог ветви живёт в dedup",
        "=%d" % dd.BRANCH_ITEM_MIN,
    )
    good &= ok(
        subs is None,
        "   ветвь тоньше порога выпала → осталась одна → ветвление снято",
        repr(subs),
    )
    subs2 = fl.carry_subs(
        src,
        set("abcdefgh"),
        {"переводы за рубеж": "Transfers abroad", "карты и снятие": "Cards"},
    )
    good &= ok(
        subs2 is not None
        and [s.get("key") for s in subs2] == ["transfers-abroad", "cards"],
        "   а полные ветви донесли АДРЕС в язык",
        str([s.get("key") for s in (subs2 or [])]),
    )

    # ── 7. Шлюз и адрес — из одного признака. Проверяем ИСХОДНИК: флаг shared_tail обязан
    #    выводиться из того же v.get("key"), иначе шаблон соврёт при верном адресе.
    src_pg = open(f"{HERE}/pages.py", encoding="utf-8").read()
    good &= ok(
        '"shared_tail": bool(v.get("key"))' in src_pg,
        "7. флаг shared_tail выведен из того же ключа",
    )
    good &= ok(
        'slug(v["zadacha"])' not in src_pg and "slug(g['tema'])" not in src_pg,
        "   и прямых slug() по локализованной метке в pages.py больше нет",
    )
    tpl = open(
        f"{os.path.dirname(HERE)}/templates/base.html.j2", encoding="utf-8"
    ).read()
    good &= ok(
        "{%- if page.shared_tail %}" in tpl,
        "   hreflang и свитчер закрыты этим же признаком",
    )

    print("\nVERDICT:", "OK — адрес один на все языки" if good else "FAIL")
    sys.exit(0 if good else 1)
