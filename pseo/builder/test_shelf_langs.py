"""ПОЛОЧНЫЙ КОНТУР НЕ ОДНОЯЗЫЧЕН.

Повод (2026-08-08). Полки рендерились ТОЛЬКО по-русски из-за `if lang == "ru"` в pages.py.
Условие было честным 11.07: имя полки бралось из русской таксономии, чип типа был русским
словом. Обе причины сняли 27.07 (84b0a19 — перевод понёс локализованное имя и латинский
`key`), а условие осталось и прожило причину на 12 дней. Причём ТОТ ЖЕ коммит правил эту
же функцию: добавил `_sk()` с докстрингом про «адрес полки обязан совпадать во всех языках»
внутрь блока, который для не-ru не исполнялся.

Цена: ~1100 запросов на перевод текстов хвоста в 12 языков не дали ни одной страницы,
375 полок × 3 сборных языка = 1125 страниц не выкладывались.

Форма дефекта — заплатка живёт дальше причины, потому что ничем с причиной не связана.
Поэтому фикстура сторожит не «нет ли строки `lang == "ru"`», а САМО СВОЙСТВО: язык, у
которого есть полочный копирайт, обязан получать полочные страницы, и на них не должно
быть русских слов.

Сети, ключей и БД не требует. Запуск:  python test_shelf_langs.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pages as pg  # noqa: E402
import tail_taxonomy as tax  # noqa: E402

SHELF_KEYS = ("shelf_title", "shelf_desc", "shelf_intro", "shelf_list_label")


def ok(cond, what, got=""):
    print("%-60s %-26s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def lang_geo(shelf_name, key, chip_type):
    """Языковой файл гео: без фактовых видов, одна полка на 4 абзаца с типом.
    Имя полки УЖЕ переведено, `key` латинский — как их кладёт facet_lang."""
    items = [
        {
            "id": "s%d" % i,
            "text": "Advice %d. Details follow here." % i,
            "type": chip_type,
        }
        for i in range(4)
    ]
    return {
        "geo": "xx",
        "views_by_task": [],
        "shelves": [
            {
                "shelf": shelf_name,
                "key": key,
                "items": items,
                "groups": [
                    {"rep": it["id"], "ids": [it["id"]], "n": 1} for it in items
                ],
            }
        ],
    }


if __name__ == "__main__":
    good = True
    RU_TYPE = tax.TYPES[0][1]  # русское имя типа из таксономии, не выдуманное
    RU_TYPE_KEY = tax.TYPES[0][0]

    # ── 1. НАБОР КЛЮЧЕЙ СИММЕТРИЧЕН русскому — проверяем свойство, а не список.
    #    ⛔ Перечислять требуемые ключи вручную бесполезно: часть имён СОБИРАЕТСЯ склейкой
    #    (`blurb()` → C[key + "_blurb"]), и мой греп по литералам `C["..."]` их не увидел —
    #    `shelf_blurb` и `bridge_shelf_blurb` нашлись только когда упал этот тест. Сравнение
    #    наборов ловит любой будущий пропуск само, без правки фикстуры.
    #    Исключение — ru-only формы склонения `*_w`: в других языках их и не должно быть.
    ru_keys = {k for k in pg.COPY["ru"] if not k.endswith("_w")}
    for lang in pg.COPY:
        miss = sorted(ru_keys - set(pg.COPY[lang]))
        good &= ok(
            not miss,
            "1. COPY[%s]: набор ключей как у русского" % lang,
            ("нет: %s" % miss[:3]) if miss else "%d ключей" % len(ru_keys),
        )
    good &= ok(
        all(k in pg.COPY["en"] for k in SHELF_KEYS),
        "   в т.ч. полочный копирайт",
    )

    # ── 2. Чипы типов по языкам, и фолбэк АНГЛИЙСКИЙ. Русский фолбэк — видимый брак:
    #    на немецкой странице чип «лайфхак» никто не поймает автотестом, только глазами.
    for lang in pg.COPY:
        good &= ok(
            lang in pg.TYPE_SHORT,
            "2. чипы типов есть для %s" % lang,
            str(sorted(pg.TYPE_SHORT.get(lang, {}).values()))[:34],
        )
    # ⚠️ Код «zz» намеренно несуществующий. Сначала здесь стоял «de» как пример языка без
    # чипов — и проверка упала в тот же день, когда немецкий чипы получил. Фикстура не должна
    # кодировать ВРЕМЕННЫЙ факт («у такого-то языка чипов нет»), только СВОЙСТВО.
    good &= ok(
        pg.TYPE_SHORT.get("zz", pg.TYPE_SHORT["en"]) is pg.TYPE_SHORT["en"],
        "   у языка без чипов фолбэк на английский, не на русский",
    )

    # ── 3. E2E: собрать гео на АНГЛИЙСКОМ и убедиться, что полочная страница вышла.
    tmp = tempfile.mkdtemp()
    os.makedirs(f"{tmp}/out_facet_en", exist_ok=True)
    os.makedirs(f"{tmp}/out_facet", exist_ok=True)
    json.dump(
        lang_geo("Money and banks", "finance", RU_TYPE),
        open(f"{tmp}/out_facet_en/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    out = tempfile.mkdtemp()
    pg.BUILT, pg.DATA = tmp, out
    pg.build_geo("xx", "en")
    files = sorted(os.listdir(out))
    # ⚠️ под префикс подходит и ХАБ полок (en_xx_s_hub.json) — он законен, но это
    #    другая страница; сама полка это en_xx_s_<key>.json
    shelf_pages = [
        f for f in files if f.startswith("en_xx_s_") and not f.endswith("_hub.json")
    ]
    good &= ok(
        len(shelf_pages) == 1,
        "3. полочная страница собралась НА АНГЛИЙСКОМ",
        "файлов %d: %s" % (len(files), shelf_pages),
    )
    if shelf_pages:
        page = json.load(open(f"{out}/{shelf_pages[0]}", encoding="utf-8"))
        good &= ok(
            page["path"] == "/en/xx/s/finance/",
            "   адрес из латинского key файла",
            page["path"],
        )
        good &= ok(
            "Money and banks" in page["h1"] or "Money and banks" in page["title"],
            "   заголовок из ПЕРЕВЕДЁННОГО имени полки",
            page["title"][:36],
        )
        chips = [f.get("type") for f in page.get("faqs") or []]
        good &= ok(
            chips and all(c == pg.TYPE_SHORT["en"][RU_TYPE_KEY] for c in chips),
            "   ⭐ чип типа по-английски, а не «%s»" % pg.TYPE_SHORT["ru"][RU_TYPE_KEY],
            str(set(chips)),
        )
        good &= ok(
            page.get("list_label") == pg.COPY["en"]["shelf_list_label"],
            "   подпись списка из английского копирайта",
            repr(page.get("list_label")),
        )

    # ── 4. И русский не сломан тем же кодом.
    json.dump(
        lang_geo("Деньги и банки", "finance", RU_TYPE),
        open(f"{tmp}/out_facet/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    out_ru = tempfile.mkdtemp()
    pg.DATA = out_ru
    pg.build_geo("xx", "ru")
    ru_pages = [
        f
        for f in os.listdir(out_ru)
        if f.startswith("ru_xx_s_") and not f.endswith("_hub.json")
    ]
    good &= ok(len(ru_pages) == 1, "4. русская полка на месте", str(ru_pages))
    if ru_pages:
        pr = json.load(open(f"{out_ru}/{ru_pages[0]}", encoding="utf-8"))
        good &= ok(
            all(
                f.get("type") == pg.TYPE_SHORT["ru"][RU_TYPE_KEY]
                for f in pr.get("faqs") or []
            ),
            "   и её чип по-русски",
            str({f.get("type") for f in pr.get("faqs") or []}),
        )

    print("\nVERDICT:", "OK — полки выходят на всех сборных языках" if good else "FAIL")
    sys.exit(0 if good else 1)
