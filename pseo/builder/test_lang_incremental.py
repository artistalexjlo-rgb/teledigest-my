"""ПОВТОРНЫЙ ПЕРЕВОД НИЧЕГО НЕ ПОКУПАЕТ ЗАНОВО.

Повод (2026-08-08). `facet_lang.run()` собирал языковой файл с нуля: переводил ВСЕ тексты,
хотя они уже лежали под теми же языко-независимыми id, и терял короткие ответы, потому что
писал словарь без них (а `dedup.kratko_lang` наполняет только пустые).

Следствие: цена смены формата равнялась цене КОРПУСА, а не размеру изменения. Замер того же
дня: полный пере-перевод 90 гео × 13 языков = ~3900 запросов текста + ~1500 меток +
23 506 коротких ответов ≈ 29 000, шесть дневных пулов, ~110 часов. Это уже случилось один
раз (полки 27.07), и мой же коммит 929afae поставил в очередь второй: `is_fresh` требует
`branches_carried`, которого нет ни в одном из 1170 лежащих файлов.

Проверяем ПРАВИЛА, а не числа:
  1. первый прогон покупает всё;
  2. повторный прогон того же материала покупает НОЛЬ — ни текстов, ни меток;
  3. короткий ответ переносится при НЕИЗМЕННОМ составе;
  4. состав изменился → короткий ответ НЕ переносится (выжимка по другому содержимому =
     неправда), а уже переведённые тексты всё равно переносятся;
  5. переписанный источник под тем же id → эта муха переводится заново (иначе перевод
     устарел бы навсегда и незаметно);
  6. неоднозначное соединение узлов → не переносим ничего (чужой короткий ответ хуже
     оплаты нового).

Сети, ключей и БД не требует: переводчик и БД подменены. Запуск:
  python test_lang_incremental.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facet_lang as fl  # noqa: E402

BOUGHT = {"texts": 0, "labels": 0}


def ok(cond, what, got=""):
    print("%-58s %-24s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def ru_geo(view_ids, extra_view=None):
    def it(i):
        return {
            "id": i,
            "text": "ru-" + i,
            "sushnosti": [],
            "mesto": None,
            "uslovie": None,
        }

    views = [
        {
            "zadacha": "Обмен валюты",
            "items": [it(i) for i in view_ids],
            "groups": [{"rep": i, "ids": [i], "n": 1} for i in view_ids],
        }
    ]
    if extra_view:
        views.append(
            {
                "zadacha": extra_view[0],
                "items": [it(i) for i in extra_view[1]],
                "groups": [{"rep": i, "ids": [i], "n": 1} for i in extra_view[1]],
            }
        )
    return {"geo": "xx", "views_by_task": views, "shelves": [], "prochee": []}


def stub(tmp, db_text):
    """Подменить платный слой. БД отдаёт db_text[id]; переводчик считает покупки."""

    class _Cur:
        def __init__(self, ids):
            self.rows = [(i, db_text[i]) for i in ids if i in db_text]

        def fetchall(self):
            return self.rows

    class _Con:
        def execute(self, q, ids):
            return _Cur(ids)

        def close(self):
            pass

    fl.HERE = tmp
    fl.sqlite3 = type("S", (), {"connect": staticmethod(lambda *a, **k: _Con())})()

    def _texts(t, lang):
        BOUGHT["texts"] += len(t)
        return {k: "DE " + v for k, v in t.items()}, None

    def _labels(labels, lang):
        BOUGHT["labels"] += len(labels)
        return {x: "label-%d" % i for i, x in enumerate(labels)}

    fl.translate_texts = _texts
    fl.translate_labels = _labels
    fl.add_kratko = lambda geo, lang: 0


def force_stale(path):
    """Снять признак формата — ровно то, что делает мой коммит 929afae со всеми 1170
    лежащими файлами: `is_fresh` их отвергает и заставляет пересобрать."""
    d = json.load(open(path, encoding="utf-8"))
    d.pop("branches_carried", None)
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return d


if __name__ == "__main__":
    good = True
    tmp = tempfile.mkdtemp()
    os.makedirs(f"{tmp}/out_facet", exist_ok=True)
    IDS = ["v0", "v1", "v2", "v3", "v4"]
    db = {i: "source of " + i for i in IDS + ["v5", "v6", "v7", "v8"]}
    stub(tmp, db)
    p_ru = f"{tmp}/out_facet/xx.json"
    p_de = f"{tmp}/out_facet_de/xx.json"
    json.dump(ru_geo(IDS), open(p_ru, "w", encoding="utf-8"), ensure_ascii=False)

    # ── 1. Первый прогон: платим за всё.
    assert fl.run("xx", "de") is True
    good &= ok(
        BOUGHT["texts"] == 5 and BOUGHT["labels"] == 1,
        "1. первый прогон купил всё",
        "текстов %d, меток %d" % (BOUGHT["texts"], BOUGHT["labels"]),
    )
    # короткий ответ синтезируется отдельным ртом — впишем как будто он уже сделан
    d = json.load(open(p_de, encoding="utf-8"))
    d["views_by_task"][0]["kratko"] = "короткий ответ"
    json.dump(d, open(p_de, "w", encoding="utf-8"), ensure_ascii=False)
    good &= ok(
        all("h" in it for it in d["views_by_task"][0]["items"]),
        "   и проштамповал отпечаток источника на мухах",
    )
    good &= ok(
        d["views_by_task"][0].get("src") == "Обмен валюты",
        "   и русскую метку для точного соединения",
        repr(d["views_by_task"][0].get("src")),
    )

    # ── 2-3. Повторный прогон того же материала: НОЛЬ покупок, kratko на месте.
    BOUGHT.update(texts=0, labels=0)
    force_stale(p_de)
    assert fl.run("xx", "de") is True
    d2 = json.load(open(p_de, encoding="utf-8"))
    good &= ok(
        BOUGHT["texts"] == 0 and BOUGHT["labels"] == 0,
        "2. повторный прогон купил НОЛЬ",
        "текстов %d, меток %d" % (BOUGHT["texts"], BOUGHT["labels"]),
    )
    good &= ok(
        d2["views_by_task"][0].get("kratko") == "короткий ответ",
        "3. короткий ответ перенесён (состав тот же)",
        repr(d2["views_by_task"][0].get("kratko"))[:30],
    )
    good &= ok(
        d2["views_by_task"][0]["items"][0]["text"].startswith("DE "),
        "   и текст остался переведённым",
    )

    # ── 4. Русский вид оброс мухами: тексты старых переносятся, kratko СНЯТ.
    BOUGHT.update(texts=0, labels=0)
    json.dump(
        ru_geo(IDS + ["v5", "v6"]),
        open(p_ru, "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    force_stale(p_de)
    assert fl.run("xx", "de") is True
    d3 = json.load(open(p_de, encoding="utf-8"))
    good &= ok(
        BOUGHT["texts"] == 2,
        "4. страница обросла → куплены ТОЛЬКО новые мухи",
        "куплено %d из 7" % BOUGHT["texts"],
    )
    good &= ok(
        "kratko" not in d3["views_by_task"][0],
        "   короткий ответ СНЯТ — содержимое другое",
        repr(d3["views_by_task"][0].get("kratko")),
    )

    # ── 5. Источник переписан под тем же id → эта муха переводится заново.
    BOUGHT.update(texts=0, labels=0)
    db["v2"] = "ПЕРЕПИСАННЫЙ источник v2"
    force_stale(p_de)
    assert fl.run("xx", "de") is True
    good &= ok(
        BOUGHT["texts"] == 1,
        "5. переписанный источник → переведена ровно одна муха",
        "куплено %d" % BOUGHT["texts"],
    )

    # ── 6. Неоднозначное соединение: два русских вида, и старый узел вложен в оба.
    #    Переносить нельзя — можно приклеить чужой короткий ответ.
    tmp2 = tempfile.mkdtemp()
    os.makedirs(f"{tmp2}/out_facet", exist_ok=True)
    os.makedirs(f"{tmp2}/out_facet_de", exist_ok=True)
    stub(tmp2, db)
    json.dump(
        ru_geo(IDS + ["v7"], extra_view=("Другая тема", IDS + ["v8"])),
        open(f"{tmp2}/out_facet/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    # лежащий перевод: ОДИН узел без `src`, состав вложен в оба русских вида
    json.dump(
        {
            "geo": "xx",
            "views_by_task": [
                {
                    "zadacha": "старая метка",
                    "items": [{"id": i, "text": "DE ru-" + i} for i in IDS],
                    "kratko": "ЧУЖОЙ короткий ответ",
                }
            ],
            "shelves": [],
        },
        open(f"{tmp2}/out_facet_de/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    BOUGHT.update(texts=0, labels=0)
    assert fl.run("xx", "de") is True
    d4 = json.load(open(f"{tmp2}/out_facet_de/xx.json", encoding="utf-8"))
    good &= ok(
        all("kratko" not in v for v in d4["views_by_task"]),
        "6. соединение неоднозначно → чужой короткий ответ НЕ приклеен",
        str([v.get("kratko") for v in d4["views_by_task"]]),
    )
    good &= ok(
        BOUGHT["texts"] == 2,
        "   но тексты по id всё равно перенесены (id однозначны)",
        "куплено %d из 7" % BOUGHT["texts"],
    )

    # ── 7. ⛔ АНГЛИЙСКИЙ = ИСХОДНИК, БЕЗ ВЫЗОВА МОДЕЛИ. Сторож на мою же ошибку 08.08:
    #    я счёл абзац с кириллическим символом «русским исходником» и отправил 182 таких
    #    на перевод. Это были АНГЛИЙСКИЕ тексты с оригинальными терминами в кавычках
    #    («'туризъм' purpose», «travel itinerary (маршрутный лист)», «"Госзакупки"»), а
    #    `translate_texts` бракует ответ с кириллицей — значит модель была ВЫНУЖДЕНА
    #    термины выбросить. Итог: 191 абзац обеднён, ~76 обращений впустую, восстановление
    #    из БД. Замер, который надо было сделать ДО: текстов преимущественно кириллических
    #    НОЛЬ, дефекта не существовало.
    #    Правило: для `en` текст совпадает с источником побайтово, что бы в нём ни стояло.
    tmp3 = tempfile.mkdtemp()
    os.makedirs(f"{tmp3}/out_facet", exist_ok=True)
    SRC = {
        "e0": "Standard visas ('туризъм' purpose) are limited to 30 days.",
        "e1": "A travel itinerary (маршрутный лист) is sufficient.",
        "e2": "Plain English advice with no quotes at all.",
        "e3": 'Check the "Госзакупки" portal for tenders.',
        "e4": "Ask at the registry (like ЗАГС in Russia).",
    }
    stub(tmp3, SRC)
    json.dump(
        ru_geo(list(SRC)),
        open(f"{tmp3}/out_facet/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    BOUGHT.update(texts=0, labels=0)
    assert fl.run("xx", "en") is True
    en = json.load(open(f"{tmp3}/out_facet_en/xx.json", encoding="utf-8"))
    got = {it["id"]: it["text"] for it in en["views_by_task"][0]["items"]}
    good &= ok(
        BOUGHT["texts"] == 0,
        "7. английский НЕ покупает текст, даже с кириллицей в цитатах",
        "куплено %d" % BOUGHT["texts"],
    )
    good &= ok(
        all(got.get(i) == t for i, t in SRC.items()),
        "   и текст совпадает с источником побайтово",
        "совпало %d из %d"
        % (sum(1 for i, t in SRC.items() if got.get(i) == t), len(SRC)),
    )
    good &= ok(
        "туризъм" in (got.get("e0") or "")
        and "маршрутный лист" in (got.get("e1") or ""),
        "   ⭐ оригинальные термины на месте, не выброшены",
    )

    print("\nVERDICT:", "OK — повторный перевод не покупает заново" if good else "FAIL")
    sys.exit(0 if good else 1)
