"""ПОЛКИ ПРОХОДЯТ ЧЕРЕЗ ПЕРЕВОД.

Повод (2026-07-27): перевод нёс только `views_by_task`. Мухи хвоста в набор на перевод не
попадали вовсе, полки в выходной файл не писались — и все 392 полки существовали ТОЛЬКО
по-русски, во всех 13 языках их было ноль. Не отставание, а по построению.

Проверяем три вещи, каждая из которых ломается молча:
  1. мухи полок попадают в набор на перевод (иначе собирать полку не из чего);
  2. в выходном файле есть полки, и их адресный ключ ЛАТИНСКИЙ и общий для всех языков
     (переведённое имя в URL = разные адреса в каждом языке, hreflang связывает не то);
  3. `is_fresh` требует ключ `shelves` — иначе 13 уже лежащих языков будут скипнуты как
     «готовые» и полок не получат никогда, а прогон отрапортует успех.

Сети и ключей не требует: переводчик и БД подменены. Запуск:
  python test_lang_shelves.py
"""

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facet_lang as fl  # noqa: E402
import tail_taxonomy as tax  # noqa: E402

SHELF_RU = tax.SHELVES[0][1]  # реальное имя из таксономии, не выдуманное
SHELF_KEY = tax.SHELVES[0][0]


def ru_geo():
    """RU-файл гео: один страничный вид (4 мухи) + одна полка (3 мухи хвоста)."""

    def it(i):
        return {
            "id": i,
            "text": "ru-%s" % i,
            "sushnosti": [],
            "mesto": None,
            "uslovie": None,
        }

    def grp(i):  # дедуп-группа: реальные файлы их несут, и is_fresh на них смотрит
        return {"rep": i, "ids": [i], "n": 1}

    return {
        "geo": "xx",
        "views_by_task": [
            {
                "zadacha": "Обмен валюты",
                "subshelves": [
                    {"name": "ветка-один", "reps": ["v0", "v1"]},
                    {"name": "ветка-два", "reps": ["v2", "v3"]},
                ],
                "items": [it("v%d" % i) for i in range(4)],
                "groups": [grp("v%d" % i) for i in range(4)],
            }
        ],
        "shelves": [
            {
                "shelf": SHELF_RU,
                "subshelves": [
                    {"name": "полка-ветка-а", "reps": ["s0", "s1"]},
                    {"name": "полка-ветка-б", "reps": ["s2", "s3"]},
                ],
                "items": [dict(it("s%d" % i), type="лайфхак") for i in range(3)],
                "groups": [grp("s%d" % i) for i in range(3)],
            }
        ],
        "prochee": [],
    }


def ok(cond, what, got=""):
    print("%-56s %-26s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    os.makedirs(f"{tmp}/out_facet", exist_ok=True)
    json.dump(ru_geo(), open(f"{tmp}/out_facet/xx.json", "w", encoding="utf-8"))
    fl.HERE = tmp

    seen_ids = {}

    # Подмена платного слоя: БД отдаёт текст по id, переводчик — префикс. Ключей не жжём.
    class _Cur:
        def __init__(self, ids):
            self.rows = [(i, "text-" + i) for i in ids]

        def fetchall(self):
            return self.rows

    class _Con:
        def execute(self, q, ids):
            return _Cur(ids)

        def close(self):
            pass

    fl.sqlite3 = type("S", (), {"connect": staticmethod(lambda *a, **k: _Con())})()
    fl.translate_texts = lambda t, lang: (
        seen_ids.update(t) or {k: "EN " + v for k, v in t.items()},
        None,
    )
    # ⚠️ Метка обязана быть БЕЗ кириллицы: из неё строится URL, и код намеренно
    # выбрасывает непереведённые метки. Дублёр, оставляющий русский, тихо обнуляет
    # весь прогон — проверено собой.
    fl.translate_labels = lambda labels, lang: {
        x: "translated-%d" % i for i, x in enumerate(labels)
    }
    fl.add_kratko = lambda geo, lang: 0

    good = True
    assert fl.run("xx", "de") is True, "перевод не отработал"
    out = json.load(open(f"{tmp}/out_facet_de/xx.json", encoding="utf-8"))

    # 1. Мухи ПОЛКИ попали в набор на перевод — раньше их там не было вовсе.
    good &= ok(
        {"s0", "s1", "s2"} <= set(seen_ids),
        "1. мухи хвоста ушли на перевод",
        "переведено id: %d" % len(seen_ids),
    )

    # 2. Полки в файле, имя переведено, ключ адреса — латинский из таксономии.
    sh = out.get("shelves") or []
    good &= ok(
        len(sh) == 1 and len(sh[0]["items"]) == 3,
        "2. полка в выходном файле",
        "полок %d" % len(sh),
    )
    if sh:
        good &= ok(
            sh[0]["shelf"].startswith("translated-"),
            "   имя переведено",
            sh[0]["shelf"][:28],
        )
        good &= ok(
            sh[0]["key"] == SHELF_KEY,
            "   ключ адреса латинский и общий",
            repr(sh[0].get("key")),
        )
        good &= ok(
            sh[0]["items"][0]["text"].startswith("EN "), "   текст мухи переведён"
        )
        good &= ok(
            sh[0]["items"][0]["type"] == "лайфхак",
            "   тип НЕ переведён (pages.py мапит по нему)",
        )

    # 3. is_fresh: старый файл (без ключа shelves) обязан считаться НЕсвежим.
    old = {"views_by_task": [{"zadacha": "x", "groups": [], "items": []}]}
    p_old, p_new = f"{tmp}/old.json", f"{tmp}/out_facet_de/xx.json"
    json.dump(old, open(p_old, "w", encoding="utf-8"))
    good &= ok(
        not fl.is_fresh(p_old), "3. файл без полок = НЕ готов (иначе скип навсегда)"
    )
    good &= ok(fl.is_fresh(p_new), "   свежий файл с полками = готов")

    # 4. ⛔ ВТОРАЯ КОПИЯ ПРАВИЛА. Решение «переводить или скипнуть» принимает НЕ facet_lang,
    #    а lang_runner.done() ДО него. Пока у него была своя проверка, знавшая только про
    #    groups, 36 гео × 13 языков (468 файлов, самые крупные гео) остались без полок —
    #    переводчик для них не звался вовсе, а прогон отрапортовал успех. Проверка №3 была
    #    при этом зелёной: она проверяла то, до чего исполнение не доходило.
    import lang_runner as lr

    lr.HERE = tmp
    lr._fresh.clear()
    os.makedirs(f"{tmp}/out_facet_old", exist_ok=True)
    json.dump(old, open(f"{tmp}/out_facet_old/xx.json", "w", encoding="utf-8"))
    good &= ok(
        not lr.done("xx", "old"),
        "4. РАННЕР тоже не считает готовым файл без полок",
        "done=%r" % lr.done("xx", "old"),
    )
    good &= ok(lr.done("xx", "de"), "   и считает готовым файл с полками")

    # 5. ⭐ ВЕТВЛЕНИЕ ДОЕЗЖАЕТ ДО ЯЗЫКА. Перевод его не нёс ВООБЩЕ, а `pages.py` строит хаб
    #    с ветками именно по `subshelves` — значит языки собирались бы простынями при уже
    #    разрезанном русском. И `is_fresh` про ветви не знал: файл считался готовым НАВСЕГДА,
    #    и исправить это было бы нечем, кроме ручного сноса файлов.
    tv = out["views_by_task"][0]
    tsh = (out.get("shelves") or [{}])[0]
    good &= ok(
        len(tv.get("subshelves") or []) == 2,
        "5. ветвление вида доехало до перевода",
        "ветвей %d" % len(tv.get("subshelves") or []),
    )
    good &= ok(
        len(tsh.get("subshelves") or []) == 2,
        "   и ветвление полки тоже",
        "ветвей %d" % len(tsh.get("subshelves") or []),
    )
    good &= ok(
        all(not re.search("[а-яё]", x["name"], re.I) for x in tv["subshelves"]),
        "   имена ветвей переведены (иначе кириллический URL)",
        str([x["name"] for x in tv["subshelves"]]),
    )
    good &= ok(
        out.get("branches_carried") is True,
        "   признак формата branches_carried стоит",
    )

    print("\nVERDICT:", "OK — полки доезжают до переводов" if good else "FAIL")
    sys.exit(0 if good else 1)
