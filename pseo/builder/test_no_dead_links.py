"""ССЫЛКА ЕСТЬ → СТРАНИЦА ЕСТЬ. Отсев один и до сиблингов.

Повод (2026-08-08). Проверка «похожих тем» строилась ДО того, как отпадали страницы:
причин выпасть у страницы две — нет адреса и метка не перевелась, — и вторая проверялась
НИЖЕ, внутри цикла записи. Итог: страница не писалась, а ссылка на неё в «похожие темы»
уже уехала. Замер по собранному сайту: 118 битых ссылок в 13 языках.

Форма дефекта та же, что весь день: одно решение в двух местах. Здесь оно опасно тем, что
битую ссылку не видно ни в логе, ни в тесте отдельной страницы — только обходом всего сайта.

Проверяем инвариант: КАЖДЫЙ адрес, на который ссылается собранная страница, принадлежит
собранной странице. Плюс что причина отсева печатается, а не проглатывается.

Сети, ключей и БД не требует. Запуск:  python test_no_dead_links.py
"""

import io
import json
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pages as pg  # noqa: E402


def ok(cond, what, got=""):
    print("%-58s %-26s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def view(zadacha, n, key=None):
    it = [
        {"id": "%s%d" % (zadacha[:2], i), "text": "Advice %d. Detail." % i}
        for i in range(n)
    ]
    v = {
        "zadacha": zadacha,
        "items": it,
        "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in it],
        "kratko": "short",
    }
    if key:
        v["key"] = key
    return v


if __name__ == "__main__":
    good = True
    tmp, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(f"{tmp}/out_facet_de", exist_ok=True)
    # Три вида: годный; с НЕПЕРЕВЕДЁННОЙ (кириллической) меткой; без адреса вообще.
    json.dump(
        {
            "geo": "xx",
            "views_by_task": [
                view("Banking", 6, key="banking"),
                view("Обмен валюты", 6, key="currency"),  # метка не перевелась
                # ⚠️ Чтобы адреса НЕ БЫЛО, метка обязана быть НЕЛАТИНСКОЙ: у латинской
                # («Taxes») фолбэк-слаг честно работает, и страница законно собирается.
                # Первая версия фикстуры этого не учла и требовала отсева там, где его нет.
                view("银行和钱", 6),  # адреса нет: слаг вычищается целиком
            ],
            "shelves": [],
        },
        open(f"{tmp}/out_facet_de/xx.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    pg.BUILT, pg.DATA = tmp, out
    buf = io.StringIO()
    with redirect_stdout(buf):
        pg.build_geo("xx", "de")
    log = buf.getvalue()

    written = set()
    refs = set()
    for fn in os.listdir(out):
        d = json.load(open(f"{out}/{fn}", encoding="utf-8"))
        if d.get("path"):
            written.add(d["path"])
        blob = json.dumps(d, ensure_ascii=False)
        refs |= set(re.findall(r'"url": "(/[a-z]{2}/[^"]*)"', blob))

    dead = sorted(r for r in refs if r not in written)
    good &= ok(not dead, "1. ни одной ссылки на несобранную страницу", str(dead[:3]))
    good &= ok(
        "/de/xx/banking/" in written,
        "2. годная страница собралась",
        str(sorted(written)[:3]),
    )
    good &= ok(
        "/de/xx/currency/" not in written and not any("currency" in r for r in refs),
        "3. вид с непереведённой меткой ни собран, ни упомянут",
    )
    good &= ok(
        not any(r.endswith("/tema/") for r in refs) and "/de/xx/tema/" not in written,
        "4. вид без адреса не упомянут",
    )
    good &= ok(
        "пропущено видов 2" in log
        and "метка не перевелась" in log
        and "без адреса" in log,
        "5. причина отсева НАПЕЧАТАНА, а не проглочена",
        re.sub(r"\s+", " ", log.strip())[:52],
    )

    # 6. Полочная страница объявляет общий хвост — иначе весь полочный контур идёт без
    #    hreflang (замер 08.08: 5616 страниц), хотя ключ полки латинский из таксономии и
    #    одинаков во всех языках.
    src = open(pg.__file__, encoding="utf-8").read()
    i_path = src.find('"path": f"/{lang}/{geo}/s/{sk}/"')
    good &= ok(
        i_path != -1 and '"shared_tail": True' in src[i_path : i_path + 400],
        "6. полочная страница несёт признак общего хвоста",
    )

    print("\nVERDICT:", "OK — битых ссылок нет" if good else "FAIL")
    sys.exit(0 if good else 1)
