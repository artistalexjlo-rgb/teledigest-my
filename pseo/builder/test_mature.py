"""ЗРЕЛОСТЬ ГЕО ВЫЧИСЛЯЕТСЯ, а мёртвые мухи её не держат.

Повод (2026-08-07): гейт публикации в `ship` пускал гео только по `runner_stamps.json`, а
писал этот файл `pseo-runner` — снесённый 20.07 за то, что жил невидимкой и жёг ключи.
Файл замёрз на 36 гео из 90: всё собранное позже не поехало бы НИКОГДА, причём не из-за
сырости, а потому что штамповать стало некому. Плюс в sitemap попадали 188 адресов
задержанных гео — то есть карта звала Google на страницы, которых нет.

Проверяем правило, а не текст: зрелое = `load_flies` больше ничего не отдаёт.

⛔ Ключевой случай — мёртвая муха. Если её считать живой, одна непереваримая муха держит
гео недозревшим вечно. Ровно так 26 мух висели в меню пульта, и ВСЕ 26 были мёртвыми.

Ни сети, ни ключей, ни боевых данных: своя временная база и свои tags/. Запуск:
  python test_mature.py
"""

import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facet  # noqa: E402

LONG = "х" * (facet.MIN_LEN + 40)  # длиннее порога, иначе муха не считается вовсе


def setup(flies_by_geo, tagged=None, fails=None):
    """flies_by_geo: {гео: [id, ...]}. tagged/fails: {гео: [...]} / {гео: {id: счёт}}."""
    d = tempfile.mkdtemp()
    os.chdir(d)
    os.makedirs("out_facet")
    os.makedirs("tags")
    facet.DB = f"{d}/db.sqlite"
    c = sqlite3.connect(facet.DB)
    c.execute("CREATE TABLE extracted_patterns(id TEXT, country TEXT, ai_lesson TEXT)")
    for geo, ids in flies_by_geo.items():
        json.dump({"geo": geo}, open(f"out_facet/{geo}.json", "w", encoding="utf-8"))
        for i in ids:
            c.execute("INSERT INTO extracted_patterns VALUES(?,?,?)", (i, geo, LONG))
        if tagged and geo in tagged:
            json.dump(
                [{"id": i} for i in tagged[geo]],
                open(f"tags/{geo}.json", "w", encoding="utf-8"),
            )
        if fails and geo in fails:
            json.dump(fails[geo], open(f"tags/{geo}_fails.json", "w", encoding="utf-8"))
    c.commit()
    c.close()
    return d


def ok(cond, what, got=""):
    print("%-58s %-22s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    good = True

    # 1. Всё протеговано → зрелое. Ничего не протеговано → сырое.
    setup({"aa": ["1", "2"], "bb": ["3", "4"]}, tagged={"aa": ["1", "2"]})
    m = facet.mature_geos()
    good &= ok(m == {"aa": True, "bb": False}, "1. протеговано всё → зрелое", str(m))

    # 2. ⭐ ГЛАВНОЕ: осталась ОДНА муха, и она МЁРТВАЯ (>=DEAD_AT) → гео ЗРЕЛОЕ.
    #    Со старым штампом и с наивным «есть непротегованные» такое гео висело бы вечно.
    setup(
        {"cc": ["1", "2"]},
        tagged={"cc": ["1"]},
        fails={"cc": {"2": facet.DEAD_AT}},
    )
    m = facet.mature_geos()
    good &= ok(m == {"cc": True}, "2. осталась только МЁРТВАЯ муха → зрелое", str(m))

    # 3. Та же муха, но провалов на один меньше — ещё живая, гео сырое.
    setup(
        {"dd": ["1", "2"]},
        tagged={"dd": ["1"]},
        fails={"dd": {"2": facet.DEAD_AT - 1}},
    )
    m = facet.mature_geos()
    good &= ok(
        m == {"dd": False}, "3. провалов < DEAD_AT → муха живая, гео сырое", str(m)
    )

    # 4. DEAD_AT — константа МОДУЛЯ, а не локальная. На этом я споткнулся: ссылался на
    #    facet.DEAD_AT, которого не существовало, и порог дублировался тройками по коду.
    good &= ok(
        isinstance(getattr(facet, "DEAD_AT", None), int),
        "4. facet.DEAD_AT доступен извне",
        "DEAD_AT=%r" % getattr(facet, "DEAD_AT", None),
    )

    # 5. Гео без мух вообще (пустой корпус) — зрелое, а не вечно сырое.
    setup({"ee": []})
    m = facet.mature_geos()
    good &= ok(m == {"ee": True}, "5. гео без мух → зрелое", str(m))

    print(
        "\nVERDICT:", "OK — зрелость считается, мёртвые не держат" if good else "FAIL"
    )
    sys.exit(0 if good else 1)
