"""ДОЗРЕВАНИЕ ГЕО НЕ СТИРАЕТ НАЖИТОЕ.

Повод (2026-08-08). `facet.run()` собирает файл гео С НУЛЯ и пишет поверх. Значит ровно в
тот момент, когда гео ДОЗРЕВАЕТ (remaining==0) — то есть когда разметка удаётся, — из файла
исчезают дедуп-группы, короткие ответы, ветвление и адреса страниц.

Замер того же дня: 58 гео из 90 стоят в очереди разметки, и в этих 58 лежит ВЕСЬ корпус —
1889 страниц, 1889 коротких ответов (100%), 182 ветвления. Месяц это не срабатывало только
потому, что ни одно гео не довели до конца: при remaining>0 функция выходит ДО записи.
То есть машинерия рушила наработанное ровно на успехе, и заметить это можно было лишь
доведя разметку — чего мы и не делали.

Правильный образец лежал всё это время в 30 строках ниже: `run_assign_tail` читает файл и
мёржит ключи, а не пересобирает.

Проверяем ПРАВИЛО, а не факт:
  1. состав узла не изменился → перенеслось ВСЁ;
  2. состав изменился → перенёсся ТОЛЬКО адрес, а группы и короткий ответ СНЯТЫ (страница
     рендерится из групп: старые группы скрыли бы новые мухи — тихая потеря);
  3. адрес живёт при росте страницы (иначе проиндексированный URL станет 404);
  4. один старый адрес не достаётся двум новым узлам;
  5. новый узел не получает чужой адрес;
  6. первый прогон гео (файла нет) не падает;
  7. `run()` действительно зовёт перенос ДО записи — иначе всё выше проверяет мимо.

Сети, ключей и БД не требует. Запуск:  python test_facet_merge.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facet  # noqa: E402


def ok(cond, what, got=""):
    print("%-60s %-22s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def view(zadacha, ids, **extra):
    return {
        "zadacha": zadacha,
        "items": [{"id": i, "text": "t-" + i} for i in ids],
        **extra,
    }


if __name__ == "__main__":
    good = True
    tmp = tempfile.mkdtemp()
    p = f"{tmp}/xx.json"

    # СТАРЫЙ файл: три вида, у всех нажитое добро и адрес.
    old = {
        "geo": "xx",
        "views_by_task": [
            view(
                "деньги",
                ["a", "b", "c", "d"],
                groups=[{"rep": "a", "ids": ["a", "b"], "n": 2}],
                kratko="ответ про деньги",
                subshelves=[
                    {"name": "переводы", "reps": ["a", "b"], "key": "transfers"},
                    {"name": "карты", "reps": ["c", "d"], "key": "cards"},
                ],
                key="money",
            ),
            view(
                "жильё",
                ["e", "f", "g", "h"],
                groups=[{"rep": "e", "ids": ["e"], "n": 1}],
                kratko="ответ про жильё",
                branch_tried=True,
                key="housing",
            ),
            view("виза", ["i", "j"], groups=[], kratko="ответ про визу", key="visa"),
        ],
        "shelves": [],
    }
    json.dump(old, open(p, "w", encoding="utf-8"), ensure_ascii=False)

    # НОВЫЙ файл после пересборки: «деньги» без изменений; «жильё» доросло вдвое (новые
    # мухи k,l,m,n); «виза» распалась, вместо неё совсем другой вид.
    new = {
        "geo": "xx",
        "views_by_task": [
            view("деньги", ["a", "b", "c", "d"]),
            view("жильё и аренда", ["e", "f", "g", "h", "k", "l", "m", "n"]),
            view("работа", ["x", "y", "z", "w"]),
        ],
        "shelves": [],
    }
    facet.carry_forward(p, new, "xx")
    v_same, v_grown, v_new = new["views_by_task"]

    # 1. Состав тот же → перенеслось ВСЁ.
    good &= ok(
        v_same.get("kratko") == "ответ про деньги"
        and len(v_same.get("groups") or []) == 1
        and len(v_same.get("subshelves") or []) == 2
        and v_same.get("key") == "money",
        "1. состав не изменился → перенеслось всё",
        "key=%s" % v_same.get("key"),
    )
    good &= ok(
        [s.get("key") for s in v_same["subshelves"]] == ["transfers", "cards"],
        "   и адреса ветвей тоже",
    )

    # 2-3. Состав изменился → ТОЛЬКО адрес, остальное снято на пересчёт.
    good &= ok(
        v_grown.get("key") == "housing",
        "2. страница доросла вдвое → адрес СОХРАНЁН",
        "key=%s" % v_grown.get("key"),
    )
    good &= ok(
        "groups" not in v_grown and "kratko" not in v_grown,
        "3. а группы и короткий ответ СНЯТЫ (иначе новые мухи невидимы)",
        "полей осталось: %s" % sorted(set(v_grown) - {"zadacha", "items", "key"}),
    )
    good &= ok(
        "branch_tried" not in v_grown,
        "   и «ветвление пробовали» снято — содержимое другое",
    )

    # 5. Новый узел чужого адреса не получает.
    good &= ok(
        "key" not in v_new,
        "5. новый вид без адреса (получит при штамповке)",
        repr(v_new.get("key")),
    )

    # 4. Один адрес — одному узлу. Два новых вида, похожих на один старый.
    json.dump(
        {
            "geo": "yy",
            "views_by_task": [
                view("деньги", ["a", "b", "c", "d"], key="money", kratko="k")
            ],
            "shelves": [],
        },
        open(f"{tmp}/yy.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    split = {
        "geo": "yy",
        "views_by_task": [
            view("деньги наличные", ["a", "b", "c", "q"]),
            view("деньги переводы", ["a", "b", "c", "r"]),
        ],
        "shelves": [],
    }
    facet.carry_forward(f"{tmp}/yy.json", split, "yy")
    keys = [v.get("key") for v in split["views_by_task"]]
    good &= ok(
        keys.count("money") == 1,
        "4. один старый адрес достался ровно одному узлу",
        str(keys),
    )

    # 6. Первый прогон гео: файла нет — не падаем и ничего не портим.
    virgin = {"geo": "zz", "views_by_task": [view("новое", ["p"])], "shelves": []}
    try:
        facet.carry_forward(f"{tmp}/нет-такого.json", virgin, "zz")
        good &= ok(
            "key" not in virgin["views_by_task"][0],
            "6. файла нет → тихо ничего не делаем",
        )
    except Exception as e:
        good &= ok(False, "6. файла нет → не падаем", "%s: %s" % (type(e).__name__, e))

    # 7. ⛔ ГЛАВНОЕ. Проверки 1-6 бессмысленны, если `run()` не зовёт перенос ДО записи.
    #    Ровно так уже было с `is_fresh`: проверка зелёная, а исполнение до неё не доходит.
    src = open(facet.__file__, encoding="utf-8").read()
    i_carry = src.find('carry_forward(f"out_facet/{geo}.json", page')
    i_write = src.find('_atomic_json(f"out_facet/{geo}.json", page)')
    good &= ok(
        i_carry != -1 and i_write != -1 and i_carry < i_write,
        "7. run() зовёт перенос ДО записи файла",
        "перенос@%d < запись@%d" % (i_carry, i_write),
    )

    print("\nVERDICT:", "OK — дозревание не стирает нажитое" if good else "FAIL")
    sys.exit(0 if good else 1)
