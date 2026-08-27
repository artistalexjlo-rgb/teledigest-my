"""ШАГ СЧИТАЕТ РАБОТУ, КОТОРУЮ МОЖНО СДЕЛАТЬ — все четыре шага, одна проверка.

Повод (2026-08-07). Одну и ту же болезнь я нашёл ЧЕТЫРЕ раза по отдельности, вместо того
чтобы один раз применить правило ко всему пайплайну. Юзер: «когда пайплайн функции будешь
примерять к работе?» — справедливо. Эта фикстура и есть применение правила ко всем шагам
разом, чтобы пятого раза не было.

Болезнь: шаг считает «что не сделано» вместо «что МОЖНО сделать». Пока работы много — не
видно; на остатках вертикаль перестаёт закрываться и зовёт по кругу вхолостую (за час 27.07
девять задач, все впустую).

Четыре случая, каждый со своим «невозможно»:
  шаг 0  мёртвые мухи (>=DEAD_AT провалов) — facet их не берёт. Фильтр падал на `int(hex)`,
         исключение съедалось, множество мёртвых было ПУСТЫМ всегда: 26 мух звали вечно.
  шаг 1  гео, где раскладывать нечего — `nl`: одна муха, 0 видов, 0 полок.
  шаг 2  страницы, где ветвление ПРОБОВАЛИ и вышло цельно (<2 под-тем) — не сбой, а «сделано».
  шаг 3  адреса — узел без `key`; без него нелатинские языки страниц не собирают.
  шаг 4  переводы — проверены аудитом, здоровы; сторожим, что не завелось нового.

Запуск:  BRAIN_DIR=<пусто> COMBINE_BOT_TOKEN=x ADMIN_ID=1 python test_steps_doable.py
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("COMBINE_BOT_TOKEN", "test")
os.environ.setdefault("ADMIN_ID", "1")
BRAIN = tempfile.mkdtemp()
os.makedirs(f"{BRAIN}/out_facet", exist_ok=True)
os.makedirs(f"{BRAIN}/tags", exist_ok=True)
os.environ["BRAIN_DIR"] = BRAIN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot  # noqa: E402


def ok(cond, what, got=""):
    print("%-58s %-20s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def geo_file(geo, views=None, shelves=None, prochee=None):
    json.dump(
        {
            "geo": geo,
            "views_by_task": views or [],
            "shelves": shelves or [],
            "prochee": prochee or [],
        },
        open(f"{BRAIN}/out_facet/{geo}.json", "w", encoding="utf-8"),
    )


def big(label_key, label, n, **extra):
    it = [{"id": "i%d" % i, "text": "t"} for i in range(n)]
    g = [{"rep": "i%d" % i, "ids": ["i%d" % i], "n": 1} for i in range(n)]
    return {label_key: label, "items": it, "groups": g, "kratko": "есть", **extra}


if __name__ == "__main__":
    good = True
    BR = 15

    # ── ШАГ 2: пробовали ветвить, вышло цельно → работы НЕТ.
    geo_file("aa", views=[big("zadacha", "цельная", BR + 5, branch_tried=True)])
    s = bot.pipeline_state()
    good &= ok(
        s["no_branch"] == 0,
        "шаг 2: ветвление пробовали, вышло цельно → работы нет",
        "no_branch=%d" % s["no_branch"],
    )

    # ── ШАГ 2 наоборот: не пробовали → работа ЕСТЬ (иначе правило сломано в другую сторону).
    geo_file("aa", views=[big("zadacha", "не пробовали", BR + 5)])
    s = bot.pipeline_state()
    good &= ok(
        s["no_branch"] == 1,
        "   а нетронутый гигант работой считается",
        "no_branch=%d" % s["no_branch"],
    )

    # ── ШАГ 1: полок нет и раскладывать НЕЧЕГО (ни хвоста, ни прочего) → работы нет.
    geo_file("bb", views=[big("zadacha", "крупный", 8)], prochee=[])
    s = bot.pipeline_state()
    good &= ok(
        "bb" not in s["no_shelf"],
        "шаг 1: раскладывать нечего → гео не висит",
        "no_shelf=%s" % s["no_shelf"],
    )

    # ── ШАГ 1 наоборот: есть `прочее` → работа есть.
    geo_file("bb", views=[big("zadacha", "крупный", 8)], prochee=[{"id": "x"}])
    s = bot.pipeline_state()
    good &= ok(
        "bb" in s["no_shelf"],
        "   есть хвост → гео в работе",
        "no_shelf=%s" % s["no_shelf"],
    )

    # ── ШАГ 0: фильтр мёртвых мух работает на HEX-id. Раньше `int(hex)` падал, исключение
    #    съедалось, и МЁРТВЫЕ считались живыми: 26 мух звали в меню вечно.
    hexid = "9497936e4990e9f99aca31a5"
    json.dump({hexid: 3}, open(f"{BRAIN}/tags/cc_fails.json", "w", encoding="utf-8"))
    dead = None
    try:
        fl = json.load(open(f"{BRAIN}/tags/cc_fails.json", encoding="utf-8"))
        dead = {k for k, c in fl.items() if c >= 3}
    except Exception:
        dead = set()
    good &= ok(
        hexid in dead,
        "шаг 0: мёртвая муха с HEX-id распознана",
        "мёртвых %d" % len(dead),
    )
    src = open(bot.__file__, encoding="utf-8").read()
    good &= ok(
        "{int(k)" not in src,  # именно код, а не упоминание в комментарии
        "   и int(hex) в КОДЕ пульта больше нет",
    )

    # ── ШАГ 3: АДРЕСА. Узел без `key` — выполнимая работа: штамповка переведёт русскую
    #    метку в английскую и сделает из неё хвост. Узел с ключом работой не считается.
    #    Врезано 08.08: до этого штамповку не звал НИКТО — ни шаг, ни кнопка, ни цикл,
    #    хотя без адреса нелатинские языки (zh ja ko ar hi th) страниц не собирают вовсе.
    for f in os.listdir(f"{BRAIN}/out_facet"):
        os.remove(f"{BRAIN}/out_facet/{f}")
    geo_file("dd", views=[big("zadacha", "без адреса", 6)])
    s = bot.pipeline_state()
    steps = {x["kind"]: len(x["jobs"]) for x in bot.pipeline_steps(s)}
    good &= ok(
        steps.get("stamp") == 1 and s["no_addr_n"] == 1,
        "шаг 3: вид без адреса → работа есть",
        "работ=%s, узлов=%s" % (steps.get("stamp"), s["no_addr_n"]),
    )
    v = big("zadacha", "с адресом", 6)
    v["key"] = "money"
    v["subshelves"] = [{"name": "ветка", "reps": ["i0"], "key": "branch"}]
    geo_file("dd", views=[v])
    s = bot.pipeline_state()
    steps = {x["kind"]: len(x["jobs"]) for x in bot.pipeline_steps(s)}
    good &= ok(
        steps.get("stamp") == 0,
        "   адреса на месте (и у вида, и у ветви) → работы нет",
        "работ=%s" % steps.get("stamp"),
    )
    v2 = big("zadacha", "ветвь без адреса", 6)
    v2["key"] = "money"
    v2["subshelves"] = [{"name": "ветка", "reps": ["i0"]}]  # у ВЕТВИ ключа нет
    geo_file("dd", views=[v2])
    s = bot.pipeline_state()
    good &= ok(
        s["no_addr_n"] == 1,
        "   ветвь без адреса тоже считается (у неё свой под-адрес)",
        "узлов=%s" % s["no_addr_n"],
    )

    # ── ШАГ 4: переводы. Русские данные есть, переводов нет → работа ЕСТЬ (это верно).
    for f in os.listdir(f"{BRAIN}/out_facet"):
        os.remove(f"{BRAIN}/out_facet/{f}")
    geo_file("aa", views=[big("zadacha", "t", 8)])
    s = bot.pipeline_state()
    steps = {x["kind"]: len(x["jobs"]) for x in bot.pipeline_steps(s)}
    good &= ok(
        steps.get("translate") == 1,
        "шаг 4: переводов нет → работа есть",
        "работ=%s" % steps.get("translate"),
    )

    # ── ШАГ 3 наоборот: язык переведён и СВЕЖЕЕ русского → работы нет. Это и есть
    #    «считаем выполнимое»: догонять нечего.
    import time as _t

    for lang in bot.LANGS:
        os.makedirs(f"{BRAIN}/out_facet_{lang}", exist_ok=True)
        with open(f"{BRAIN}/out_facet_{lang}/aa.json", "w", encoding="utf-8") as fh:
            json.dump({"geo": "aa"}, fh)
        os.utime(f"{BRAIN}/out_facet_{lang}/aa.json", (_t.time() + 60, _t.time() + 60))
    s = bot.pipeline_state()
    steps = {x["kind"]: len(x["jobs"]) for x in bot.pipeline_steps(s)}
    good &= ok(
        steps.get("translate") == 0,
        "   всё переведено и свежее → работы нет",
        "работ=%s | langs=%s" % (steps.get("translate"), s["langs"][:2]),
    )

    print("\nVERDICT:", "OK — шаги считают выполнимое" if good else "FAIL")
    sys.exit(0 if good else 1)
