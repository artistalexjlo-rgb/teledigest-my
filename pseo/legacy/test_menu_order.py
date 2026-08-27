"""ВЕРТИКАЛЬ МЕНЮ = ПОРЯДОК ИСПОЛНЕНИЯ. Один список, разъехаться нечему.

Повод (2026-07-27, вопрос юзера «при двух стрелках акцента нужно всё равно идти сверху
вниз?»): меню рисовало шаг 0 «Разметка» и ставило на него стрелку, а `start_cycle` собирал
цепочку ТОЛЬКО из assign/kratko/translate. Кнопка «ПОЛНЫЙ ЦИКЛ по порядку» молча пропускала
разметку и гнала kratko с переводами поверх непротегованных мух — против правила «facet
первым», записанного в самом же файле. Плюс блоки 🆕/🔧 висели НАД нумерованными шагами,
из-за чего вертикаль расходилась с нумерацией.

Проверяем инвариант, а не текст кнопок: что бы ни было в состоянии, порядок работ у меню
и у цикла ОДИН И ТОТ ЖЕ, и шаг 0 в цикл входит.

  BRAIN_DIR=/tmp COMBINE_BOT_TOKEN=x ADMIN_ID=1 python test_menu_order.py
"""

import os
import sys

os.environ.setdefault("COMBINE_BOT_TOKEN", "test")
os.environ.setdefault("ADMIN_ID", "1")
os.environ.setdefault("BRAIN_DIR", "/tmp/nonexistent-brain")

import bot  # noqa: E402


def state(**kw):
    """Состояние тракта. Пустое = всё готово; ключами наполняем нужный случай.

    ⛔ Форму состояния берём У САМОГО ПУЛЬТА (`pipeline_state` на пустом каталоге), а не
    переписываем словарь здесь. Своя копия формы уже сломала эту фикстуру: 08.08 в тракт
    добавился шаг «Адреса страниц», в состоянии появились `no_addr`/`no_addr_n`, а тут
    остался прежний набор — и тест упал с KeyError на ПРАВИЛЬНОМ коде. Копия формы = та же
    болезнь «одно правило в двух местах», только в сторожевой обвязке.
    """
    s = bot.pipeline_state()  # BRAIN_DIR указывает в несуществующий каталог → всё пусто
    s.update(kw)
    return s


def ok(cond, what, got=""):
    print("%-58s %-24s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    good = True

    full = state(
        pending_facet=[{"geo": "ph", "n": 58}, {"geo": "by", "n": 58}],
        failed=[{"geo": "vn", "n": 1, "flies": 37, "what": "перевозка"}],
        no_shelf=["br", "kz"],
        no_kratko=452,
        no_addr=["br", "vn", "me"],
        no_addr_n=57,
        langs=[("en", 10, 2), ("es", 5, 1)],
    )
    steps = bot.pipeline_steps(full)

    # 1. Вертикаль шагов — жёсткая и именно эта.
    good &= ok(
        [st["kind"] for st in steps]
        == ["facet", "assign", "kratko", "stamp", "translate"],
        "1. порядок шагов жёсткий",
        str([st["kind"] for st in steps]),
    )

    # 2. ⭐ ГЛАВНОЕ: цикл = ВСЯ вертикаль, включая шаг 0. Прежде шага 0 в нём не было.
    chain = [j for st in steps for j in st["jobs"]]
    good &= ok(
        [k for k, _ in chain[:3]] == ["facet", "facet", "facet"],
        "2. цикл НАЧИНАЕТСЯ с разметки, а не с assign",
        str([k for k, _ in chain[:4]]),
    )
    # ⛔ Проверяем ИНВАРИАНТ, а не перечень ротов. Прежняя форма сравнивала цепочку со
    #    списком `["facet"]*3 + ["assign"]*2 + [...]` — и ломалась на КАЖДОМ новом шаге
    #    тракта, хотя код был верен (08.08, шаг «Адреса страниц»). Инвариант же вечен:
    #    в цепочке ровно все работы шагов, и они идут группами в порядке вертикали.
    order = {st["kind"]: i for i, st in enumerate(steps)}
    idx = [order[k] for k, _ in chain]
    good &= ok(
        idx == sorted(idx) and len(chain) == sum(len(st["jobs"]) for st in steps),
        "   цепочка = все работы шагов, в порядке вертикали",
        "%d работ, порядок %s"
        % (len(chain), "не нарушен" if idx == sorted(idx) else "СБИТ"),
    )

    # 3. Брак идёт ПЕРВЫМ внутри шага 0: доделать сломанное прежде, чем брать новое.
    q = bot.facet_queue(full)
    good &= ok(
        q[0]["geo"] == "vn" and q[0]["broken"],
        "3. брак впереди новых мух",
        " → ".join(x["geo"] for x in q),
    )

    # 4. Гео, которое И сломано, И с новыми мухами, — ОДИН заход, не два.
    dup = state(
        failed=[{"geo": "vn", "n": 1, "flies": 37, "what": "x"}],
        pending_facet=[{"geo": "vn", "n": 12}, {"geo": "ph", "n": 5}],
    )
    q2 = [x["geo"] for x in bot.facet_queue(dup)]
    good &= ok(q2 == ["vn", "ph"], "4. дубль гео не удваивает работу", str(q2))

    # 4-БИС. ⛔ КНОПКА ШАГА РАСКРЫВАЕТСЯ В ЕГО ЖЕ РАБОТЫ — для ЛЮБОГО рота, а не для тех
    #    двух, кому в обработчике завели ветку руками. Именно на этом упала штамповка 08.08:
    #    шаг «Адреса страниц» был в реестре ртов, в метрике и в вертикали, а в обработчике
    #    кнопки — нет; он ушёл в общую ветку без гео, вышло `--stamp-keys ""` и падение на
    #    пути `out_facet/.json`. Проверяем СВОЙСТВО: у каждого шага с работой первая работа
    #    несёт непустой аргумент, если рот вообще по-гео (у kratko/translate он None по
    #    построению — им гео не нужно).
    PER_GEO = {"facet", "assign", "stamp"}  # рты, которым argv подставляет {geo}
    for st in bot.pipeline_steps(full):
        if not st["jobs"]:
            continue
        kind, arg = st["jobs"][0]
        good &= ok(
            kind == st["kind"]
            and (arg is not None if kind in PER_GEO else arg is None),
            "4-бис. шаг «%s»: первая работа с аргументом" % st["kind"],
            "%r → %r" % (kind, arg),
        )
    stamp_step = next(x for x in bot.pipeline_steps(full) if x["kind"] == "stamp")
    good &= ok(
        all(g for _, g in stamp_step["jobs"]),
        "   у штамповки НИ ОДНОЙ работы без гео",
        str(stamp_step["jobs"][:3]),
    )

    # 5. Шаг без работы не исчезает (позиции стабильны), но и в цикл не попадает.
    part = state(no_kratko=7)
    st2 = bot.pipeline_steps(part)
    good &= ok(
        len(st2) == 5 and [len(x["jobs"]) for x in st2] == [0, 0, 1, 0, 0],
        "5. пустые шаги на месте, но в цикл не идут",
        str([len(x["jobs"]) for x in st2]),
    )

    # 6. Всё готово → цикл пуст (кнопки «ВСЁ ПО ПОРЯДКУ» не будет).
    good &= ok(
        sum(len(x["jobs"]) for x in bot.pipeline_steps(state())) == 0,
        "6. нечего делать → пустой цикл",
    )

    # 7. Стрелка ровно одна и стоит на первом шаге С РАБОТОЙ.
    only_tr = state(langs=[("en", 1, 0)])
    first = next((x["kind"] for x in bot.pipeline_steps(only_tr) if x["jobs"]), None)
    good &= ok(first == "translate", "7. стрелка на первом шаге с работой", str(first))

    # 8. ⭐ ШАГ 2 = РАБОТА dedup, а не только kratko. `dedup.py` делает две вещи: короткие
    #    ответы И ветвление страниц-гигантов. Метрика смотрела лишь на первую, и шаг
    #    показывал ✅ при 95 нетронутых страницах — кнопка не срабатывала, а «ВСЁ ПО
    #    ПОРЯДКУ» шаг пропускало. Третий за сутки случай одной болезни: шаг считает не ту
    #    работу. Проверяем случай, где kratko готов, а ветвить есть что.
    only_branch = state(no_branch=7)
    st_k = [x for x in bot.pipeline_steps(only_branch) if x["kind"] == "kratko"][0]
    good &= ok(
        len(st_k["jobs"]) == 1 and "ветвлен" in st_k["label"],
        "8. шаг 2 видит ветвление как работу (не только kratko)",
        st_k["label"],
    )
    good &= ok(
        next((x["kind"] for x in bot.pipeline_steps(only_branch) if x["jobs"]), None)
        == "kratko",
        "   и стрелка встаёт на него",
    )

    print("\nVERDICT:", "OK — меню и цикл идут по одному списку" if good else "FAIL")
    sys.exit(0 if good else 1)
