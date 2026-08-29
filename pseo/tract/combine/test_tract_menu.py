"""Сторож меню пульта под тракт «тема и подтема» (канон §0.19).

⛔ Старые кнопки сняты вместе с их кодом: место страницы решалось трижды (тема мухе по тексту,
список дел ротом, тема разбору снова ротом). Теперь тема и подтема ставятся ОДИН раз в разметке,
всё остальное — код. Сторож держит две вещи: новые кнопки на месте и старые не вернулись.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# ⚠️ Основное дерево тракта (`HERE.parent` = pseo/tract), а не копия пульта: в копии
# (`combine/tract/`) нет `country_codes`, её кладёт образ. Без него `import facet` падает,
# и состояние возвращает ошибку вместо работы. 28.08: тракт больше не на уровень ниже —
# `HERE.parent` УЖЕ и есть `pseo/tract`, второй `/ "tract"` был бы `pseo/tract/tract`.
sys.path[:0] = [str(HERE), str(HERE.parent)]

os.environ.setdefault("COMBINE_BOT_TOKEN", "test")
os.environ.setdefault("ADMIN_ID", "1")

import bot  # noqa: E402

SNYATO = {
    "facet",
    "assign",
    "assignv",
    "flyshelf",
    "reshelf",
    "deals",
    "probe",
    "kratko",
}


def test_new_steps_are_in_the_menu():
    """Шаги тракта зовут НОВЫЙ модуль, а не старый facet."""
    for kind, flag in (
        ("collapse", "--collapse"),
        ("mark", "--mark"),
        ("summarize", "--summarize"),
        ("build_corpus", "--build"),
    ):
        assert kind in bot.MENU, kind
        argv = bot.MENU[kind][1]
        assert any(a.endswith("tract.py") for a in argv), argv
        assert flag in argv and "{geo}" in argv, argv


def test_old_steps_are_gone():
    """Старые кнопки не должны вернуться — вместе с ними вернулась бы старая нарезка."""
    lишние = SNYATO & set(bot.MENU)
    assert not lишние, f"старые кнопки на месте: {sorted(lишние)}"


def test_steps_go_in_tract_order():
    """Вертикаль меню = порядок тракта (§0.19): схлопывание → разметка → обобщение →
    корпус по справочнику → переводы → сборка сайта.

    Схлопывание стоит ПЕРВЫМ не для красоты: размечать оно даёт представителей, и после
    разметки его звать поздно — рту уже заплачено за каждую почти-копию.

    ⛔ Переводы ПЕРЕД сборкой (28.08, было наоборот): сборщик страниц (`site.py`) читает
    `out_facet_<язык>`, который кладут переводы — раньше сборка бежала первой и заставала
    только английский.
    """
    kinds = [
        st["kind"]
        for st in bot.pipeline_steps(
            {
                "mark": [],
                "mark_n": 0,
                "summarize": [],
                "build_corpus": [],
                "geos": 0,
                "views": 0,
            }
        )
    ]
    assert kinds == [
        "collapse",
        "mark",
        "summarize",
        "build_corpus",
        "translate",
        "build",
        "readiness",
    ], kinds


def test_job_tuples_are_kind_then_geo():
    """Каждая работа шага — `(kind шага, гео-или-None)`. `job.start(kind, geo)` читает
    именно в этом порядке.

    ⛔ 28.08: у «переводов» стояло `(гео, None)` — кнопка звала `job.start("gr", None)`,
    внутри `MENU["gr"]` — `KeyError`. Кнопка была первой некликнутой веткой нового шага,
    и до реального прогона на VPS никто её не нажимал — сторож на форму кортежа не было.
    """
    s = {
        "collapse": ["cz"],
        "mark": [{"geo": "cz", "n": 5}],
        "mark_n": 5,
        "summarize": ["cz"],
        "build_corpus": ["cz"],
        "to_translate": ["cz"],
        "geos": 1,
        "views": 3,
    }
    for st in bot.pipeline_steps(s):
        for job in st["jobs"]:
            assert job[0] == st["kind"], f"{st['kind']}: job {job} не начинается с kind"


def test_translate_is_done_when_every_language_is_fresh(tmp_path, monkeypatch):
    """Кнопка «Переводы» гасит галку, когда у гео есть корпус и ВСЕ целевые языки не
    старше него — не просто «корпус существует».

    ⛔ 28.08, юзер поймал: после успешного прогона (все 14 языков, 791 совет) кнопка
    осталась без ✅ — старая проверка (`to_translate`, тогда `sobrano`) считала работой
    сам факт наличия корпуса, а не отставание переводов от него.
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    out = tmp_path / "tests" / "out_facet_en"
    out.mkdir(parents=True)
    (out / "gr.json").write_text('{"views_by_task": []}', encoding="utf-8")

    st = bot.pipeline_state()
    assert "gr" in st["to_translate"], "без единого перевода шаг обязан быть НЕ готов"

    for lang in bot._SITE["languages"]:
        if lang == "en":
            continue
        d = tmp_path / "tests" / f"out_facet_{lang}"
        d.mkdir(parents=True)
        (d / "gr.json").write_text('{"views_by_task": []}', encoding="utf-8")

    st = bot.pipeline_state()
    assert "gr" not in st["to_translate"], "все языки свежие — шаг обязан быть ✅"


def test_collapse_is_stale_when_new_flies_are_not_covered(tmp_path, monkeypatch):
    """Схлопывание не гасит галку, если протокол дедупа не покрывает мух гео — не
    просто «файла нет вовсе».

    ⛔ 29.08: протокол мог остаться от первого прогона, а новые мухи (пришедшие позже)
    в нём не значились — кнопка молчала, хотя новые мухи никогда не проходили дедуп.
    """
    import sqlite3

    import corpus
    import vectors

    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    monkeypatch.setattr(bot, "GEO_FILE", str(tmp_path / "GEO"))
    dbpath = str(tmp_path / "messages_fts.db")
    conn = sqlite3.connect(dbpath)
    conn.execute(
        "CREATE TABLE extracted_patterns (country TEXT, id TEXT, ai_lesson TEXT)"
    )
    long_text = "x" * 200
    for i in range(3):
        conn.execute(
            "INSERT INTO extracted_patterns VALUES (?,?,?)", ("gr", f"g{i}", long_text)
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(corpus, "DB", dbpath)
    # ⛔ Без своей vec.db грузим дефолтный `/root/embed_ab/...` — невалидный путь на
    # Windows рушит проход по гео ДО проверки схлопывания, а не только счётчик векторов.
    vecdb = str(tmp_path / "local_vec.db")
    vc = sqlite3.connect(vecdb)
    vc.execute("CREATE TABLE vec (doc_id TEXT, v BLOB)")
    vc.commit()
    vc.close()
    monkeypatch.setattr(vectors, "VEC_DB", vecdb)
    bot.set_test_geo("gr")

    dedup_dir = tmp_path / "tests" / "dedup"
    dedup_dir.mkdir(parents=True)
    # протокол покрывает ТОЛЬКО g0, g1 — g2 (новая муха) в него не попал
    (dedup_dir / "gr.json").write_text(
        json.dumps({"groups": [{"rep": "g0", "ids": ["g0", "g1"]}]}), encoding="utf-8"
    )

    st = bot.pipeline_state()
    assert (
        "gr" in st["collapse"]
    ), "новая муха не покрыта протоколом — шаг обязан быть НЕ готов"

    (dedup_dir / "gr.json").write_text(
        json.dumps({"groups": [{"rep": "g0", "ids": ["g0", "g1", "g2"]}]}),
        encoding="utf-8",
    )
    st = bot.pipeline_state()
    assert "gr" not in st["collapse"], "все мухи покрыты — шаг обязан быть ✅"


def test_build_is_done_only_when_data_is_not_older_than_corpus(tmp_path, monkeypatch):
    """Кнопка «Сборка сайта» гасит галку, только когда данные (`site.py`) не старше
    корпуса — не просто «корпус существует» (29.08, та же болезнь, что у переводов).
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    corpus_dir = tmp_path / "tests" / "out_facet_en"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "gr.json").write_text('{"views_by_task": []}', encoding="utf-8")

    st = bot.pipeline_state()
    assert st["build_done"] is False, "данных ещё нет — шаг обязан быть НЕ готов"

    data_dir = tmp_path / "tests" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "en_gr.json").write_text("{}", encoding="utf-8")
    old = os.path.getmtime(corpus_dir / "gr.json") - 10
    os.utime(data_dir / "en_gr.json", (old, old))

    st = bot.pipeline_state()
    assert st["build_done"] is False, "данные старше корпуса — шаг обязан быть НЕ готов"

    fresh = os.path.getmtime(corpus_dir / "gr.json") + 10
    os.utime(data_dir / "en_gr.json", (fresh, fresh))
    st = bot.pipeline_state()
    assert st["build_done"] is True, "данные свежее корпуса — шаг обязан быть ✅"


def test_readiness_is_done_only_when_ready_json_is_fresh(tmp_path, monkeypatch):
    """Кнопка «Готовность» гасит галку, только когда `ready.json` не старше собранных
    данных — не просто «корпус существует».
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    monkeypatch.setattr(bot, "TRACT", str(tmp_path))
    data_dir = tmp_path / "tests" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "en_gr.json").write_text("{}", encoding="utf-8")

    st = bot.pipeline_state()
    assert (
        st["readiness_done"] is False
    ), "ready.json ещё нет — шаг обязан быть НЕ готов"

    ready_fn = tmp_path / "ready.json"
    ready_fn.write_text("{}", encoding="utf-8")
    old = os.path.getmtime(data_dir / "en_gr.json") - 10
    os.utime(ready_fn, (old, old))
    st = bot.pipeline_state()
    assert (
        st["readiness_done"] is False
    ), "ready.json старше данных — шаг обязан быть НЕ готов"

    fresh = os.path.getmtime(data_dir / "en_gr.json") + 10
    os.utime(ready_fn, (fresh, fresh))
    st = bot.pipeline_state()
    assert st["readiness_done"] is True, "ready.json свежее данных — шаг обязан быть ✅"


def test_switching_probe_does_not_borrow_another_geos_readiness(tmp_path, monkeypatch):
    """Свежевыбранная проба не должна показывать готовым то, что готово у ДРУГОГО гео.

    ⛔ 29.08, юзер поймал: переключил пробу с `gr` (полностью собранной) на `gb` — и
    «Сборка сайта»/«Готовность» сразу стояли ✅, хотя `gb` только начал разметку. Свежесть
    считалась по ВСЕМ гео сразу, а не по выбранной пробе, как остальные шаги.
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    monkeypatch.setattr(bot, "TRACT", str(tmp_path))
    monkeypatch.setattr(bot, "GEO_FILE", str(tmp_path / "GEO"))

    # gr — полностью собран и готов
    (tmp_path / "tests" / "out_facet_en").mkdir(parents=True)
    (tmp_path / "tests" / "out_facet_en" / "gr.json").write_text(
        '{"views_by_task": []}', encoding="utf-8"
    )
    (tmp_path / "tests" / "data").mkdir(parents=True)
    (tmp_path / "tests" / "data" / "en_gr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ready.json").write_text("{}", encoding="utf-8")

    bot.set_test_geo("gr")
    st = bot.pipeline_state()
    assert st["build_done"] is True, "gr собран — шаг обязан быть ✅"
    assert st["readiness_done"] is True, "gr проверен — шаг обязан быть ✅"

    # переключились на gb — у него корпуса и данных ещё нет вовсе
    bot.set_test_geo("gb")
    st = bot.pipeline_state()
    assert st["build_done"] is False, "у gb корпуса нет — шаг НЕ должен быть ✅"
    assert st["readiness_done"] is False, "у gb данных нет — шаг НЕ должен быть ✅"
    assert st["geos"] == 0, "проба gb — счёт не должен показывать корпус gr"
    assert st["views"] == 0, st["views"]


def test_work_is_counted_as_undone(tmp_path, monkeypatch):
    """Работа шага «списки» = гео, у которых разметка новее корпуса или корпуса нет.

    ⛔ Иначе шаг покажет ✅ на несобранном гео, и упавшее заново не встанет (канон §0.17).
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    (tmp_path / "tests" / "tags").mkdir(parents=True)
    (tmp_path / "tests" / "tags" / "cz.json").write_text(
        '[{"id": "a", "perevod": "текст", "theme": "transport", "subtheme": "аренда"}]',
        encoding="utf-8",
    )
    st = bot.pipeline_state()
    assert "cz" in st["summarize"], st["summarize"]


def test_menu_builds_on_a_live_state(monkeypatch):
    """⛔ Меню обязано собираться на РЕАЛЬНОМ состоянии, а не падать на отсутствующем ключе.

    Повод: после переделки тракта пульт написал «стартовое меню не отправилось: KeyError
    'geos'» — строки по гео брались из ключа, которого у новых шагов нет. Проверяем сборкой
    целиком, подменив только отправку в Telegram.
    """
    sent = []
    monkeypatch.setattr(bot, "tg", lambda method, **kw: sent.append((method, kw)) or {})
    monkeypatch.setattr(
        bot,
        "pipeline_state",
        lambda: {
            "mark": [{"geo": "cz", "n": 191}],
            "mark_n": 191,
            "summarize": ["cz"],
            "build_corpus": ["cz"],
            "geos": 1,
            "views": 12,
            "langs": [],
            "no_vec": 0,
        },
    )
    bot.send_menu(None)
    assert sent, "меню не отправлено"
    text = str(sent[-1][1])
    assert "cz" in text and "191" in text, text


def test_every_geo_step_gets_its_own_rows(monkeypatch):
    """У шага, чья работа разложена по гео, в меню есть строка НА ГЕО.

    Повод: схлопывание завелось строкой «94 гео» и без строк по гео — запустить его можно
    было только на всём корпусе разом, а проба идёт на одной стране.
    """
    sent = []
    monkeypatch.setattr(bot, "tg", lambda method, **kw: sent.append((method, kw)) or {})
    monkeypatch.setattr(
        bot,
        "pipeline_state",
        lambda: {
            "collapse": ["gr", "me"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "summarize": [],
            "build_corpus": [],
            "geos": 0,
            "views": 0,
            "langs": [],
            "no_vec": 0,
        },
    )
    bot.send_menu(None)
    text = str(sent[-1][1])
    assert "run:collapse:gr" in text, text
    assert "run:collapse:me" in text, text
    assert "run:mark:gr" in text, text


def test_menu_shows_only_the_trial_country(tmp_path, monkeypatch):
    """Пока идёт проба, меню считает ОДНУ страну.

    Повод: без этого пульт рисовал строку на каждое из 94 гео у каждого шага — полотно
    кнопок, где нужную надо искать, а «ВСЁ ПО ПОРЯДКУ» звало на весь корпус (188 шагов,
    ~23 тысячи запросов при потолке ~5 280 в день).
    """
    monkeypatch.setattr(bot, "GEO_FILE", str(tmp_path / "GEO"))
    bot.set_test_geo("gr")
    assert bot.test_geo() == "gr"

    sent = []
    monkeypatch.setattr(bot, "tg", lambda method, **kw: sent.append((method, kw)) or {})
    monkeypatch.setattr(
        bot,
        "pipeline_state",
        lambda: {
            "collapse": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "summarize": [],
            "build_corpus": [],
            "geos": 0,
            "views": 0,
            "langs": [],
            "no_vec": 0,
        },
    )
    bot.send_menu(None)
    text = str(sent[-1][1])
    assert "run:collapse:gr" in text and "run:mark:gr" in text, text
    for chuzhoy in ("br", "kr", "vn", "any"):
        assert f"run:collapse:{chuzhoy}" not in text, chuzhoy
    assert "проба: gr" in text, text

    bot.set_test_geo("-")  # снятие возвращает счёт по всему корпусу
    assert bot.test_geo() == ""


def test_trial_country_is_picked_by_button(tmp_path, monkeypatch):
    """Страна пробы меняется КНОПКОЙ: строка в меню + список стран под ней.

    Набирать `/geo gr` руками — значит помнить команду и код страны; пульт весь на кнопках,
    и этот выбор не исключение.
    """
    monkeypatch.setattr(bot, "GEO_FILE", str(tmp_path / "GEO"))
    bot.set_test_geo("gr")
    sent = []
    monkeypatch.setattr(bot, "tg", lambda method, **kw: sent.append((method, kw)) or {})
    monkeypatch.setattr(
        bot,
        "pipeline_state",
        lambda: {
            "all_geos": [{"geo": "br", "n": 3011}, {"geo": "gr", "n": 765}],
            "collapse": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "summarize": [],
            "build_corpus": [],
            "geos": 0,
            "views": 0,
            "langs": [],
            "no_vec": 0,
        },
    )
    bot.send_menu(None)
    assert "geo:pick" in str(sent[-1][1]), sent[-1][1]

    bot.send_geo_picker()
    picker = str(sent[-1][1])
    assert "geo:br" in picker and "geo:gr" in picker, picker
    assert "geo:-" in picker, "нет кнопки «весь корпус»"


def test_cycle_counts_the_current_tract(monkeypatch):
    """«ВСЁ ПО ПОРЯДКУ» считает расход по НЫНЕШНИМ шагам и не падает.

    Повод: 22.08 оценка тянула ключи умершей схемы (`no_shelf`, `no_kratko`, `no_branch`),
    и первое же нажатие уронило пульт с KeyError — вместе со всей обработкой кнопок.
    """
    said = []
    monkeypatch.setattr(bot, "say", lambda t: said.append(t))
    monkeypatch.setattr(
        bot,
        "pipeline_state",
        lambda: {
            "all_geos": [{"geo": "gr", "n": 765}],
            "collapse": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "summarize": [],
            "build_corpus": [],
            "geos": 0,
            "views": 0,
            "langs": [],
        },
    )

    class FakeJob:
        chain = []

        def start(self, kind, geo, _chain=False):
            self.started = (kind, geo)

    j = FakeJob()
    bot.start_cycle(j)
    assert j.started == ("collapse", "gr"), getattr(j, "started", None)
    text = " ".join(said)
    assert "весь тракт" in text, text
    # разметка 765/25=31 вызов, списки ≤(13+8) вызовов, по 4 запроса worst-case
    assert "~" in text and "запросов" in text, text


def test_cycle_goes_country_by_country_not_step_by_step():
    """«ВСЁ ПО ПОРЯДКУ» проходит ОДНУ страну целиком (все её шаги), потом следующую —
    а не шаг поперёк всех стран.

    ⛔ 29.08, юзер поймал: прежняя сборка `[j for st in steps for j in st["jobs"]]` шла ПО
    ШАГУ — сперва схлопывание всех гео, потом разметка всех. При массовом прогоне на много
    стран пауза на бюджете (день исчерпан) размазывала бы недоделанность по всем странам
    сразу, вместо чистой границы «эти готовы, эта в работе». Сборка/готовность (глобальные,
    без гео) обязаны стоять в ХВОСТЕ каждой страны, а не только в конце всего цикла.
    """
    s = {
        "collapse": ["gr"],
        "mark": [{"geo": "gr", "n": 5}],
        "mark_n": 5,
        "summarize": ["gr"],
        "build_corpus": ["gr"],
        "to_translate": ["gr", "gb"],
        "geos": 2,
        "views": 10,
        "build_done": False,
        "readiness_done": False,
    }
    chain = bot._country_major_chain(bot.pipeline_steps(s))
    assert chain == [
        ("collapse", "gr"),
        ("mark", "gr"),
        ("summarize", "gr"),
        ("build_corpus", "gr"),
        ("translate", "gr"),
        ("build", None),
        ("readiness", None),
        ("translate", "gb"),
        ("build", None),
        ("readiness", None),
    ], chain


def test_one_bad_button_does_not_kill_the_pult():
    """Обработка обновления вызывается ПОД `try` — сбой стоит кнопки, а не пульта."""
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert "def handle_update(" in src, "обработка не вынесена из цикла"
    i = src.index("for u in r.get(")
    tail = src[i : i + 400]
    assert "try:" in tail and "handle_update(u, job)" in tail, tail
    assert "except Exception" in tail, tail


def test_pult_asks_the_tract_what_is_undone():
    """Работу шага считает САМ ШАГ — у пульта своей арифметики нет.

    Повод (22.08, Греция): правило «кого возьмёт разметка» жило дважды — в тракте и своей
    формулой в пульте. В тракт добавился фильтр представителей, пульт про него не узнал, и
    после полной разметки 751 из 751 шаг висел с «14 мух». Сторож требует ОДНОГО места.
    """
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert "_tract.undone(" in src, "пульт не спрашивает тракт о работе шага"
    assert (
        "ids - tagged.get(" not in src
    ), "в пульте осталась своя формула работы разметки"


def test_optional_argument_is_dropped_without_a_pair():
    """`mark cz` без пары обязан запускаться: лишний аргумент выкидывается из команды.

    ⛔ Повод: раньше на месте этой ветки стоял поиск устаревшей полки, и `mark cz` получал в
    ответ «полок из старой таксономии нет» вместо прогона.
    """
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert (
        'argv = [a for a in argv if "{shelf}" not in a]' in src
    ), "аргумент не выкидывается"
    assert "stale_shelf" not in src, "вернулась ветка про устаревшую полку"


def test_tract_writes_only_into_tests():
    """⛔ Прогоны новой схемы живут в `tests/` и только там (заказ юзера 20.08).

    Повод: 21.08 разметка ушла в боевую `tags/`, смешав 146 старых записей с 42 новыми, а свод
    переписал боевой корпус Чехии. Каталог обязан решаться ОДНОЙ константой.
    """
    src = (HERE.parent / "tract.py").read_text(encoding="utf-8")
    assert 'TESTS = "tests"' in src, "каталог прогонов не задан константой"
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if 'f"tags/' in code or 'f"out_facet/' in code:
            raise AssertionError(f"тракт пишет в боевую папку: {line.strip()}")


def test_build_assembles_pages_before_rendering():
    """Сборка = страницы КАЖДОГО языка, потом ОДИН рендер. `rendered=0` уже случалось —
    и на пустом языке (собирали раньше переводов), и на едином рендере без сборки.
    """
    cmd = " ".join(bot.MENU["build"][1])
    assert "site.py" in cmd and "render.py" in cmd, cmd
    assert cmd.index("site.py") < cmd.rindex("site.py") < cmd.index("render.py"), cmd
    # ⛔ Каталог задаётся ПЕРЕМЕННЫМИ: `site.py` читает корпус из BUILT_DIR/PSEO_DATA,
    # `render.py` пишет в PSEO_OUT. С одним `cd` сборка брала бы боевой корпус — проверено
    # прогоном 21.08.
    assert "BUILT_DIR=" in cmd and "/tests" in cmd, cmd
    assert "PSEO_DATA=" in cmd and "/tests/data" in cmd, cmd
    assert "PSEO_OUT=" in cmd and "/tests/out" in cmd, cmd
    # ⛔ ПО ВСЕМ ЯЗЫКАМ, не только ru (28.08): раньше `site.py --all` без языка собирал
    # молча только русский, а тринадцать остальных не строились никогда.
    assert cmd.count("site.py --all") == len(bot._SITE["languages"]), cmd
    for lang in bot._SITE["languages"]:
        assert f"site.py --all {lang}" in cmd, f"{lang} не собирается: {cmd}"
    assert cmd.count("render.py --all") == 1, "render должен идти одним проходом"


def test_state_counts_the_test_folder():
    """Состояние считает по тестовой разметке, иначе шаги покажут боевые числа."""
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert 'TESTS = "tests"' in src
    assert "{BRAIN}/{TESTS}/tags/" in src, "состояние читает боевые теги"
