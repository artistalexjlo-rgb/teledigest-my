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
# ⚠️ Основное дерево билдера, а не копия пульта: в копии нет `country_codes`, её кладёт
# образ. Без него `import facet` падает, и состояние возвращает ошибку вместо работы.
sys.path[:0] = [str(HERE), str(HERE.parent / "builder")]

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
    """Разметка и списки зовут НОВЫЙ модуль тракта, а не старый facet."""
    assert "mark" in bot.MENU and "svod" in bot.MENU
    for kind, flag in (("mark", "--mark"), ("svod", "--svod")):
        argv = bot.MENU[kind][1]
        assert any(a.endswith("tract.py") for a in argv), argv
        assert flag in argv and "{geo}" in argv, argv


def test_old_steps_are_gone():
    """Старые кнопки не должны вернуться — вместе с ними вернулась бы старая нарезка."""
    lишние = SNYATO & set(bot.MENU)
    assert not lишние, f"старые кнопки на месте: {sorted(lишние)}"


def test_steps_go_in_tract_order():
    """Вертикаль меню = порядок тракта: схлопывание → разметка → списки → сборка → переводы.

    Схлопывание стоит ПЕРВЫМ не для красоты: размечать оно даёт представителей, и после
    разметки его звать поздно — рту уже заплачено за каждую почти-копию.
    """
    kinds = [
        st["kind"]
        for st in bot.pipeline_steps(
            {"mark": [], "mark_n": 0, "svod": [], "geos": 0, "views": 0}
        )
    ]
    assert kinds == ["sgusti", "mark", "svod", "build", "translate"], kinds


def test_work_is_counted_as_undone(tmp_path, monkeypatch):
    """Работа шага «списки» = гео, у которых разметка новее корпуса или корпуса нет.

    ⛔ Иначе шаг покажет ✅ на несобранном гео, и упавшее заново не встанет (канон §0.17).
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    (tmp_path / "tests" / "tags").mkdir(parents=True)
    (tmp_path / "tests" / "tags" / "cz.json").write_text(
        '[{"id": "a", "perevod": "текст", "tema": "transport", "podtema": "аренда"}]',
        encoding="utf-8",
    )
    st = bot.pipeline_state()
    assert "cz" in st["svod"], st["svod"]


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
            "svod": ["cz"],
            "geos": 1,
            "views": 12,
            "langs": [],
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
            "sgusti": ["gr", "me"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "svod": [],
            "geos": 0,
            "views": 0,
            "langs": [],
        },
    )
    bot.send_menu(None)
    text = str(sent[-1][1])
    assert "run:sgusti:gr" in text, text
    assert "run:sgusti:me" in text, text
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
            "sgusti": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "svod": [],
            "geos": 0,
            "views": 0,
            "langs": [],
        },
    )
    bot.send_menu(None)
    text = str(sent[-1][1])
    assert "run:sgusti:gr" in text and "run:mark:gr" in text, text
    for chuzhoy in ("br", "kr", "vn", "any"):
        assert f"run:sgusti:{chuzhoy}" not in text, chuzhoy
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
            "sgusti": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "svod": [],
            "geos": 0,
            "views": 0,
            "langs": [],
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
            "sgusti": ["gr"],
            "mark": [{"geo": "gr", "n": 765}],
            "mark_n": 765,
            "svod": [],
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
    assert j.started == ("sgusti", "gr"), getattr(j, "started", None)
    text = " ".join(said)
    assert "весь тракт" in text, text
    # разметка 765/25=31 вызов, списки ≤(13+8) вызовов, по 4 запроса worst-case
    assert "~" in text and "запросов" in text, text


def test_one_bad_button_does_not_kill_the_pult():
    """Обработка обновления вызывается ПОД `try` — сбой стоит кнопки, а не пульта."""
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert "def handle_update(" in src, "обработка не вынесена из цикла"
    i = src.index("for u in r.get(")
    tail = src[i : i + 400]
    assert "try:" in tail and "handle_update(u, job)" in tail, tail
    assert "except Exception" in tail, tail


def test_swallowed_flies_are_not_counted_as_work(tmp_path, monkeypatch):
    """Проглоченные схлопыванием мухи — не работа разметки.

    Повод (22.08, Греция): схлопывание оставило 751 представителя из 765, разметка прошла
    все 751 — а шаг остался красным с «14 мух». Эти 14 разметке недоступны никогда:
    тракт сам берёт только представителей, и кнопка звала вхолостую.
    """
    brain = tmp_path
    monkeypatch.setattr(bot, "BRAIN", str(brain))
    os.makedirs(brain / bot.TESTS / "dedup", exist_ok=True)
    with open(brain / bot.TESTS / "dedup" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"groups": [{"rep": "a", "ids": ["a", "b"]}, {"rep": "c", "ids": ["c"]}]},
            fh,
        )
    assert bot._reps("gr") == {"a", "c"}
    assert bot._reps("me") == set(), "у гео без схлопывания представителей нет"


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
    src = (HERE.parent / "builder" / "tract.py").read_text(encoding="utf-8")
    assert 'TESTS = "tests"' in src, "каталог прогонов не задан константой"
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if 'f"tags/' in code or 'f"out_facet/' in code:
            raise AssertionError(f"тракт пишет в боевую папку: {line.strip()}")


def test_build_assembles_pages_before_rendering():
    """Сборка = страницы, потом рендер. Один рендер даёт `rendered=0` — это уже случалось."""
    cmd = " ".join(bot.MENU["build"][1])
    assert "pages.py" in cmd and "render.py" in cmd, cmd
    assert cmd.index("pages.py") < cmd.index("render.py"), cmd
    # ⛔ Каталог задаётся ПЕРЕМЕННЫМИ: `pages.py` читает корпус из BUILT_DIR, `render.py`
    # пишет в PSEO_OUT. С одним `cd` сборка брала бы боевой корпус — проверено прогоном.
    assert "BUILT_DIR=" in cmd and "/tests" in cmd, cmd
    assert "PSEO_OUT=" in cmd and "/tests/out" in cmd, cmd
    # ⛔ И середина тракта — собранные страницы — тоже в `tests/`, ОБЕИМ половинам. Без
    # `PSEO_DATA` они ложились в `/app/data` внутри образа: вне маунта, вне тестовой папки
    # и насмерть при редеплое.
    halves = cmd.split("&&")
    assert len(halves) == 2, cmd
    for half in halves:
        assert "PSEO_DATA=" in half and "/tests/data" in half, half


def test_state_counts_the_test_folder():
    """Состояние считает по тестовой разметке, иначе шаги покажут боевые числа."""
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    assert 'TESTS = "tests"' in src
    assert "{BRAIN}/{TESTS}/tags/" in src, "состояние читает боевые теги"
