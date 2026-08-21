"""Сторож меню пульта под тракт «тема и подтема» (канон §0.19).

⛔ Старые кнопки сняты вместе с их кодом: место страницы решалось трижды (тема мухе по тексту,
список дел ротом, тема разбору снова ротом). Теперь тема и подтема ставятся ОДИН раз в разметке,
всё остальное — код. Сторож держит две вещи: новые кнопки на месте и старые не вернулись.
"""

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
    """Вертикаль меню = порядок тракта: разметка → списки → сборка → переводы."""
    kinds = [
        st["kind"]
        for st in bot.pipeline_steps(
            {"mark": [], "mark_n": 0, "svod": [], "geos": 0, "views": 0}
        )
    ]
    assert kinds == ["mark", "svod", "build", "translate"], kinds


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
