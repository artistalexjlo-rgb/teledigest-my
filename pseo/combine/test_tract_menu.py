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


def test_build_step_costs_no_keys():
    """Сборка — код, а не рот: зовёт рендер, а не билдерский рот."""
    argv = bot.MENU["build"][1]
    assert any(a.endswith("render.py") for a in argv), argv
    assert "--all" in argv, argv


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
    (tmp_path / "tags").mkdir()
    (tmp_path / "tags" / "cz.json").write_text(
        '[{"id": "a", "perevod": "текст", "tema": "transport", "podtema": "аренда"}]',
        encoding="utf-8",
    )
    st = bot.pipeline_state()
    assert "cz" in st["svod"], st["svod"]
