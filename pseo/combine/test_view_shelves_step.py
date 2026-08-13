"""Сторож: полку виду даёт РОТ `assign`, и эта работа стоит в шаге пульта.

Задача юзера дословно (13.08): «прогнать нужные полки методом пульта». Способ ОДИН —
существующий рот, та же закрытая таксономия, тот же учёт ключей. Свою векторную механику я
успел написать и снёс: заказа не было, а второй способ = вторая правда.

⛔ Каждое правило здесь ПРОВЕРЕНО ПОЛОМКОЙ: код ломали, тест краснел, код возвращали. Тест,
который зелёный на правильном коде, сам по себе не доказывает ничего — он мог бы быть зелёным
всегда. Утром такое уже было: сторож проверял «строка начинается с pseo/» вместо «путь от
корня репо» и покраснел на верном коде.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "builder")]

os.environ.setdefault("COMBINE_BOT_TOKEN", "test")
os.environ.setdefault("ADMIN_ID", "1")

import bot  # noqa: E402
import tail_taxonomy as tax  # noqa: E402


def _view(n=5, shelf=None):
    v = {"zadacha": "Тема", "items": [{"id": f"i{j}"} for j in range(n)]}
    if shelf:
        v["shelf"] = shelf
    return v


def _geo(root, name, views, version=None, shelf="Визовые процедуры"):
    d = root / "out_facet"
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(
        json.dumps(
            {
                "geo": name,
                "views_by_task": views,
                "shelves": [{"shelf": shelf, "items": [{"id": f"{name}9"}]}],
                "prochee": [],
                "taxonomy_version": version or tax.VERSION,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_views_without_shelf_are_work(tmp_path, monkeypatch):
    """Вид-страница без полки — работа. Вид с полкой — нет: иначе шаг зовёт сделанное и
    тратит ключи впустую (ровно та болезнь, что весь день: шаг считает не ту работу)."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _geo(tmp_path, "gr", [_view(), _view(shelf="Визовые процедуры")])
    st = bot.pipeline_state()
    assert st["no_view_shelf"] == ["gr"]
    assert st["no_view_shelf_n"] == 1, "с полкой не должен считаться работой"


def test_thin_views_need_no_shelf(tmp_path, monkeypatch):
    """Вид тоньше страничного порога страницей не станет, значит и тема ему не нужна."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _geo(tmp_path, "br", [_view(1), _view(3), _view(shelf="Транспорт и логистика")])
    st = bot.pipeline_state()
    assert st["no_view_shelf"] == [] and st["no_view_shelf_n"] == 0


def test_step_carries_views_job_and_says_it(tmp_path, monkeypatch):
    """Шаг обязан и выдать задание, и сказать о нём в подписи — иначе кнопка пустая."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _geo(tmp_path, "gr", [_view()])
    st = bot.pipeline_state()
    step = next(s for s in bot.pipeline_steps(st) if s["kind"] == "assign")
    assert ("assignv", "gr") in step["jobs"], step["jobs"]
    assert "темы" in step["label"], step["label"]


def test_views_block_does_not_break_shelf_chain(tmp_path, monkeypatch):
    """⛔ РЕГРЕСС 13.08: блок про виды я вставил между `if` и его `elif`, и `elif` про версию
    таксономии привязался к нему. Итог: `stale_tax` не наполнялся совсем, целевая
    пере-раскладка не предложилась бы никогда. Оба признака должны считаться независимо.
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _geo(
        tmp_path,
        "gr",
        [_view()],
        version="v0-2026-07-19",
        shelf="Работа, учёба, сообщества и быт",
    )
    st = bot.pipeline_state()
    assert st["stale_tax"] == ["gr"], "цепочка про полки сломана блоком про виды"
    assert st["no_view_shelf"] == ["gr"], "виды тоже должны считаться"


def test_menu_calls_the_existing_mouth():
    """Команда кнопки — тот же `facet.py` с флагом раскладки видов. Проверяем СМЫСЛ, а не
    позиции в списке: позиционная проверка проходит на неверной команде и падает на верной.
    """
    assert "assignv" in bot.MENU
    argv = bot.MENU["assignv"][1]
    assert any(a.endswith("facet.py") for a in argv), argv
    assert "--assign-views" in argv, argv
    assert not any("shelf_assign" in a for a in argv), "векторный велосипед вернулся"
