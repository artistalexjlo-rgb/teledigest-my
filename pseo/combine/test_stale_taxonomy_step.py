"""Сторож пульта: шаг «Хвост→полки» видит УСТАРЕВШУЮ таксономию, а не только отсутствие полок.

⛔ Повод, замеренный 13.08. Счётчик шага наполнялся по правилу «у гео вообще нет полок» — и
показывал «1 гео», хотя набор полок сменился и пере-разложить надо все. Пляжи и рестораны
лежали под подписью «Работа, учёба, сообщества и быт», а для пульта это выглядело как
«полки есть, работать не надо». Тот же класс, что весь тот день: счётчик мерит НАЛИЧИЕ
вместо ВЕРНОСТИ, и кривая раскладка проходит как успех.

Версия таксономии уже пишется в файл гео самим facet — оставалось её читать.
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


def _corpus(root, geo, version, shelf_name="Визовые процедуры"):
    d = root / "out_facet"
    d.mkdir(exist_ok=True)
    (d / f"{geo}.json").write_text(
        json.dumps(
            {
                "geo": geo,
                "views_by_task": [
                    {"zadacha": "T", "items": [{"id": f"{geo}{i}"} for i in range(5)]}
                ],
                "shelves": [{"shelf": shelf_name, "items": [{"id": f"{geo}9"}]}],
                "prochee": [],
                "taxonomy_version": version,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_version_comes_from_taxonomy_not_literal():
    """Версию пульт берёт из того же модуля, что и рты. Литерал разъехался бы при следующем
    подъёме — на копиях чисел этот проект уже горел."""
    assert bot._TAX_VERSION == tax.VERSION


def test_stale_geo_is_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _corpus(tmp_path, "gr", "v0-2026-07-19")
    _corpus(tmp_path, "br", tax.VERSION)
    st = bot.pipeline_state()
    assert "gr" in st["stale_tax"], "гео со старой версией не попало в работу шага"
    assert "br" not in st["stale_tax"], "гео с текущей версией зря позвали"


def test_step_label_and_jobs_include_stale(tmp_path, monkeypatch):
    """Шаг обязан и посчитать гео, и выдать РАБОТУ — иначе кнопка нарисуется пустой."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _corpus(tmp_path, "gr", "v0-2026-07-19")
    st = bot.pipeline_state()
    step = next(s for s in bot.pipeline_steps(st) if s["kind"] == "assign")
    assert "1 гео" in step["label"], step["label"]
    assert ("reshelf", "gr") in step["jobs"], step["jobs"]


def test_stale_shelf_name_comes_from_data(tmp_path, monkeypatch):
    """Имя полки для целевого режима берётся из данных: полка, которой нет в таксономии.
    Так переименование или разбор полки пульт узнаёт сам, без правки константы."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _corpus(
        tmp_path, "gr", "v0-2026-07-19", shelf_name="Работа, учёба, сообщества и быт"
    )
    assert bot.stale_shelf("gr") == "Работа, учёба, сообщества и быт"
    _corpus(tmp_path, "br", "v0-2026-07-19", shelf_name="Визовые процедуры")
    assert (
        bot.stale_shelf("br") is None
    ), "живую полку нельзя предлагать к пере-раскладке"


def test_menu_has_targeted_mode():
    """Целевой режим доступен кнопкой, и команда несёт и гео, и имя полки."""
    assert "reshelf" in bot.MENU
    argv = bot.MENU["reshelf"][1]
    assert "--reassign-shelf" in argv and "{geo}" in argv and "{shelf}" in argv
