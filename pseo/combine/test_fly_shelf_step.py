"""Сторож пульта: шаг «раздел мухам» — ось нарезки запускается КНОПКОЙ, не с десктопа.

Ось нарезки (канон §0.15) держится на том, что у каждой мухи есть раздел. Если этот шаг
нельзя нажать из пульта, ось живёт только у ассистента на машине — а заказ юзера дословно:
«мне нужен путь, работающий без тебя».

⛔ Работа шага считается по СКЛАДУ РАЗМЕТКИ (`tags/`), а не по корпусу: раздел ставится
МУХЕ. Счётчик, который мерил бы не то, — та же болезнь, что весь день 13.08: шаг зовёт
сделанное или молчит о несделанном.
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


def _tags(root, geo, flies):
    d = root / "tags"
    d.mkdir(exist_ok=True)
    (d / f"{geo}.json").write_text(
        json.dumps(flies, ensure_ascii=False), encoding="utf-8"
    )


def _fly(fid, shelf_key=None):
    r = {"id": fid, "perevod": f"Совет {fid}.", "zadachi": ["Дело"], "sushnosti": []}
    if shelf_key:
        r["shelf_key"] = shelf_key
    return r


def test_flies_without_section_are_work(tmp_path, monkeypatch):
    """Муха без раздела — работа шага. Муха с разделом — нет: иначе шаг зовёт сделанное."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a"), _fly("b", "visa")])
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == ["gr"], st["no_fly_shelf"]
    assert st["no_fly_shelf_n"] == 1, "с разделом не должна считаться работой"


def test_all_marked_means_no_work(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a", "visa"), _fly("b", "prochee")])
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == [] and st["no_fly_shelf_n"] == 0


def test_fails_file_is_not_counted_as_geo(tmp_path, monkeypatch):
    """⚠️ Рядом с разметкой лежит `<geo>_fails.json` — это не гео, и работой быть не может."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a")])
    # ⛔ Фикстура нарочно в ХУДШЕЙ форме — списком записей: боевой файл сбоев это словарь
    # {id: счётчик}, и на словаре мутация «не отсеивать _fails» оставалась зелёной, потому
    # что перебор словаря даёт строки. Отсев по имени обязан работать при любой форме.
    (tmp_path / "tags" / "gr_fails.json").write_text(
        json.dumps([{"id": "x"}]), encoding="utf-8"
    )
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == ["gr"], st["no_fly_shelf"]
    assert st["no_fly_shelf_n"] == 1, st["no_fly_shelf_n"]


def test_step_carries_the_job_and_says_it(tmp_path, monkeypatch):
    """Шаг обязан и выдать задание, и сказать о нём в подписи — иначе кнопка пустая."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a")])
    st = bot.pipeline_state()
    step = next(s for s in bot.pipeline_steps(st) if s["kind"] == "assign")
    assert ("flyshelf", "gr") in step["jobs"], step["jobs"]
    assert "мухи" in step["label"], step["label"]


def test_menu_calls_the_new_flag():
    """Кнопка зовёт тот же `facet.py` с флагом оси. Проверяем СМЫСЛ, не позиции в списке."""
    assert "flyshelf" in bot.MENU
    argv = bot.MENU["flyshelf"][1]
    assert any(a.endswith("facet.py") for a in argv), argv
    assert "--assign-flies" in argv, argv
    assert "{geo}" in argv, argv
