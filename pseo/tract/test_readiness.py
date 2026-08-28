# -*- coding: utf-8 -*-
"""Сторож звена 7: гейт честно отражает проблему и честно отражает её отсутствие.

Настоящий `readycheck.py`/`render.py` сюда не зовём (у них свои сторожа) — подменяем
`subprocess.run`, проверяем только СКЛЕЙКУ: код передал правильные каталоги, прочитал
`ready.json`, вернул верный вердикт.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import readiness  # noqa: E402


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_problem_reported_by_the_gate_fails_the_check(tmp_path, monkeypatch):
    """readycheck нашёл проблему — check() возвращает False, а не молчит."""

    def fake_run(cmd, cwd, env, capture_output, text):
        json.dump(
            {"страниц_всего": 5, "готово_к_деплою": 3, "проблем": 2},
            open(f"{readiness.HERE}/ready.json", "w", encoding="utf-8"),
        )
        return _Proc(1)

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    ok, rep = readiness.check("built", "data", "out")
    assert ok is False
    assert rep["проблем"] == 2


def test_clean_snapshot_passes(tmp_path, monkeypatch):
    """Ни одной проблемы — check() возвращает True с числами."""

    def fake_run(cmd, cwd, env, capture_output, text):
        json.dump(
            {"страниц_всего": 5, "готово_к_деплою": 5, "проблем": 0},
            open(f"{readiness.HERE}/ready.json", "w", encoding="utf-8"),
        )
        return _Proc(0)

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    ok, rep = readiness.check("built", "data", "out")
    assert ok is True
    assert rep["готово_к_деплою"] == 5


def test_directories_are_passed_through_env_untouched(monkeypatch):
    """Каталоги идут ТЕ, что передали, — не боевые по умолчанию, не выдуманные."""
    seen = {}

    def fake_run(cmd, cwd, env, capture_output, text):
        seen["BUILT_DIR"] = env["BUILT_DIR"]
        seen["PSEO_DATA"] = env["PSEO_DATA"]
        seen["PSEO_OUT"] = env["PSEO_OUT"]
        json.dump(
            {"проблем": 0}, open(f"{readiness.HERE}/ready.json", "w", encoding="utf-8")
        )
        return _Proc(0)

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    readiness.check("/brain/tests", "/brain/tests/data", "/brain/tests/out")
    assert seen == {
        "BUILT_DIR": "/brain/tests",
        "PSEO_DATA": "/brain/tests/data",
        "PSEO_OUT": "/brain/tests/out",
    }


def test_no_report_file_is_a_failure_not_a_silent_pass(monkeypatch, tmp_path):
    """readycheck не написал ready.json (упал раньше) — check() не считает это готовностью."""
    old_ready = f"{readiness.HERE}/ready.json"
    if os.path.exists(old_ready):
        os.remove(old_ready)

    def fake_run(cmd, cwd, env, capture_output, text):
        return _Proc(1, stderr="упал")

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    ok, rep = readiness.check("built", "data", "out")
    assert ok is False
    assert "ошибка" in rep
