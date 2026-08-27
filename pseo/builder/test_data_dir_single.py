"""Сторож: каталог СОБРАННЫХ СТРАНИЦ — одно определение на обе половины сборки.

⛔ Зачем. Сборка идёт в два шага: `site.py` превращает корпус в страницы (json), `render.py`
их рендерит. Каталог вывода рендера (`PSEO_OUT`) и каталог корпуса (`BUILT_DIR`) переменными
задавались, а середина — нет: сборщик писал в `BASE/data` литералом с 11.07. В пульте это
`/app/data` ВНУТРИ образа: не в маунте, не в `tests/`, пропадает при редеплое, снаружи не
посмотреть. Испытательный прогон новой схемы обязан жить в `tests/` целиком.

Проверяем поведением, в отдельном процессе: подсовываем `PSEO_DATA` и смотрим, что обе
половины показывают ОДИН путь, а без переменной — прежний `pseo/data` (десктопный тракт
не должен поехать).
"""

import os
import pathlib
import subprocess
import sys

PSEO = pathlib.Path(__file__).resolve().parent.parent

# ⛔ Сборщик грузим ПО ПУТИ, а не `import site`: `site` — имя стандартного модуля Python,
# и обычный импорт приносит его, а не наш файл. Поймано этим же сторожем 27.08.
PROBE = """
import sys, importlib.util
sys.path[:0] = [r"{pseo}", r"{pseo}/builder"]
import render
_s = importlib.util.spec_from_file_location("sitebuild", r"{pseo}/builder/site.py")
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
print(str(render.DATA))
print(str(_m.DATA))
"""


def _probe(env_data: str | None) -> list[str]:
    env = dict(os.environ)
    env.pop("PSEO_DATA", None)
    if env_data:
        env["PSEO_DATA"] = env_data
    r = subprocess.run(
        [sys.executable, "-c", PROBE.format(pseo=PSEO.as_posix())],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PSEO),
    )
    assert r.returncode == 0, f"проба не запустилась: {r.stderr[-800:]}"
    return [pathlib.Path(x).as_posix() for x in r.stdout.strip().splitlines()]


def test_default_is_data_next_to_pseo():
    """Без переменной обе половины смотрят в `pseo/data` — прежнее поведение."""
    got = _probe(None)
    assert got == [(PSEO / "data").as_posix()] * 2, got


def test_env_moves_both_halves(tmp_path):
    """С `PSEO_DATA` переезжают ОБЕ. Осталась одна на своём — тест красный: рендер собрал бы
    сайт из боевого корпуса, а тестовые страницы остались бы лежать."""
    target = (tmp_path / "site").as_posix()
    got = _probe(target)
    assert got == [target, target], got


def test_builder_writes_where_data_points(tmp_path, monkeypatch):
    """Не только константа совпадает, но и запись идёт туда же — и каталог заводится сам:
    в `tests/` его до первого прогона нет."""
    target = tmp_path / "site"
    monkeypatch.setenv("PSEO_DATA", str(target))
    for mod in ("sitebuild",):
        sys.modules.pop(mod, None)
    sys.path[:0] = [str(PSEO / "builder")]
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "sitebuild", str(PSEO / "builder" / "site.py")
    )
    site = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(site)

    assert pathlib.Path(site.DATA) == target
    site._write("proba.json", {"path": "/ru/gr/proba/", "title": "проба"})
    assert (target / "proba.json").is_file(), "страница не легла в PSEO_DATA"
    sys.modules.pop("sitebuild", None)
