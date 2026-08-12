"""Сторож: каталог вывода — ОДНО определение на весь тракт.

⛔ Зачем. `BASE/out` было выписано трижды: в `render.py`, `readycheck.py` и `ship.py`. Пока
все трое бегут на десктопе, копии совпадают и проблема невидима. Но публикация переезжает в
пульт, где рендер обязан писать в примонтированный каталог (`PSEO_OUT`): с тремя копиями
рендер напишет в маунт, гейт проверит пустой `/app/out` и отрапортует «готово 0» при
собранном сайте, а push уедет с третьим. Это ровно та болезнь, которая на этом проекте
стоила месяца публикации, — одно знание в нескольких местах.

Проверяем поведением, в отдельном процессе: подсовываем `PSEO_OUT` и смотрим, что все три
модуля показывают ОДИН путь. Грепом по исходникам такое не поймать — копия может быть
вычислена иначе и всё равно совпасть текстом.
"""

import os
import pathlib
import subprocess
import sys

PSEO = pathlib.Path(__file__).resolve().parent.parent

PROBE = """
import sys
sys.path[:0] = [r"{pseo}", r"{pseo}/builder"]
import render, readycheck, ship
print(str(render.OUT))
print(str(readycheck.OUT))
print(str(ship.OUT))
"""


def _probe(env_out: str | None) -> list[str]:
    env = dict(os.environ)
    env.pop("PSEO_OUT", None)
    if env_out:
        env["PSEO_OUT"] = env_out
    r = subprocess.run(
        [sys.executable, "-c", PROBE.format(pseo=PSEO.as_posix())],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PSEO),
    )
    assert r.returncode == 0, f"проба не запустилась: {r.stderr[-800:]}"
    return [pathlib.Path(x).as_posix() for x in r.stdout.strip().splitlines()]


def test_default_is_out_next_to_pseo():
    """Без переменной поведение прежнее — `pseo/out`, чтобы десктопный тракт не поехал."""
    got = _probe(None)
    assert len(got) == 3
    assert all(p == (PSEO / "out").as_posix() for p in got), got


def test_env_moves_all_three(tmp_path):
    """С `PSEO_OUT` переезжают ВСЕ трое. Если хоть один остался на своём — тест красный."""
    target = (tmp_path / "site").as_posix()
    got = _probe(target)
    assert got == [target, target, target], got


def test_render_writes_where_out_points(tmp_path, monkeypatch):
    """Не только константа совпадает, но и запись идёт туда же: рендер обязан создать
    sitemap/robots в каталоге из `PSEO_OUT`, а не рядом с кодом."""
    monkeypatch.setenv("PSEO_OUT", str(tmp_path / "site"))
    for mod in ("render", "config.site"):
        sys.modules.pop(mod, None)
    sys.path[:0] = [str(PSEO)]
    import render

    assert pathlib.Path(render.OUT) == tmp_path / "site"
    (tmp_path / "site").mkdir(parents=True, exist_ok=True)
    n = render.copy_assets()  # первый писатель, который зовётся в build_all
    assert n >= 1, "ассеты не скопировались — писать в PSEO_OUT не получилось"
    assert (tmp_path / "site" / "assets").is_dir()
    sys.modules.pop("render", None)
