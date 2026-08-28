# -*- coding: utf-8 -*-
"""ЗВЕНО 7 ГОТОВНОСТЬ: проверить снимок песочницы, до боевой публикации ещё не дошли.

PLAN.md, звено 7: «рендер, карта сайта, выкладка ПОЛНЫМ снимком в каталог раздачи». Первая
часть уже есть — шаг «Сборка» зовёт `render.py --all` и кладёт снимок в `{BRAIN}/tests/out`.
Второй половины — проверки, что снимок не битый, — не было вовсе. Здесь она и есть.

⛔ **Каталога раздачи СЕГОДНЯ нет** (юзер 28.08: «сейчас есть только одна папка — тест»).
Раздача наружу — отдельная задача, когда пульт пройдёт пробу до конца; этот файл её не
делает и не готовит: ни `rsync`, ни боевого каталога, ни второго домена тут нет.

⛔ **Модуль новый, живёт в `pseo/tract/`** — истина тракта только здесь. `readycheck.py`
и `render.py` (28.08) тоже переписаны заново и живут рядом, в этом же каталоге — не
перенесены, а переписаны: старый `readycheck.py` проверял только русский язык (см. его
собственный докстринг и `test_readycheck.py`).
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # …/pseo/tract — тут же readycheck.py


def check(built_dir, data_dir, out_dir):
    """Прогон readycheck.py на указанные каталоги. Возвращает (готово_ли, отчёт)."""
    env = dict(os.environ)
    env["BUILT_DIR"] = built_dir
    env["PSEO_DATA"] = data_dir
    env["PSEO_OUT"] = out_dir
    r = subprocess.run(
        [sys.executable, f"{HERE}/readycheck.py"],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    ready_path = f"{HERE}/ready.json"
    rep = None
    if os.path.exists(ready_path):
        rep = json.load(open(ready_path, encoding="utf-8"))
    if r.returncode != 0 or not rep:
        return False, rep or {"ошибка": (r.stderr or r.stdout)[-500:]}
    return not rep.get("проблем"), rep


if __name__ == "__main__":
    brain = os.environ.get("BRAIN_DIR", "/root/pseo_builder")
    ok, rep = check(f"{brain}/tests", f"{brain}/tests/data", f"{brain}/tests/out")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)
