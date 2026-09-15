# -*- coding: utf-8 -*-
"""keybroker — АДРЕС, не код. Мозг живёт ОДНИМ экземпляром в
`src/teledigest/keybroker.py` (15.09: экстрактор бота переведён на мозг, и код обязан быть
общим, а не копией — копия разошлась бы за неделю).

Здесь только загрузка того файла под именем `keybroker`, чтобы `import keybroker` /
`from keybroker import call` в тракте и пульте работали как раньше. `sys.modules` подменяется
на настоящий модуль — monkeypatch в тестах бьёт по нему же, а не по обёртке.

Где искать файл: `KEYBROKER_PATH` (образ пульта кладёт его в /app/_brain/keybroker.py),
иначе — репозиторий, `../../src/teledigest/keybroker.py` относительно этого файла.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.environ.get("KEYBROKER_PATH") or os.path.normpath(
    os.path.join(_HERE, "..", "..", "src", "teledigest", "keybroker.py")
)
if not os.path.exists(_PATH):
    raise ImportError(
        f"мозг не найден: {_PATH} — задай KEYBROKER_PATH или проверь COPY в Dockerfile"
    )
_spec = importlib.util.spec_from_file_location(__name__, _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _mod
_spec.loader.exec_module(_mod)
