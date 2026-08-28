"""Сторож: в образе пульта есть ВСЁ, что нужно рендеру сайта.

⛔ Зачем. Публикация до сих пор возможна только с десктопа: в пульте живёт мозг (facet,
keybroker), а рендера — `render.py`, `site.py`, `templates/`, `i18n/`,
`static/` — там не было вообще. Класс отказа, который тут ловится, для проекта родной:
код зовёт модуль, на десктопе он есть, в образе его нет, и падает это в проде.

Проверяем не «лежит ли файл», а **покрыт ли он инструкцией COPY** — образ собирает
Dockerfile, и правда про состав образа лежит в нём. Нужное берём из САМОГО КОДА (разбор
импортов через ast), а не из списка пожеланий: список отстанет, разбор — нет.
"""

import ast
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent  # .../pseo/combine
PSEO = HERE.parent  # .../pseo
ROOT = PSEO.parent  # корень репо = контекст сборки
DOCKERFILE = HERE / "Dockerfile"

# Точки входа рендера. Всё остальное вычисляется из их импортов.
# плоский адрес страницы) и в образ пульта больше не едет (26.08).
ENTRIES = [
    "render.py",
    "tract/site.py",
    # ⛔ Переводчик тракта — `translation.py`. Старый `facet_lang.py` уехал в legacy 27.08:
    # он переводил С РУССКОГО и ждал корпус отменённой схемы.
    "tract/translation.py",
    "builder/readycheck.py",
]

# Каталоги, которые рендер читает ПО ИМЕНИ в рантайме (ast их не видит):
# шаблоны — FileSystemLoader, словари — load_i18n, ассеты — copy_assets, site-конфиг.
RUNTIME_DIRS = ["templates", "i18n", "static", "config"]


def _copy_sources() -> list[str]:
    """Источники всех COPY из Dockerfile (пути относительно контекста сборки)."""
    out = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*COPY\s+(.+)$", line, re.I)
        if not m:
            continue
        parts = m.group(1).split()
        out += parts[:-1]  # последний аргумент — приёмник
    return out


SOURCES = _copy_sources()


def _copied(rel_from_root: str) -> bool:
    """Попадает ли путь ОТ КОРНЯ РЕПО в образ — сам или вместе с родительским каталогом.

    ⚠️ От корня, а не от `pseo/`: модуль репо может лежать и вне него — справочник стран
    живёт в `src/teledigest/`, и сборщик берёт имена оттуда.
    """
    target = rel_from_root.rstrip("/")
    for src in SOURCES:
        s = src.rstrip("/")
        if target == s or target.startswith(s + "/"):
            return True
    return False


def test_context_is_repo_root():
    """Все COPY адресуются ОТ КОРНЯ РЕПО. Это не стиль, а условие существования схемы: из
    контекста `pseo/combine` файлы рендера недостижимы, и их пришлось бы дублировать.

    ⛔ Проверяем СВОЙСТВО, а не префикс. Сначала тут стояло «строка начинается с pseo/» — и
    правило покраснело на первом же законном пути из `src/`, хотя свойство соблюдалось.
    Сторож, проверяющий похожесть вместо сути, ломается на правильном коде.
    """
    tops = {x.name for x in ROOT.iterdir()}
    bad = [s for s in SOURCES if s.split("/")[0] not in tops]
    assert not bad, f"COPY не от корня репо: {bad} — контекст сборки должен быть `.`"


def test_copy_sources_exist():
    """Каждый источник COPY существует в репо. Переименовали каталог — падает тут, а не
    на сборке образа в панели, где юзер увидит только «deploy failed»."""
    missing = [s for s in SOURCES if not (ROOT / s).exists()]
    assert not missing, f"COPY указывает на несуществующее: {missing}"


def test_entries_copied():
    for rel in ENTRIES:
        assert (PSEO / rel).exists(), f"{rel} нет в репо"
        assert _copied(f"pseo/{rel}"), f"{rel} не попадает в образ"


def test_runtime_dirs_copied():
    for d in RUNTIME_DIRS:
        assert (PSEO / d).is_dir(), f"pseo/{d} нет в репо"
        assert _copied(f"pseo/{d}"), f"каталог {d} не попадает в образ"


def _local_and_third_party(path: pathlib.Path):
    """Импорты файла, разложенные на «локальный модуль репо» и «сторонний пакет».

    Локальный отдаётся СПИСКОМ мест, где модуль реально лежит: один и тот же `slugs`
    существует и в `builder/`, и в дубле `combine/tract/`. В образе достаточно любого —
    в `/app/builder/` они приезжают в одно место.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    names -= set(sys.stdlib_module_names)
    local, third = {}, set()
    for n in sorted(names):
        here = path.parent.name if path.parent != PSEO else ""
        cands = [
            f"pseo/{here}/{n}.py" if here else f"pseo/{n}.py",
            f"pseo/{n}.py",
            f"pseo/{n}/__init__.py",
            f"pseo/combine/tract/{n}.py",
            f"pseo/{n}" if (PSEO / n).is_dir() else "",  # пакет без __init__ (config/)
        ]
        found = [c for c in cands if c and (ROOT / c).exists()]
        if found:
            local[n] = found
        else:
            third.add(n)
    return local, third


def test_local_imports_copied():
    """Каждый модуль репо, который зовёт рендер, тоже едет в образ."""
    for rel in ENTRIES:
        local, _ = _local_and_third_party(PSEO / rel)
        for name, places in local.items():
            assert any(
                _copied(p) for p in places
            ), f"{rel} импортирует {name} — ни одно из {places} не едет в образ"


# Расходятся ОСМЫСЛЕННО, копировать целиком нельзя (28.07 чуть не сломало прод):
#   keybroker.py  — свой `get_keys` (ключи приходят из env контейнера);
#   lang_runner.py — `PY=sys.executable` (хостового venv в контейнере нет) и раздельные
#                    HERE (каталог дублей) / DATA (примонтированные данные хоста).
DIVERGING = {"keybroker.py"}


def test_duplicates_identical():
    """Дубли ртов, кроме двух осмысленно расходящихся, обязаны быть ДОСЛОВНЫМИ копиями.

    ⛔ Правило «правишь одну копию — правь и вторую» до сих пор держалось на моей памяти,
    и трижды не удержалось. Тут оно проверяется машиной: разъехался файл — падает тест,
    а не прод. Осознанное расхождение добавляется в DIVERGING вместе с причиной.
    """
    drift = []
    for src in sorted((PSEO / "combine" / "tract").glob("*.py")):
        twin = PSEO / "tract" / src.name
        if not twin.exists() or src.name in DIVERGING:
            continue
        a = twin.read_bytes().replace(b"\r\n", b"\n")
        b = src.read_bytes().replace(b"\r\n", b"\n")
        if a != b:
            drift.append(src.name)
    assert (
        not drift
    ), f"дубли разъехались: {drift} — синхронизировать или внести в DIVERGING"


def test_third_party_in_requirements():
    """Сторонние пакеты рендера объявлены — иначе контейнер упадёт на импорте."""
    req = (HERE / "requirements.txt").read_text(encoding="utf-8").lower()
    have = {
        re.split(r"[=<>~\[]", ln.strip())[0] for ln in req.splitlines() if ln.strip()
    }
    for rel in ENTRIES:
        _, third = _local_and_third_party(PSEO / rel)
        for pkg in third:
            assert pkg.lower() in have, f"{rel} требует {pkg}, нет в requirements.txt"


def test_readme_documents_panel_fields():
    """Схема живёт в двух местах: Dockerfile и поля в панели. Второе — руки юзера, значит
    оно обязано быть в README, иначе редеплой соберёт образ старым контекстом и упадёт.
    """
    txt = (HERE / "README.md").read_text(encoding="utf-8")
    for field in ("Dockerfile Path", "Docker Context Path"):
        assert field in txt, f"README не описывает поле «{field}» для панели Dokploy"
