"""Сторож: «кто есть страница» решает ОДНО место — `tail_taxonomy.PAGE_MIN`.

⛔ Повод (19.08). Число решали ДЕВЯТЬ мест: два своих `PAGE_MIN` (facet, dedup) и семь
литералов `>= 4` (pages, facet_lang ×3, combine/bot ×3). Между ними была щель, уносившая
абзацы С САЙТА: дело из 2-3 мух проходило нарезку, мухи помечались нарезанными и в хвост уже
не попадали, а страницей вид не становился (`pages.py` режет по четырём). Верный предикат
хвоста при этом ЛЕЖАЛ В ТОМ ЖЕ ФАЙЛЕ тридцатью строками ниже — правило было написано верно
один раз и рядом же скопировано неверно.

Проверяем не «есть ли константа», а ОТСУТСТВИЕ ВТОРОЙ ПРАВДЫ: второго определения и литералов.
Живая связь (значение владельца реально решает состав видов) проверяется подменой в
`test_carve_axis.py::test_page_threshold_comes_from_the_owner_at_runtime`.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PSEO = HERE.parent
sys.path[:0] = [str(HERE)]

import tail_taxonomy as tax  # noqa: E402

OWNER = HERE / "tail_taxonomy.py"

# ⛔ ДВА МЁРТВЫХ ФАЙЛА, у которых своя копия порога так и осталась. Проверено грепом 19.08:
# их не импортирует и не зовёт НИКТО в репозитории. `page_builder.py` — тупиковая ветвь,
# закрытая каноном навсегда (ни разу не бежала); `questions_page.py` — генератор снятого
# контура `/q/`, его вызов убран из `runner.py` 19.08. Сносить их — отдельное решение юзера,
# а не побочный эффект этой правки; пока просто не считаем их читателями порога.
DEAD = {"page_builder.py", "questions_page.py"}

# Читатели порога. Дубли из `combine/builder` не берём — их держит дословными
# `combine/test_image_has_render.py::test_duplicates_identical`.
FILES = (
    [
        p
        for p in sorted(HERE.glob("*.py"))
        if not p.name.startswith("test_") and p.name not in DEAD
    ]
    + [PSEO / "combine" / "bot.py"]
    + [PSEO / "render.py"]
)

# Сравнение размера с литералом 4: `len(...) >= 4`, `< 4`, `> 4`. `4\b` не ловит 400/4096.
LITERAL = re.compile(r"len\([^()]*(?:\([^()]*\)[^()]*)*\)\s*[<>]=?\s*4\b")
SECOND_DEF = re.compile(r"^\s*PAGE_MIN\s*=\s*\d", re.M)


def test_owner_holds_a_number():
    assert isinstance(tax.PAGE_MIN, int) and tax.PAGE_MIN > 0, tax.PAGE_MIN


def test_no_second_definition():
    """Второе определение = вторая правда. Псевдоним `PAGE_MIN = tax.PAGE_MIN` тоже запрещён:
    он копирует значение НА МОМЕНТ ИМПОРТА, и подмена владельца до него не доходит."""
    bad = []
    for p in FILES:
        if p == OWNER or not p.exists():
            continue
        for m in SECOND_DEF.finditer(p.read_text(encoding="utf-8")):
            bad.append(f"{p.name}: {m.group(0).strip()}")
    assert not bad, "порог определён не только у владельца: " + "; ".join(bad)


def test_no_literal_threshold_in_readers():
    """Литерал `len(...) >= 4` — это и есть та копия, через которую уходили абзацы."""
    bad = []
    for p in FILES:
        if p == OWNER or not p.exists():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if LITERAL.search(code):
                bad.append(f"{p.name}:{n}: {line.strip()}")
    assert not bad, "порог страницы литералом вне владельца: " + "; ".join(bad)


def test_readers_actually_ask_the_owner():
    """⛔ Обратная сторона двух проверок выше: файл мог просто ПЕРЕСТАТЬ решать про страницу
    (условие удалили) — тогда литералов нет, а щель шире прежней. Требуем, чтобы каждый
    читатель звал владельца по имени."""
    silent = []
    for p in FILES:
        if p == OWNER or not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        # ⛔ Спрашивать владельца обязан тот, кто РЕШАЕТ, а не тот, кто считает. Признак
        # решения — сравнение размера (`len(...) >= …`). Пульт после перехода на тракт §0.19
        # только считает готовое, и требовать от него порог было бы придиркой.
        if not re.search(r"len\([^)]*(?:items|groups)[^)]*\)\s*[<>]=?", txt):
            continue
        if not re.search(r"\b(?:tax|_tax)\.PAGE_MIN\b", txt):
            silent.append(p.name)
    assert not silent, "решает про страницы, но владельца не спрашивает: " + ", ".join(
        silent
    )
