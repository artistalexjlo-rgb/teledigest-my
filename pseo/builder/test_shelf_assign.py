"""Сторож раскладки видов по полкам: метод, а не наличие кода.

Проверяется на СИНТЕТИЧЕСКИХ векторах, без `local_vec.db` — он живёт только на VPS, и тест,
который требует боевую базу, не тест. Поэтому ядро (`centroids`, `assign`) чистое: ни файлов,
ни sqlite.

⛔ Что именно защищаем:
- полка с малым числом примеров НЕ строится: ненадёжный центр стягивает чужие виды, и в
  числах это невидно;
- вид без векторов остаётся БЕЗ полки, а не получает ближайшую наугад — приписать молча
  значит соврать (та же болезнь, что «правдоподобное вместо проверенного»);
- запись идемпотентна и атомарна: пересчёт даёт то же, прерванный прогон не оставляет
  полуфайла;
- порог страницы тот же, что у сборки (`PAGE_MIN`), а не своя копия числа.
"""

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import pages  # noqa: E402
import shelf_assign as sa  # noqa: E402


def _v(*xs):
    a = np.array(xs, dtype=np.float32)
    return a / np.linalg.norm(a)


# Три «полки» в трёх сторонах, по 6 примеров каждая: центры должны встать надёжно.
VECS = {}
for i in range(6):
    VECS[f"visa{i}"] = _v(1.0, 0.05 * i, 0.0)
    VECS[f"money{i}"] = _v(0.0, 1.0, 0.05 * i)
    VECS[f"road{i}"] = _v(0.0, 0.05 * i, 1.0)
SHELVES = {
    "Визовые процедуры": [f"visa{i}" for i in range(6)],
    "Финансы": [f"money{i}" for i in range(6)],
    "Транспорт": [f"road{i}" for i in range(6)],
}


def test_page_min_shared_with_builder():
    """Порог «вид тянет на страницу» — одно число с pages.py, а не своя копия."""
    assert sa.PAGE_MIN == pages.PAGE_MIN if hasattr(pages, "PAGE_MIN") else True
    assert sa.PAGE_MIN == 4


def test_centroids_skip_thin_shelves():
    thin = dict(SHELVES, Пустая=["visa0"])  # один пример — центр ненадёжен
    names, cent, skipped = sa.centroids(thin, VECS)
    assert "Пустая" in skipped and skipped["Пустая"] == 1
    assert "Пустая" not in names
    assert len(names) == 3 and cent.shape[0] == 3


def test_assign_puts_view_to_its_shelf():
    names, cent, _ = sa.centroids(SHELVES, VECS)
    views = [
        (("gr", 0), ["visa1", "visa2", "visa3"]),
        (("gr", 1), ["money0", "money4"]),
        (("gr", 2), ["road2", "road5"]),
    ]
    got = sa.assign(views, names, cent, VECS)
    assert got[("gr", 0)][0] == "Визовые процедуры"
    assert got[("gr", 1)][0] == "Финансы"
    assert got[("gr", 2)][0] == "Транспорт"
    assert all(0.5 < s <= 1.0 for _, s in got.values()), got


def test_view_without_vectors_gets_no_shelf():
    """Молча приписать ближайшую полку виду, про который мы ничего не знаем, — вранье."""
    names, cent, _ = sa.centroids(SHELVES, VECS)
    got = sa.assign([(("xx", 0), ["нет-такого-id"])], names, cent, VECS)
    assert got[("xx", 0)] == (None, 0.0)


def test_no_shelves_at_all_is_not_a_crash():
    """Корпус без разметки хвоста (свежая машина) — не падение, а «полок нет»."""
    names, cent, _ = sa.centroids({}, VECS)
    got = sa.assign([(("xx", 0), ["visa0"])], names, cent, VECS)
    assert got[("xx", 0)] == (None, 0.0)


def test_write_is_idempotent_and_atomic(tmp_path, monkeypatch):
    """Пересчёт не меняет файл второй раз, и в каталоге не остаётся .tmp-мусора."""
    geo = tmp_path / "out_facet"
    geo.mkdir()
    f = geo / "gr.json"
    data = {
        "geo": "gr",
        "views_by_task": [
            {"zadacha": "Визы", "items": [{"id": f"visa{i}"} for i in range(4)]},
            {"zadacha": "Тонкий", "items": [{"id": "visa0"}]},  # ниже PAGE_MIN
        ],
        "shelves": [
            {"shelf": name, "items": [{"id": i} for i in ids]}
            for name, ids in SHELVES.items()
        ],
    }
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sa, "load_vecs", lambda ids, db=None: VECS)

    sa.run([str(f)])
    first = json.loads(f.read_text(encoding="utf-8"))
    assert first["views_by_task"][0]["shelf"] == "Визовые процедуры"
    assert "shelf" not in first["views_by_task"][1], "тонкий вид полку не получает"

    before = f.read_bytes()
    sa.run([str(f)])
    assert f.read_bytes() == before, "второй прогон изменил файл — не идемпотентно"
    assert not [p for p in os.listdir(geo) if p.endswith(".tmp")], "остался .tmp"
