# -*- coding: utf-8 -*-
"""Сторож шага схлопывания почти-копий (тракт, шаг 2).

Три вещи, ради которых шаг вообще есть:
  1. быстрый путь (компоненты + average-link внутри) даёт РОВНО то же, что честный перебор —
     иначе «ускорение» тихо меняло бы результат;
  2. шаг пишет ПРОТОКОЛ с текстами проглоченных, а не счётчик «схлопнуто N» — на счётчике
     порог 0.86 и прокололся: за числом пряталось съеденное содержимое;
  3. разметка после схлопывания берёт ТОЛЬКО представителей — иначе платим рту за каждую
     почти-копию, и смысл шага теряется.
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tract  # noqa: E402
import vectors as dedup  # noqa: E402


def _naive(ids, vv, thr):
    """Честный перебор без разбиения на компоненты — эталон для быстрого пути."""
    have = [i for i in ids if i in vv]
    m = np.stack([vv[i] for i in have])
    out = [[have[i] for i in c] for c in dedup.avg_link(m @ m.T, thr)]
    return sorted(sorted(g) for g in out)


def _vecs(rows):
    """rows: id → вектор-заготовка; нормируем, как это делает свипер."""
    return {
        i: np.array(v, dtype=np.float32) / np.linalg.norm(v) for i, v in rows.items()
    }


def test_fast_path_equals_honest_bruteforce():
    """Компоненты — не приближение: средняя связь не выше максимальной, поэтому группы без
    ребра ≥ порога слиться не могут. Проверяем совпадением с эталоном."""
    rng = np.random.default_rng(7)
    vv = {}
    for c in range(6):  # шесть кучек по четыре близких вектора + шум
        base = rng.normal(size=32)
        for k in range(4):
            v = base + 0.05 * rng.normal(size=32)
            vv[f"{c}-{k}"] = (v / np.linalg.norm(v)).astype(np.float32)
    ids = list(vv)
    for thr in (0.80, 0.93):
        fast = sorted(sorted(g) for g in dedup.groups_all(ids, vv, thr))
        assert fast == _naive(ids, vv, thr), thr


def test_flies_without_vector_are_kept_alone():
    """Муху без вектора не судим и не теряем — она идёт своей группой."""
    vv = _vecs({"a": [1.0, 0.0], "b": [1.0, 0.001]})
    groups = dedup.groups_all(["a", "b", "нет-вектора"], vv, 0.93)
    assert ["нет-вектора"] in groups
    assert sorted(len(g) for g in groups) == [1, 2]


def test_undone_is_the_single_rule(tmp_path):
    """`undone` — единственное место, где решается «кого возьмёт разметка».

    Проглоченные схлопыванием не берутся никогда, уже размеченные — не берутся повторно,
    а корень данных передаётся явно: рот бежит с cwd=BRAIN, пульт зовёт из своего процесса.
    """
    os.makedirs(tmp_path / "tests" / "dedup")
    os.makedirs(tmp_path / "tests" / "tags")
    with open(tmp_path / "tests" / "dedup" / "xx.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"groups": [{"rep": "a", "ids": ["a", "b"]}, {"rep": "c", "ids": ["c"]}]},
            fh,
        )
    ids = ["a", "b", "c"]
    assert tract.undone("xx", ids, base=str(tmp_path)) == ["a", "c"]

    with open(tmp_path / "tests" / "tags" / "xx.json", "w", encoding="utf-8") as fh:
        json.dump([{"id": "a"}], fh)
    assert tract.undone("xx", ids, base=str(tmp_path)) == ["c"]

    # без схлопывания берутся все неразмеченные
    assert tract.undone("yy", ids, base=str(tmp_path)) == ids


@pytest.fixture
def geo_probe(tmp_path, monkeypatch):
    """Гео из четырёх мух: две — почти-копии, две — разные. Вектора подставляем."""
    flies = [
        ("f1", "Права нужно менять в первый год, иначе штраф."),
        ("f2", "Права меняют в течение первого года, потом штрафуют."),
        ("f3", "Мусор выносят по средам, бак у ворот."),
        ("f4", "Интернет подключают за три дня, нужен паспорт."),
    ]
    vv = _vecs(
        {
            "f1": [1.0, 0.0, 0.0],
            "f2": [0.999, 0.045, 0.0],
            "f3": [0.0, 1.0, 0.0],
            "f4": [0.0, 0.0, 1.0],
        }
    )
    monkeypatch.setattr(tract, "load_flies", lambda geo: list(flies))
    monkeypatch.setattr(dedup, "load_vecs", lambda ids: dict(vv))
    monkeypatch.chdir(tmp_path)
    return flies


def test_protocol_shows_what_was_swallowed(geo_probe, capsys):
    """В протоколе — текст представителя и ПОЛНЫЙ текст проглоченного. Судить порог по
    текстам, а не по числу."""
    tract.collapse("xx")
    proto = open(os.path.join("tests", "dedup", "xx.txt"), encoding="utf-8").read()
    assert "Права меняют в течение первого года" in proto
    assert "Права нужно менять в первый год" in proto
    assert "СХЛОПНУТА" in proto and "ОСТАЁТСЯ" in proto
    # непохожие мухи в протокол склеек не попадают
    assert "Мусор выносят" not in proto
    out = capsys.readouterr().out
    assert "склеек 1" in out and "схлопнуто 1" in out


def test_mark_takes_only_representatives(geo_probe, monkeypatch):
    """После схлопывания рту достаются представители, и за каждым несётся число мух."""
    tract.collapse("xx")
    sg = json.load(open(os.path.join("tests", "dedup", "xx.json"), encoding="utf-8"))
    reps = {g["rep"] for g in sg["groups"]}
    assert len(reps) == 3, sg["groups"]

    seen = {}

    def fake_call(user, sysprompt, **kw):
        idx = json.loads(user)
        seen.update(idx)
        return {
            "rows": [
                {"i": k, "perevod": v, "theme": "docs", "subtheme": "проба"}
                for k, v in idx.items()
            ]
        }

    monkeypatch.setattr(tract, "call", fake_call)
    tract.mark("xx")
    assert len(seen) == 3, seen  # четвёртая муха — почти-копия, рту не показана
    done = json.load(open(os.path.join("tests", "tags", "xx.json"), encoding="utf-8"))
    assert sorted(r["n"] for r in done) == [1, 1, 2], done
