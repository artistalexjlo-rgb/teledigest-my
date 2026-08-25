# -*- coding: utf-8 -*-
"""Сторож деления: страница не бывает толще порога, хвост-обрезок уходит в остаток.

Правило юзера: «все пачки свыше 15 делятся на эту пропорцию, чтоб не было простыней».
24.08 я снял верхний порог своим прочтением архива, и имя на 38 советов собиралось одной
простынёй — этот сторож ловит возврат к такому поведению.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tail_taxonomy as tax  # noqa: E402
import tract  # noqa: E402


def test_pack_is_cut_by_the_threshold():
    """Режем по порогу подряд, последняя часть неполная: 38 → 15/15/8."""
    assert [len(c) for c in tract.podeli(list(range(38)), 15)] == [15, 15, 8]
    assert [len(c) for c in tract.podeli(list(range(15)), 15)] == [15]
    assert [len(c) for c in tract.podeli(list(range(45)), 15)] == [15, 15, 15]
    assert [len(c) for c in tract.podeli(list(range(4)), 15)] == [4]


def _geo(tmp_path, monkeypatch, n, kanon="visa documents"):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"id": f"h{i}", "tema": "visa", "podtema": "label", "n": 1, "kanon": kanon}
        for i in range(n)
    ]
    os.makedirs(tmp_path / "tests" / "tags", exist_ok=True)
    with open(tmp_path / "tests" / "tags" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    pary = [(r["id"], f"advice {j}") for j, r in enumerate(rows)]
    monkeypatch.setattr(tract, "load_flies", lambda geo: list(pary))
    tract.sborka("gr")
    return json.load(
        open(tmp_path / "tests" / "out_facet" / "gr.json", encoding="utf-8")
    )


def test_thick_name_becomes_several_pages(tmp_path, monkeypatch):
    """38 советов под одним именем → три страницы, а не простыня. Адреса разные."""
    out = _geo(tmp_path, monkeypatch, 38)
    views = out["views_by_task"]
    assert [len(v["items"]) for v in views] == [15, 15, 8], views
    assert [v["adres"] for v in views] == [
        "visa-documents",
        "visa-documents-2",
        "visa-documents-3",
    ]
    assert all(len(v["items"]) <= tract.PAGE_MAX for v in views)


def test_tail_shorter_than_page_min_goes_to_the_remainder(tmp_path, monkeypatch):
    """16 → 15 и хвост в 1 пункт: хвост страницей не становится, он в остатке."""
    out = _geo(tmp_path, monkeypatch, 16)
    assert [len(v["items"]) for v in out["views_by_task"]] == [15]
    assert sum(len(sh["items"]) for sh in out["shelves"]) == 1
    assert tax.PAGE_MIN == 4  # порог, по которому хвост признан обрезком


def test_single_page_keeps_its_plain_name(tmp_path, monkeypatch):
    """Пачка влезла целиком — имя и адрес без суффиксов и без «(1)»."""
    out = _geo(tmp_path, monkeypatch, 12)
    v = out["views_by_task"][0]
    assert v["zadacha"] == "visa documents" and v["adres"] == "visa-documents"
