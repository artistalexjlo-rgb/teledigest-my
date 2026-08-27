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
    assert [len(c) for c in tract.split(list(range(38)), 15)] == [15, 15, 8]
    assert [len(c) for c in tract.split(list(range(15)), 15)] == [15]
    assert [len(c) for c in tract.split(list(range(45)), 15)] == [15, 15, 15]
    assert [len(c) for c in tract.split(list(range(4)), 15)] == [4]


def _geo(tmp_path, monkeypatch, n, kanon="visa documents"):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"id": f"h{i}", "theme": "visa", "subtheme": "label", "n": 1, "name": kanon}
        for i in range(n)
    ]
    os.makedirs(tmp_path / "tests" / "tags", exist_ok=True)
    with open(tmp_path / "tests" / "tags" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    pairs = [(r["id"], f"advice {j}") for j, r in enumerate(rows)]
    monkeypatch.setattr(tract, "load_flies", lambda geo: list(pairs))
    tract.build_corpus("gr")
    return json.load(
        open(tmp_path / "tests" / "out_facet" / "gr.json", encoding="utf-8")
    )


def test_thick_name_becomes_several_pages(tmp_path, monkeypatch):
    """38 советов под одним именем → три страницы, а не простыня. Адреса разные."""
    out = _geo(tmp_path, monkeypatch, 38)
    views = out["views_by_task"]
    assert [len(v["items"]) for v in views] == [15, 15, 8], views
    assert [v["slug"] for v in views] == [
        "visa-documents",
        "visa-documents-2",
        "visa-documents-3",
    ]
    assert all(len(v["items"]) <= tract.PAGE_MAX for v in views)


def test_tail_stays_in_its_own_branch(tmp_path, monkeypatch):
    """16 → 15 + 1. Хвост ветки НЕ сбрасывается: «не будем мы бегать и прибираться».

    Слова юзера 27.08. У этих советов имя уже есть, и сброс в безымянный остаток терял бы
    принадлежность. `PAGE_MIN` решает другое — заводить ли ветку с нуля.
    """
    out = _geo(tmp_path, monkeypatch, 16)
    assert [len(v["items"]) for v in out["views_by_task"]] == [15, 1]
    assert not out["shelves"], "хвост ветки уехал в остаток"
    assert tax.PAGE_MIN == 4  # порог входа в ветку, к делению отношения не имеет


def test_single_page_keeps_its_plain_name(tmp_path, monkeypatch):
    """Пачка влезла целиком — имя и адрес без суффиксов и без «(1)»."""
    out = _geo(tmp_path, monkeypatch, 12)
    v = out["views_by_task"][0]
    assert v["title"] == "visa documents" and v["slug"] == "visa-documents"


def test_parts_share_one_clean_name_and_carry_their_number(tmp_path, monkeypatch):
    """Номер части живёт ПОЛЕМ, а не в имени: имя переводится, «(2)» уехало бы в 14 языков."""
    out = _geo(tmp_path, monkeypatch, 38)
    views = out["views_by_task"]
    assert {v["title"] for v in views} == {"visa documents"}
    assert [(v["branch"], v["part"], v["parts"]) for v in views] == [
        ("visa-documents", 1, 3),
        ("visa-documents", 2, 3),
        ("visa-documents", 3, 3),
    ]


def _geo_mix(tmp_path, monkeypatch, bez_imeni, s_imenem=0, tema="visa"):
    """Гео из советов с именем и без: без имени — это и есть остаток темы."""
    monkeypatch.chdir(tmp_path)
    rows = [
        {
            "id": f"y{i}",
            "theme": tema,
            "subtheme": "label",
            "n": 1,
            "name": "visa documents",
        }
        for i in range(s_imenem)
    ]
    rows += [
        {"id": f"n{i}", "theme": tema, "subtheme": "label", "n": 1}
        for i in range(bez_imeni)
    ]
    os.makedirs(tmp_path / "tests" / "tags", exist_ok=True)
    with open(tmp_path / "tests" / "tags" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    pairs = [(r["id"], f"advice {j}") for j, r in enumerate(rows)]
    monkeypatch.setattr(tract, "load_flies", lambda geo: list(pairs))
    tract.build_corpus("gr")
    return json.load(
        open(tmp_path / "tests" / "out_facet" / "gr.json", encoding="utf-8")
    )


def test_remainder_is_a_branch_like_any_other(tmp_path, monkeypatch):
    """Остаток — ТАКАЯ ЖЕ ветка: одно имя, части внутри, хвост при себе.

    33 совета → 15/15/3. Третья часть короче `PAGE_MIN` и всё равно остаётся: правило
    деления одно на всё, отдельного для остатка нет.
    """
    out = _geo_mix(tmp_path, monkeypatch, bez_imeni=33)
    misc = [v for v in out["views_by_task"] if v.get("branch") == tract.MISC_SLUG]
    assert [len(v["items"]) for v in misc] == [15, 15, 3], misc
    assert [v["slug"] for v in misc] == ["misc", "misc-2", "misc-3"]
    assert {v["title"] for v in misc} == {"Other"}, "имя ветки должно быть английским"
    assert not out["shelves"], "остаток остался кучей на теме"


def test_small_remainder_stays_on_the_theme(tmp_path, monkeypatch):
    """Меньше порога страницы — остаётся списком на теме, страницей не становится."""
    out = _geo_mix(tmp_path, monkeypatch, bez_imeni=2)
    assert not [
        v for v in out["views_by_task"] if v["slug"].startswith(tract.MISC_SLUG)
    ]
    assert sum(len(sh["items"]) for sh in out["shelves"]) == 2


def test_nothing_is_lost_between_pages_and_the_theme(tmp_path, monkeypatch):
    """Счёт сходится: советы либо на страницах, либо на теме, и ни один не пропал."""
    out = _geo_mix(tmp_path, monkeypatch, bez_imeni=34, s_imenem=20)
    na_stranicah = sum(len(v["items"]) for v in out["views_by_task"])
    na_teme = sum(len(sh["items"]) for sh in out["shelves"])
    assert na_stranicah + na_teme == 54, (na_stranicah, na_teme)
