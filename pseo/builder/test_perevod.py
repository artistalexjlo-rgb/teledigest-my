# -*- coding: utf-8 -*-
"""Сторож звена 6: переводится ВСЁ видимое, платим только за новое, адреса не двигаются.

Четыре обещания, ради которых переводчик переписан заново (PLAN.md, звено 6):

1. **русский — обычный язык перевода**. Старый модуль переводил С РУССКОГО, и русская
   версия была источником; теперь источник английский, а `ru` идёт как прочие тринадцать;
2. **платим за новое и за переписанное**, остальное берётся из готового файла — иначе
   каждый прогон покупал бы весь корпус заново;
3. **адрес, ветка и части НЕ трогаются переводом** — иначе переключатель языка уведёт на
   другую страницу, а хвост адреса перестанет быть общим;
4. **мелочь остатка тоже переводится**: она показывается абзацами на странице темы, и
   английская вставка посреди русской страницы — это брак.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import perevod  # noqa: E402


def _korpus(tmp_path, monkeypatch, views, shelves=()):
    monkeypatch.setattr(perevod, "BUILT", str(tmp_path))
    monkeypatch.setattr(perevod, "TEMY_FILE", str(tmp_path / "temy.json"))
    os.makedirs(tmp_path / "out_facet", exist_ok=True)
    with open(tmp_path / "out_facet" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"geo": "gr", "views_by_task": list(views), "shelves": list(shelves)},
            fh,
            ensure_ascii=False,
        )


def _view(zadacha, adres, texts, tema="visa", branch=None, part=1, parts=1):
    return {
        "zadacha": zadacha,
        "tema": tema,
        "shelf": "Визовые процедуры",
        "adres": adres,
        "branch": branch or adres,
        "part": part,
        "parts": parts,
        "items": [
            {"id": f"{adres}-{i}", "text": t, "n": 1} for i, t in enumerate(texts)
        ],
    }


def _rot(monkeypatch, schet=None):
    """Рот-заглушка: возвращает «<lang>:<текст>». Считает, сколько текстов купили."""

    def fake_call(user, sysprompt, **kw):
        payload = json.loads(user)
        if schet is not None:
            schet.setdefault(kw["consumer"], []).extend(payload.values())
        metka = "ru" if "Russian" in sysprompt else "xx"
        return {k: f"{metka}:{v}" for k, v in payload.items()}

    monkeypatch.setattr(perevod, "call", fake_call)


def _out(tmp_path, lang="ru"):
    with open(tmp_path / f"out_facet_{lang}" / "gr.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_russian_is_an_ordinary_target_language(tmp_path, monkeypatch):
    """`ru` переводится ротом, как и прочие: отдельного пути к нему нет."""
    _korpus(
        tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a", "b"])]
    )
    schet = {}
    _rot(monkeypatch, schet)
    perevod.perevedi("gr", "ru")
    out = _out(tmp_path)
    assert [it["text"] for it in out["views_by_task"][0]["items"]] == ["ru:a", "ru:b"]
    assert out["views_by_task"][0]["zadacha"] == "ru:visa documents"
    assert sorted(schet) == ["labels", "translate"]


def test_english_costs_nothing(tmp_path, monkeypatch):
    """Английский — сам источник: копия без единого вызова рта."""
    _korpus(tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a"])])
    schet = {}
    _rot(monkeypatch, schet)
    perevod.perevedi("gr", "en")
    assert schet == {}, "за английский заплатили"
    assert _out(tmp_path, "en")["views_by_task"][0]["items"][0]["text"] == "a"


def test_second_run_buys_only_what_changed(tmp_path, monkeypatch):
    """Второй прогон покупает ТОЛЬКО переписанный текст и новое имя, остальное готово."""
    _korpus(
        tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a", "b"])]
    )
    _rot(monkeypatch)
    perevod.perevedi("gr", "ru")

    # источник переписали в одном совете, имя ветки поменяли
    _korpus(
        tmp_path, monkeypatch, [_view("visa papers", "visa-documents", ["a", "B-2"])]
    )
    schet = {}
    _rot(monkeypatch, schet)
    perevod.perevedi("gr", "ru")
    assert schet["translate"] == ["B-2"], schet
    assert schet["labels"] == ["visa papers"], schet
    out = _out(tmp_path)
    assert [it["text"] for it in out["views_by_task"][0]["items"]] == ["ru:a", "ru:B-2"]


def test_branch_name_is_bought_once_for_all_its_parts(tmp_path, monkeypatch):
    """Имя одно на ветку: три части не значат три покупки заголовка."""
    _korpus(
        tmp_path,
        monkeypatch,
        [
            _view(
                "visa documents",
                "visa-documents",
                ["a"],
                branch="visa-documents",
                part=1,
                parts=2,
            ),
            _view(
                "visa documents",
                "visa-documents-2",
                ["b"],
                branch="visa-documents",
                part=2,
                parts=2,
            ),
        ],
    )
    schet = {}
    _rot(monkeypatch, schet)
    perevod.perevedi("gr", "ru")
    assert schet["labels"] == ["visa documents"], schet


def test_addresses_and_branches_survive_translation(tmp_path, monkeypatch):
    """Перевод меняет ТЕКСТ, а не адрес: иначе переключатель языка уведёт не туда."""
    _korpus(
        tmp_path,
        monkeypatch,
        [
            _view(
                "visa documents",
                "visa-documents",
                ["a"],
                branch="visa-documents",
                part=1,
                parts=2,
            ),
            _view(
                "visa documents",
                "visa-documents-2",
                ["b"],
                branch="visa-documents",
                part=2,
                parts=2,
            ),
        ],
    )
    _rot(monkeypatch)
    perevod.perevedi("gr", "ru")
    out = _out(tmp_path)
    assert [v["adres"] for v in out["views_by_task"]] == [
        "visa-documents",
        "visa-documents-2",
    ]
    assert {v["branch"] for v in out["views_by_task"]} == {"visa-documents"}
    assert [v["part"] for v in out["views_by_task"]] == [1, 2]


def test_the_small_remainder_is_translated_too(tmp_path, monkeypatch):
    """Мелочь остатка видна читателю абзацами — значит переводится вместе со всем."""
    _korpus(
        tmp_path,
        monkeypatch,
        [_view("visa documents", "visa-documents", ["a"])],
        shelves=[
            {
                "shelf": "Визовые процедуры",
                "items": [{"id": "x1", "text": "leftover", "n": 1}],
            }
        ],
    )
    _rot(monkeypatch)
    perevod.perevedi("gr", "ru")
    out = _out(tmp_path)
    assert out["shelves"][0]["items"][0]["text"] == "ru:leftover"
