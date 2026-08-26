# -*- coding: utf-8 -*-
"""Сторож сборщика сайта: дерево ровно такое, как записано в плане.

Четыре уровня и ни одного следа отменённой схемы:

    /<язык>/                                    главная
      /<язык>/<страна>/                         хаб: плитки тем
        /<язык>/<страна>/<тема>/                тема: кнопки страниц + остаток
          /<язык>/<страна>/<тема>/<страница>/   страница: советы

Сторожим то, что РЕАЛЬНО можно нарушить правкой:

- адрес берётся из корпуса, а не считается из заголовка (иначе он свой в каждом языке);
- остаток темы не теряется при группировке;
- нет адреса — нет страницы (иначе страницы затирают друг друга).

⛔ Проверки «нет мостика `/s/`» тут не будет: путь собирается одной строкой, взяться ему
неоткуда. Тест, который не может упасть, — это не сторож, а украшение.
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location(
    "sitebuild", os.path.join(HERE, "site.py")
)
site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site)


def _korpus(tmp_path, monkeypatch, views, shelves=()):
    built = tmp_path / "built"
    os.makedirs(built / "out_facet", exist_ok=True)
    with open(built / "out_facet" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"geo": "gr", "views_by_task": list(views), "shelves": list(shelves)},
            fh,
            ensure_ascii=False,
        )
    monkeypatch.setattr(site, "BUILT", str(built))
    monkeypatch.setattr(site, "DATA", str(tmp_path / "data"))


def _view(zadacha, adres, tema="visa", n=5):
    return {
        "zadacha": zadacha,
        "tema": tema,
        "shelf": "Визовые процедуры",
        "adres": adres,
        "items": [
            {"id": f"h{i}", "text": f"advice {i}. tail", "n": 1} for i in range(n)
        ],
    }


def _pages(tmp_path):
    out = {}
    if not os.path.isdir(tmp_path / "data"):
        return out  # ничего не собралось — каталога и нет
    for name in os.listdir(tmp_path / "data"):
        page = json.load(open(tmp_path / "data" / name, encoding="utf-8"))
        out[page["path"]] = page
    return out


def test_tree_has_four_levels(tmp_path, monkeypatch):
    """Главная, хаб, тема, страница — и адрес страницы содержит СЕГМЕНТ ТЕМЫ."""
    _korpus(tmp_path, monkeypatch, [_view("visa documents", "visa-documents")])
    site.sobrat_vse("ru")
    paths = set(_pages(tmp_path))
    assert "/ru/" in paths
    assert "/ru/gr/" in paths
    assert "/ru/gr/visa/" in paths
    assert "/ru/gr/visa/visa-documents/" in paths


def test_address_comes_from_the_corpus_not_from_the_title(tmp_path, monkeypatch):
    """Адрес берётся из поля корпуса. Заголовок русский — адрес всё равно английский."""
    v = _view("сроки рассмотрения визы", "visa-processing-time")
    _korpus(tmp_path, monkeypatch, [v])
    site.sobrat_vse("ru")
    page = _pages(tmp_path)["/ru/gr/visa/visa-processing-time/"]
    assert page["h1"] == "сроки рассмотрения визы"
    assert (
        page["shared_tail"] is True
    ), "хвост общий — он из справочника, а не из заголовка"


def test_theme_page_lists_pages_and_keeps_the_remainder(tmp_path, monkeypatch):
    """Тема: кнопки страниц сверху, остаток списком ниже — советы не пропадают."""
    _korpus(
        tmp_path,
        monkeypatch,
        [_view("visa documents", "visa-documents"), _view("visa fees", "visa-fees")],
        shelves=[
            {
                "shelf": "Визовые процедуры",
                "items": [{"id": "x1", "text": "leftover advice. tail", "n": 1}],
            }
        ],
    )
    site.sobrat_vse("ru")
    tema = _pages(tmp_path)["/ru/gr/visa/"]
    assert [t["url"] for t in tema["tiles"]] == [
        "/ru/gr/visa/visa-documents/",
        "/ru/gr/visa/visa-fees/",
    ]
    assert len(tema["faqs"]) == 1, "остаток темы потерян"


def test_page_without_address_is_skipped(tmp_path, monkeypatch):
    """Нет адреса — нет страницы. Выдумывать хвост нельзя: страницы затрут друг друга."""
    v = _view("visa documents", "")
    _korpus(tmp_path, monkeypatch, [v])
    site.sobrat_vse("ru")
    assert not [p for p in _pages(tmp_path) if p.startswith("/ru/gr/visa/")]
