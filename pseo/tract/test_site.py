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


def _view(zadacha, adres, tema="visa", n=5, branch=None, part=1, parts=1):
    return {
        "title": zadacha,
        "theme": tema,
        "slug": adres,
        "branch": branch or adres,
        "part": part,
        "parts": parts,
        "items": [
            {"id": f"h{i}", "text": f"advice {i}. tail", "n": 1} for i in range(n)
        ],
    }


def _vetka(zadacha, base, chastey, tema="visa"):
    """Ветка из нескольких частей — так её отдаёт звено 5."""
    return [
        _view(
            zadacha,
            base if k == 1 else f"{base}-{k}",
            tema=tema,
            branch=base,
            part=k,
            parts=chastey,
        )
        for k in range(1, chastey + 1)
    ]


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
    site.build_all("ru")
    paths = set(_pages(tmp_path))
    assert "/ru/" in paths
    assert "/ru/gr/" in paths
    assert "/ru/gr/visa/" in paths
    assert "/ru/gr/visa/visa-documents/" in paths


def test_address_comes_from_the_corpus_not_from_the_title(tmp_path, monkeypatch):
    """Адрес берётся из поля корпуса. Заголовок русский — адрес всё равно английский."""
    v = _view("сроки рассмотрения визы", "visa-processing-time")
    _korpus(tmp_path, monkeypatch, [v])
    site.build_all("ru")
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
                "theme": "visa",
                "items": [{"id": "x1", "text": "leftover advice. tail", "n": 1}],
            }
        ],
    )
    site.build_all("ru")
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
    site.build_all("ru")
    assert not [p for p in _pages(tmp_path) if p.startswith("/ru/gr/visa/")]


def test_branch_is_one_tile_on_the_theme(tmp_path, monkeypatch):
    """Ветка из трёх частей выходит на тему ОДНОЙ плиткой (PLAN.md, 27.08).

    Соседних «имя» и «имя (2)» не бывает: части — внутреннее устройство ветки, а не
    соседи по витрине. Подпись плитки — сумма по всем частям.
    """
    _korpus(tmp_path, monkeypatch, _vetka("visa documents", "visa-documents", 3))
    site.build_all("ru")
    tema = _pages(tmp_path)["/ru/gr/visa/"]
    assert [t["url"] for t in tema["tiles"]] == ["/ru/gr/visa/visa-documents/"]
    assert tema["tiles"][0]["blurb"] == "15", "подпись плитки — сумма всех частей"
    assert [t["title"] for t in tema["tiles"]] == ["visa documents"]


def test_parts_link_to_each_other(tmp_path, monkeypatch):
    """Во вторую часть можно попасть только из первой — значит части знают сестёр."""
    _korpus(tmp_path, monkeypatch, _vetka("visa documents", "visa-documents", 3))
    site.build_all("ru")
    pages = _pages(tmp_path)
    p2 = pages["/ru/gr/visa/visa-documents-2/"]
    assert [c["url"] for c in p2["parts"]] == [
        "/ru/gr/visa/visa-documents/",
        "/ru/gr/visa/visa-documents-2/",
        "/ru/gr/visa/visa-documents-3/",
    ]
    assert [c["current"] for c in p2["parts"]] == [False, True, False]
    assert p2["h1"] == "visa documents", "номер части в имя не лезет"
    assert (
        "2/3" in p2["title"]
    ), "а в заголовок окна — лезет, страницы должны различаться"


def test_single_page_has_no_part_nav(tmp_path, monkeypatch):
    """Ветка в одну страницу переходов не показывает — нечего перелистывать."""
    _korpus(tmp_path, monkeypatch, [_view("visa fees", "visa-fees")])
    site.build_all("ru")
    assert _pages(tmp_path)["/ru/gr/visa/visa-fees/"]["parts"] == []


def test_service_pages_exist_and_carry_no_content(tmp_path, monkeypatch):
    """Шлюз и поиск пишутся всегда, даже без единого гео, и в карту сайта не идут."""
    _korpus(tmp_path, monkeypatch, [])
    site.build_all("ru")
    pages = _pages(tmp_path)
    assert pages["/ru/go/luky/"]["template"] == "go.html.j2"
    assert pages["/ru/go/luky/"]["noindex"] is True
    assert pages["/ru/find/"]["template"] == "find.html.j2"
    assert pages["/ru/find/"]["noindex"] is True


def test_about_page_carries_real_text_and_is_indexable(tmp_path, monkeypatch):
    """«О проекте» несёт настоящий текст (не пустышку) и, в отличие от шлюза и поиска,
    ИНДЕКСИРУЕТСЯ — это содержание, а не служебная переадресация."""
    _korpus(tmp_path, monkeypatch, [])
    site.build_all("ru")
    about = _pages(tmp_path)["/ru/about/"]
    assert about["noindex"] is False
    assert about["h1"] and about["body"], "страница пустая — текста нет"
    assert "href='#luky'" in about["body"], "маркер двери в продукт потерян"


def test_about_page_is_skipped_when_language_has_no_text(tmp_path, monkeypatch):
    """Нет текста на языке — нет страницы. Не показываем половину заглушкой."""
    _korpus(tmp_path, monkeypatch, [])
    empty_about = tmp_path / "no_about.json"
    empty_about.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(site, "ABOUT_FILE", str(empty_about))
    site.build_all("ru")
    assert "/ru/about/" not in _pages(tmp_path)
