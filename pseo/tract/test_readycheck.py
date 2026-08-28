# -*- coding: utf-8 -*-
"""Сторож гейта: проверяет ВСЕ языки, не только русский.

Написан ДО реализации (юзер 28.08: комментарий не проверка, инерция залезает и в него —
проверять должен тест, который нельзя тихо подогнать прозой).

⛔ Старый `readycheck.py` (снят в legacy 28.08) обходил только `out/ru` и ловил битые
ссылки только вида `/ru/...` — на 14-языковом сайте это значило: гейт мог сказать
«проблем 0», когда сломан любой из тринадцати остальных языков. Гейт для этого и
существует — солгать он не должен ни для одного языка.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import readycheck  # noqa: E402


def _page(out, lang, slug, body, noindex=False):
    d = os.path.join(out, lang, slug) if slug else os.path.join(out, lang)
    os.makedirs(d, exist_ok=True)
    robots = '<meta name="robots" content="noindex, nofollow">' if noindex else ""
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head>{robots}</head><body>{body}</body></html>")


def test_broken_link_in_a_non_russian_language_is_caught(tmp_path):
    """Битая ссылка на японской странице — не русской — гейт обязан её найти."""
    out = str(tmp_path)
    _page(out, "ru", "", "<h1>главная</h1>" + "текст " * 100)
    _page(out, "ja", "", "<h1>ホーム</h1>" + "テキスト " * 100)
    _page(
        out,
        "ja",
        "gr",
        "<h1>ギリシャ</h1>" + "テキスト " * 100 + '<a href="/ja/gr/visa/">visa</a>',
    )
    # /ja/gr/visa/ не существует — это и есть битая ссылка
    pages, broken, empty, moji = readycheck.scan(out=out)
    assert ("/ja/gr/", "/ja/gr/visa/") in broken, broken


def test_empty_page_in_a_non_russian_language_is_caught(tmp_path):
    """Пустая страница на испанском — гейт находит её, а не только на русском."""
    out = str(tmp_path)
    _page(out, "ru", "", "<h1>главная</h1>" + "текст " * 100)
    _page(out, "es", "", "<h1>vacío</h1>")  # тела почти нет — пустая
    _pages, _broken, empty, _moji = readycheck.scan(out=out)
    assert "/es/" in empty, empty


def test_utility_page_in_any_language_is_not_flagged_empty(tmp_path):
    """Служебная страница (шлюз, noindex) — тонкая по замыслу, брак ей не приписываем
    ни на русском, ни на любом другом языке."""
    out = str(tmp_path)
    _page(out, "de", "go/luky", "", noindex=True)
    _pages, _broken, empty, _moji = readycheck.scan(out=out)
    assert "/de/go/luky/" not in empty, empty


def test_the_assets_directory_is_not_mistaken_for_a_language(tmp_path):
    """`out/assets/` — общие CSS/JS (render.py, copy_assets), НЕ язык. Обход не должен
    туда лезть и путать файлы ассетов со страницами."""
    out = str(tmp_path)
    _page(out, "ru", "", "<h1>главная</h1>" + "текст " * 100)
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    with open(os.path.join(out, "assets", "index.html"), "w", encoding="utf-8") as fh:
        fh.write("не страница")
    pages, _broken, _empty, _moji = readycheck.scan(out=out)
    assert "/assets/" not in pages, pages
