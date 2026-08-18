"""Сторож корней A и B из разбора 14.08: уровень раздела доехал до страниц под ним.

Разбор по схеме `SITE_MAP.md` дал восемь расхождений, и они свелись к двум корням:

**A — раздел не доезжал до страниц ниже хаба.** Хаб отправляет через раздел, а крошка на
разборе была «Главная / Страна / Тема»: вниз пройти можно, вверх — нет. И блок «рядом по
теме» собирался из ВСЕХ видов страны подряд, хотя канон §0.12 требует «из той же полки» —
правило стояло записанным с 11.08 и не исполнялось.

**B — страница раздела не имела своей роли.** Заголовок брался из пула заголовков РАЗБОРОВ,
подпись списка гласила «Темы» (а это страницы, не темы), а счётчик плитки на хабе считал
только советы разборов и не учитывал заметки хвоста — то есть обещал меньше, чем внутри.

Сети, ключей и БД не требует.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import pages as pg  # noqa: E402
import tail_taxonomy as tax  # noqa: E402

RU = {k: n for k, n, _ in tax.SHELVES}


def _v(z, n, key, shelf=None):
    items = [{"id": f"{key}{i}", "text": f"Совет {i}. Подробность."} for i in range(n)]
    v = {
        "zadacha": z,
        "key": key,
        "items": items,
        "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items],
    }
    if shelf:
        v["shelf"] = shelf
    return v


def _sh(key, name, n=3):
    return {
        "shelf": name,
        "key": key,
        "items": [{"id": f"s{key}{i}", "text": f"Заметка {i}."} for i in range(n)],
    }


def _build(tmp, views, shelves, geo="gr", lang="ru"):
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    (tmp / "out_facet").mkdir(exist_ok=True)
    (tmp / "out_facet" / f"{geo}.json").write_text(
        json.dumps(
            {
                "geo": geo,
                "views_by_task": views,
                "shelves": shelves,
                "prochee": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp), str(out)
    pg._RU_THEMES.clear()
    pg._THEME_NAMES.clear()
    try:
        pg.build_geo(geo, lang)
    finally:
        pg.BUILT, pg.DATA = built, data
    return {
        d["path"]: d
        for d in (json.load(open(out / f, encoding="utf-8")) for f in os.listdir(out))
        if d.get("path")
    }


VIEWS = [
    _v("Сроки оформления визы", 9, "visa-terms", RU["visa"]),
    _v("Причины отказов", 6, "visa-refusals", RU["visa"]),
    _v("Паромы между островами", 5, "ferries", RU["transport"]),
]
SHELVES = [_sh("visa", RU["visa"], n=4), _sh("transport", RU["transport"], n=3)]


def test_crumb_of_a_page_leads_to_its_section(tmp_path):
    """КОРЕНЬ A: с разбора можно вернуться в свой раздел, а не только в страну."""
    pages = _build(tmp_path, VIEWS, SHELVES)
    p = pages["/ru/gr/visa-terms/"]
    assert p.get("shelf_url") == "/ru/gr/s/visa/", p.get("shelf_url")
    assert p.get("shelf_name") == RU["visa"], p.get("shelf_name")
    # и шаблон это печатает
    tpl = (HERE.parent / "templates" / "page.html.j2").read_text(encoding="utf-8")
    assert (
        "page.shelf_url" in tpl and "page.shelf_name" in tpl
    ), "крошка не выводит раздел"


def test_neighbours_come_from_the_same_section(tmp_path):
    """КОРЕНЬ A: «рядом по теме» — из своего раздела. Канон §0.12 требовал этого с 11.08."""
    pages = _build(tmp_path, VIEWS, SHELVES)
    chips = pages["/ru/gr/visa-terms/"]["chips"]
    urls = {c["url"] for c in chips}
    assert urls == {"/ru/gr/visa-refusals/"}, urls  # паромов тут быть не должно


def test_orphan_page_keeps_neighbours(tmp_path):
    """У страницы без раздела блок «рядом» не должен исчезнуть — иначе лечение хуже болезни."""
    views = VIEWS + [_v("Культура курения", 4, "smoking")]
    pages = _build(tmp_path, views, SHELVES)
    assert pages["/ru/gr/smoking/"]["chips"], "у страницы без раздела пропали соседи"


def test_section_page_declares_itself(tmp_path):
    """КОРЕНЬ B: заголовок страницы раздела — имя раздела, а не заголовок из пула разборов,
    и подпись списка говорит «страницы раздела», а не «темы»."""
    pages = _build(tmp_path, VIEWS, SHELVES)
    sec = pages["/ru/gr/s/visa/"]
    assert sec["h1"] == RU["visa"], sec["h1"]
    assert sec["list_label"] == pg.COPY["ru"]["list_label_pages"], sec["list_label"]
    assert sec["list_label"] != pg.COPY["ru"]["list_label_topics"]


def test_section_label_exists_in_every_language():
    """Подпись нужна во ВСЕХ языках: иначе часть сайта соберётся с KeyError."""
    for lang, c in pg.COPY.items():
        assert c.get("list_label_pages"), f"{lang}: нет list_label_pages"
        assert c["list_label_pages"] != c["list_label_topics"], lang


def test_tile_counter_covers_the_whole_page(tmp_path):
    """КОРЕНЬ B: счётчик плитки = советы разборов + заметки хвоста.

    Было: плитка обещала только советы разборов, а на странице лежал ещё хвост.
    """
    pages = _build(tmp_path, VIEWS, SHELVES)
    hub = pages["/ru/gr/"]
    tile = next(t for t in hub["tiles"] if t.get("url") == "/ru/gr/s/visa/")
    # 9 + 6 советов разборов + 4 заметки хвоста
    assert "19" in tile["blurb"], tile["blurb"]
    tile_tr = next(t for t in hub["tiles"] if t.get("url") == "/ru/gr/s/transport/")
    assert "8" in tile_tr["blurb"], tile_tr["blurb"]  # 5 + 3


def test_shelf_address_rule_lives_in_one_place():
    """Адрес раздела считается ОДНОЙ функцией: крошка, плитка и сам контур зовут её.

    Правило в двух копиях — болезнь проекта; раньше эта функция была локальной внутри
    полочного блока, и снаружи повторить её было нечем.
    """
    src = (HERE / "pages.py").read_text(encoding="utf-8")
    assert "def shelf_slug(" in src
    # ⛔ Ровно ОДНО вхождение: выражение «ключ раздела» я успел повторить в четырёх местах,
    # и этот сторож поймал мою же копию. Оставлено 1 — в `shelf_key_of`.
    assert "def shelf_key_of(" in src
    assert src.count("SHELF_KEY.get(sv[") == 1, "правило ключа раздела снова в копиях"
