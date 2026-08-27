# -*- coding: utf-8 -*-
"""СБОРЩИК САЙТА нового тракта: корпус → страницы дерева из PLAN.md. Ключей НЕ тратит.

Дерево (PLAN.md, «Схема страниц»):

    /<язык>/                                    главная: список стран
      /<язык>/<страна>/                         хаб: плитки тем
        /<язык>/<страна>/<тема>/                тема: кнопки страниц + остаток
          /<язык>/<страна>/<тема>/<страница>/   страница: советы

⛔ ПОЧЕМУ НОВЫЙ МОДУЛЬ, А НЕ ПРАВКА СТАРОГО. `pages.py` строит отменённое дерево: мостик
`/s/` у темы, плоский адрес у страницы, вопрос-контур, полки, `subshelves` и поле `key`.
Половина его 3 100 строк — про сущности, которых в тракте нет. Разбирать это построчно
дороже, чем собрать нужное дерево заново: тут четыре уровня и никакой отменённой схемы.

⛔ АДРЕС НЕСЁТ КОРПУС. Хвост берётся из поля `slug`, который звено 4 кладёт в справочник
имён, и он ОДИНАКОВ во всех языках. Слаг из заголовка тут не считается: заголовок
локализован, и адрес получался бы свой в каждом языке — на этом уже горели.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tract  # noqa: E402
from country_codes import COUNTRIES  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/pseo
DATA = os.environ.get("PSEO_DATA", f"{BASE}/data")
BUILT = os.environ.get("BUILT_DIR", f"{BASE}/builder")

# Имена тем для заголовков. Русские лежат в таксономии, остальные языки покупает звено 6
# ОДИН раз на язык и кладёт в `themes.json` — платить за них в каждой стране незачем.
THEME_NAME = dict(tract.THEMES)


def theme_names(lang):
    """Имена тринадцати тем на языке страницы. Нет перевода — берём ключ, а не русское
    имя: английский `visa` на японской странице честнее русских «Визовых процедур».
    """
    if lang == "ru":
        return THEME_NAME
    mp = (_load(f"{BUILT}/themes.json") or {}).get(lang) or {}
    return {k: mp.get(k) or k.replace("_", " ") for k in tract.THEME_KEYS}


def _load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write(name, page):
    """Записать страницу. Дату ставим ТОЛЬКО при реальном изменении содержимого: свежий
    `lastmod` на неизменной странице — ложь поисковику, и он от таких сигналов отучается.
    """
    import datetime as dt

    os.makedirs(DATA, exist_ok=True)
    path = f"{DATA}/{name}"
    keep = None
    prev = _load(path)
    if prev:
        strip = {"updated", "updated_iso"}
        if {k: v for k, v in prev.items() if k not in strip} == {
            k: v for k, v in page.items() if k not in strip
        }:
            keep = prev.get("updated_iso")
    today = dt.date.today()
    page["updated"] = today.strftime("%m.%Y")
    page["updated_iso"] = keep or today.isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(page, fh, ensure_ascii=False, indent=1)


# Имена стран по языкам — из Unicode CLDR, снято в файл. Это СПРАВОЧНИК, а не перевод:
# у рта их не покупаем, и на сайте они совпадают с тем, что человек видит в телефоне.
COUNTRY_NAMES = (
    _load(f"{os.path.dirname(os.path.abspath(__file__))}/countries.json") or {}
)


def _geo_name(geo, lang):
    """Имя страны на языке страницы. Нет в таблице — русское из справочника, нет и там — код."""
    nm = (COUNTRY_NAMES.get(lang) or {}).get(geo)
    if nm:
        return nm
    pair = COUNTRIES.get(geo)  # флаг и код живут там же, имена там русские
    return pair[0] if pair else geo.upper()


def _geo_flag(geo):
    pair = COUNTRIES.get(geo)
    return pair[1] if pair else ""


def corpus(geo, lang="ru"):
    """Корпус гео: страницы и остатки тем. Читается то, что положило звено 5."""
    d = "out_facet" if lang == "ru" else f"out_facet_{lang}"
    return _load(f"{BUILT}/{d}/{geo}.json")


def advice_page(page_data, geo, lang, siblings=()):
    """Страница советов: `/<язык>/<страна>/<тема>/<страница>/`.

    `siblings` — все части этой же ветки по порядку. Части знают друг о друге, потому что
    ветка выходит на тему ОДНОЙ плиткой (PLAN.md, «Ветвление одной темы»): попасть во
    вторую часть можно только отсюда.
    """
    names = theme_names(lang)
    theme = page_data["theme"]
    slug = page_data["slug"]
    path = f"/{lang}/{geo}/{theme}/{slug}/"
    parts = [
        {
            "n": p.get("part", 1),
            "url": f"/{lang}/{geo}/{theme}/{p['slug']}/",
            "current": p["slug"] == slug,
        }
        for p in siblings
    ]
    faqs = [
        {
            "q": it["text"].split(".")[0][:120],
            "a": it["text"],
            "n": it.get("n", 1),
            "n_word": "",
        }
        for it in page_data["items"]
    ]
    return path, {
        "lang": lang,
        "path": path,
        "template": "page.html.j2",
        "shared_tail": True,  # хвост общий: он из справочника, а не из заголовка
        "geo": geo,
        "geo_name": _geo_name(geo, lang),
        "shelf_url": f"/{lang}/{geo}/{theme}/",
        "shelf_name": names.get(theme, theme),
        "intent_name": page_data["title"],
        "h1": page_data["title"],
        "title": (
            f"{page_data['title']} — {_geo_name(geo, lang)}"
            if len(parts) < 2
            else f"{page_data['title']} {page_data.get('part', 1)}"
            f"/{len(parts)} — {_geo_name(geo, lang)}"
        ),
        "faqs": faqs,
        "chips": [],
        # ⛔ Номер части — в заголовке ОКНА, а не в `h1`: имя одно на всю ветку, а
        # поисковику части разными видеть надо.
        "parts": parts if len(parts) > 1 else [],
        "search_title": page_data["title"],
    }


def by_branch(pages):
    """Страницы темы → ветки: список списков, части внутри по порядку.

    Ключ — поле `branch` от звена 5. Страницы без него (старый корпус) — ветка сама себе.
    """
    vetki = {}
    for p in pages:
        vetki.setdefault(p.get("branch") or p["slug"], []).append(p)
    return [sorted(v, key=lambda p: p.get("part", 1)) for v in vetki.values()]


def theme_page(theme, pages, leftover, geo, lang):
    """Тема: кнопки веток сверху, мелочь остатка списком ниже. Своего текста не несёт."""
    names = theme_names(lang)
    path = f"/{lang}/{geo}/{theme}/"
    # ⛔ ОДНА ПЛИТКА НА ВЕТКУ (PLAN.md, 27.08). Группируем по полю `branch`, а не по имени
    # (оно переводится) и не разбором суффикса `-2` (он следствие, а не признак).
    tiles = []
    for parts in by_branch(pages):
        first = parts[0]
        tiles.append(
            {
                "icon": "",
                "title": first["title"],
                "blurb": f"{sum(len(p['items']) for p in parts)}",
                "url": f"/{lang}/{geo}/{theme}/{first['slug']}/",
            }
        )
    page = {
        "lang": lang,
        "path": path,
        "template": "index.html.j2",
        "shared_tail": True,
        "geo": geo,
        "geo_name": _geo_name(geo, lang),
        "h1": names.get(theme, theme),
        "title": f"{names.get(theme, theme)} — {_geo_name(geo, lang)}",
        "tiles": tiles,
        "search_title": names.get(theme, theme),
    }
    if leftover:
        # Остаток — то, чему имени не нашлось. Показываем списком, чтобы советы не пропали.
        page["faqs"] = [
            {
                "q": it["text"].split(".")[0][:120],
                "a": it["text"],
                "n": it.get("n", 1),
                "n_word": "",
            }
            for it in leftover
        ]
    return path, page


def hub(geo, themes, lang):
    """Хаб страны: плитки тем со счётчиком страниц."""
    path = f"/{lang}/{geo}/"
    names = theme_names(lang)
    tiles = [
        {
            "icon": "",
            "title": names.get(t, t),
            "blurb": f"{n}",
            "url": f"/{lang}/{geo}/{t}/",
        }
        for t, n in themes
    ]
    return path, {
        "lang": lang,
        "path": path,
        "template": "index.html.j2",
        "shared_tail": True,
        "geo": geo,
        "geo_name": _geo_name(geo, lang),
        "h1": _geo_name(geo, lang),
        "title": _geo_name(geo, lang),
        "tiles": tiles,
        "search_title": _geo_name(geo, lang),
    }


def home(geos, lang):
    """Главная языка: список стран, у которых есть хотя бы одна страница."""
    path = f"/{lang}/"
    tiles = [
        {
            "icon": _geo_flag(g),
            "title": _geo_name(g, lang),
            "blurb": f"{n}",
            "url": f"/{lang}/{g}/",
        }
        for g, n in geos
    ]
    return path, {
        "lang": lang,
        "path": path,
        "template": "index.html.j2",
        "shared_tail": False,  # у главной хвоста нет, альтернативы не объявляем
        "h1": "",
        "title": "",
        "tiles": tiles,
        "noindex": False,
    }


def build_geo(geo, lang="ru"):
    """Одна страна: страницы тем и советов. Возвращает (сколько страниц, темы со счётом)."""
    d = corpus(geo, lang)
    if not d:
        return 0, []
    by_theme = {}
    for v in d.get("views_by_task") or []:
        # ⛔ Тема нужна СЕГМЕНТОМ адреса, поэтому берём ключ, а не человеческое имя полки.
        theme = v.get("theme") or _theme_by_name(v.get("shelf"))
        if not theme or not v.get("slug"):
            continue
        by_theme.setdefault(theme, []).append(v)
    leftovers = {}
    for sh in d.get("shelves") or []:
        t = _theme_by_name(sh.get("shelf"))
        if t:
            leftovers[t] = sh.get("items") or []

    n = 0
    for theme, pages in by_theme.items():
        for p in pages:
            p["theme"] = theme
        for parts in by_branch(pages):
            for p in parts:
                _, page = advice_page(p, geo, lang, parts)
                _write(f"{lang}_{geo}_{theme}_{p['slug']}.json", page)
                n += 1
        _, page = theme_page(theme, pages, leftovers.get(theme), geo, lang)
        _write(f"{lang}_{geo}_{theme}.json", page)
        n += 1
    themes = [
        (t, len(v)) for t, v in sorted(by_theme.items(), key=lambda kv: -len(kv[1]))
    ]
    if themes:
        _, page = hub(geo, themes, lang)
        _write(f"{lang}_{geo}.json", page)
        n += 1
    return n, themes


def _theme_by_name(name):
    """Человеческое имя полки → ключ темы. Корпус несёт имя, адресу нужен ключ."""
    for k, n in tract.THEMES:
        if n == name:
            return k
    return None


def build_all(lang="ru"):
    """Все гео, у которых есть корпус. Печатает числа, чтобы прогон был проверяем."""
    d = "out_facet" if lang == "ru" else f"out_facet_{lang}"
    geos, total = [], 0
    for path in sorted(glob.glob(f"{BUILT}/{d}/*.json")):
        geo = os.path.basename(path)[:-5]
        n, themes = build_geo(geo, lang)
        if n:
            geos.append((geo, sum(x for _t, x in themes)))
            total += n
            print(f"  {geo}: {n} страниц-data, тем {len(themes)}", flush=True)
    if geos:
        _, page = home(geos, lang)
        _write(f"{lang}.json", page)
        total += 1
    print(f"ИТОГО {lang}: {total} страниц-data -> {DATA}", flush=True)
    return total


if __name__ == "__main__":
    _lang = sys.argv[2] if len(sys.argv) > 2 else "ru"
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        build_geo(sys.argv[1], _lang)
    else:
        build_all(_lang)
