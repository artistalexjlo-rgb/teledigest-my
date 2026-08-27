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

⛔ АДРЕС НЕСЁТ КОРПУС. Хвост берётся из поля `adres`, который звено 4 кладёт в справочник
имён, и он ОДИНАКОВ во всех языках. Слаг из заголовка тут не считается: заголовок
локализован, и адрес получался бы свой в каждом языке — на этом уже горели.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tail_taxonomy as tax  # noqa: E402
from country_codes import COUNTRIES  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/pseo
DATA = os.environ.get("PSEO_DATA", f"{BASE}/data")
BUILT = os.environ.get("BUILT_DIR", f"{BASE}/builder")

# Имена тем для заголовков берём из таксономии — второго списка не заводим.
TEMA_NAME = {k: n for k, n, _d in tax.SHELVES}


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


def _geo_name(geo, lang):
    """Имя страны. Справочник хранит пару (имя, флаг) — берём имя, флаг для плиток.

    Имена в справочнике русские: перевод названий стран — работа звена 6, не сборщика.
    """
    pair = COUNTRIES.get(geo)
    return pair[0] if pair else geo.upper()


def _geo_flag(geo):
    pair = COUNTRIES.get(geo)
    return pair[1] if pair else ""


def korpus(geo, lang="ru"):
    """Корпус гео: страницы и остатки тем. Читается то, что положило звено 5."""
    d = "out_facet" if lang == "ru" else f"out_facet_{lang}"
    return _load(f"{BUILT}/{d}/{geo}.json")


def stranica(page_data, geo, lang, vetka=()):
    """Страница советов: `/<язык>/<страна>/<тема>/<страница>/`.

    `vetka` — все части этой же ветки по порядку. Части знают друг о друге, потому что
    ветка выходит на тему ОДНОЙ плиткой (PLAN.md, «Ветвление одной темы»): попасть во
    вторую часть можно только отсюда.
    """
    tema = page_data["tema"]
    adres = page_data["adres"]
    path = f"/{lang}/{geo}/{tema}/{adres}/"
    chasti = [
        {
            "n": p.get("part", 1),
            "url": f"/{lang}/{geo}/{tema}/{p['adres']}/",
            "current": p["adres"] == adres,
        }
        for p in vetka
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
        "shelf_url": f"/{lang}/{geo}/{tema}/",
        "shelf_name": TEMA_NAME.get(tema, tema),
        "intent_name": page_data["zadacha"],
        "h1": page_data["zadacha"],
        "title": (
            f"{page_data['zadacha']} — {_geo_name(geo, lang)}"
            if len(chasti) < 2
            else f"{page_data['zadacha']} {page_data.get('part', 1)}"
            f"/{len(chasti)} — {_geo_name(geo, lang)}"
        ),
        "faqs": faqs,
        "chips": [],
        # ⛔ Номер части — в заголовке ОКНА, а не в `h1`: имя одно на всю ветку, а
        # поисковику части разными видеть надо.
        "chasti": chasti if len(chasti) > 1 else [],
        "search_title": page_data["zadacha"],
    }


def po_vetkam(stranicy):
    """Страницы темы → ветки: список списков, части внутри по порядку.

    Ключ — поле `branch` от звена 5. Страницы без него (старый корпус) — ветка сама себе.
    """
    vetki = {}
    for p in stranicy:
        vetki.setdefault(p.get("branch") or p["adres"], []).append(p)
    return [sorted(v, key=lambda p: p.get("part", 1)) for v in vetki.values()]


def tema_stranica(tema, stranicy, ostatok, geo, lang):
    """Тема: кнопки страниц сверху, остаток списком ниже. Своего текста не несёт."""
    path = f"/{lang}/{geo}/{tema}/"
    # ⛔ ОДНА ПЛИТКА НА ВЕТКУ (PLAN.md, 27.08). Группируем по полю `branch`, а не по имени
    # (оно переводится) и не разбором суффикса `-2` (он следствие, а не признак).
    tiles = []
    for chasti in po_vetkam(stranicy):
        pervaya = chasti[0]
        tiles.append(
            {
                "icon": "",
                "title": pervaya["zadacha"],
                "blurb": f"{sum(len(p['items']) for p in chasti)}",
                "url": f"/{lang}/{geo}/{tema}/{pervaya['adres']}/",
            }
        )
    page = {
        "lang": lang,
        "path": path,
        "template": "index.html.j2",
        "shared_tail": True,
        "geo": geo,
        "geo_name": _geo_name(geo, lang),
        "h1": TEMA_NAME.get(tema, tema),
        "title": f"{TEMA_NAME.get(tema, tema)} — {_geo_name(geo, lang)}",
        "tiles": tiles,
        "search_title": TEMA_NAME.get(tema, tema),
    }
    if ostatok:
        # Остаток — то, чему имени не нашлось. Показываем списком, чтобы советы не пропали.
        page["faqs"] = [
            {
                "q": it["text"].split(".")[0][:120],
                "a": it["text"],
                "n": it.get("n", 1),
                "n_word": "",
            }
            for it in ostatok
        ]
    return path, page


def hub(geo, temy, lang):
    """Хаб страны: плитки тем со счётчиком страниц."""
    path = f"/{lang}/{geo}/"
    tiles = [
        {
            "icon": "",
            "title": TEMA_NAME.get(t, t),
            "blurb": f"{n}",
            "url": f"/{lang}/{geo}/{t}/",
        }
        for t, n in temy
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


def glavnaya(geos, lang):
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


def sobrat_geo(geo, lang="ru"):
    """Одна страна: страницы тем и советов. Возвращает (сколько страниц, темы со счётом)."""
    d = korpus(geo, lang)
    if not d:
        return 0, []
    po_teme = {}
    for v in d.get("views_by_task") or []:
        # ⛔ Тема нужна СЕГМЕНТОМ адреса, поэтому берём ключ, а не человеческое имя полки.
        tema = v.get("tema") or _tema_po_imeni(v.get("shelf"))
        if not tema or not v.get("adres"):
            continue
        po_teme.setdefault(tema, []).append(v)
    ostatki = {}
    for sh in d.get("shelves") or []:
        t = _tema_po_imeni(sh.get("shelf"))
        if t:
            ostatki[t] = sh.get("items") or []

    n = 0
    for tema, stranicy in po_teme.items():
        for p in stranicy:
            p["tema"] = tema
        for chasti in po_vetkam(stranicy):
            for p in chasti:
                _, page = stranica(p, geo, lang, chasti)
                _write(f"{lang}_{geo}_{tema}_{p['adres']}.json", page)
                n += 1
        _, page = tema_stranica(tema, stranicy, ostatki.get(tema), geo, lang)
        _write(f"{lang}_{geo}_{tema}.json", page)
        n += 1
    temy = [(t, len(v)) for t, v in sorted(po_teme.items(), key=lambda kv: -len(kv[1]))]
    if temy:
        _, page = hub(geo, temy, lang)
        _write(f"{lang}_{geo}.json", page)
        n += 1
    return n, temy


def _tema_po_imeni(name):
    """Человеческое имя полки → ключ темы. Корпус несёт имя, адресу нужен ключ."""
    for k, n, _d in tax.SHELVES:
        if n == name:
            return k
    return None


def sobrat_vse(lang="ru"):
    """Все гео, у которых есть корпус. Печатает числа, чтобы прогон был проверяем."""
    d = "out_facet" if lang == "ru" else f"out_facet_{lang}"
    geos, vsego = [], 0
    for path in sorted(glob.glob(f"{BUILT}/{d}/*.json")):
        geo = os.path.basename(path)[:-5]
        n, temy = sobrat_geo(geo, lang)
        if n:
            geos.append((geo, sum(x for _t, x in temy)))
            vsego += n
            print(f"  {geo}: {n} страниц-data, тем {len(temy)}", flush=True)
    if geos:
        _, page = glavnaya(geos, lang)
        _write(f"{lang}.json", page)
        vsego += 1
    print(f"ИТОГО {lang}: {vsego} страниц-data -> {DATA}", flush=True)
    return vsego


if __name__ == "__main__":
    _lang = sys.argv[2] if len(sys.argv) > 2 else "ru"
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        sobrat_geo(sys.argv[1], _lang)
    else:
        sobrat_vse(_lang)
