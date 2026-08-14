"""Сторож шага 5: хаб гео = ПЛИТКИ РАЗДЕЛОВ, а адреса перечислены на странице раздела.

Канон §0.12 целиком: «хаб страны = плитки полок со счётчиком. Адреса живут ВНУТРИ плитки,
**хаб их не перечисляет**». Замер до правила: `/ru/gr/` отдавал 63 ссылки плоским списком,
`/ru/any/` — 87.

⛔ Урок дня, из-за которого этот файл переписан целиком: первый заход я сделал плитку-
аккордеон — адреса лежали в HTML хаба под кликом. Формально «внутри плитки», а вторую
половину правила («хаб их не перечисляет») это нарушало. Обрезанная цитата и есть механизм:
я построил CSS, шаблон и сторожей вокруг своей выдумки. Плитка — ССЫЛКА на страницу раздела.

⛔ ГЛАВНОЕ, ЧТО ЗАЩИЩАЕМ: ни один адрес не теряется. Пропавшую ссылку глазами не отличить
от исправной плитки, поэтому проверяем инвариантом на множествах: каждый разбор достижим
либо со страницы своего раздела, либо карточкой с хаба.

⛔ Каждое правило проверено ПОЛОМКОЙ кода. Зелёный тест на верном коде не доказывает ничего.

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

RU_NAME = {k: n for k, n, _ in tax.SHELVES}


def _view(zadacha, n, key, shelf=None):
    """Разбор: n абзацев (порог страницы — 4), латинский адрес `key`."""
    items = [{"id": f"{key}{i}", "text": f"Совет {i}. Подробность."} for i in range(n)]
    v = {
        "zadacha": zadacha,
        "key": key,
        "items": items,
        "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items],
    }
    if shelf:
        v["shelf"] = shelf
    return v


def _shelf(key, name, n=3):
    """Хвост раздела. Три заметки — порог, при котором страница раздела вообще строится."""
    return {
        "shelf": name,
        "key": key,
        "items": [{"id": f"s{key}{i}", "text": f"Заметка {i}."} for i in range(n)],
    }


def _build(tmp, geo, lang="ru", corpora=None):
    """Собрать одно гео в изоляции. corpora: {язык: {"views": [...], "shelves": [...]}}.

    Кеши имён и разделов чистим: они модульные, а корпус у каждого теста свой.
    """
    out = str(tmp / "out")
    os.makedirs(out, exist_ok=True)
    for cl, body in (corpora or {}).items():
        d = tmp / ("out_facet" if cl == "ru" else f"out_facet_{cl}")
        d.mkdir(exist_ok=True)
        (d / f"{geo}.json").write_text(
            json.dumps(
                {
                    "geo": geo,
                    "views_by_task": body.get("views") or [],
                    "shelves": body.get("shelves") or [],
                    "prochee": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp), out
    pg._RU_THEMES.clear()
    pg._THEME_NAMES.clear()
    try:
        pg.build_geo(geo, lang)
    finally:
        pg.BUILT, pg.DATA = built, data
    files = {
        f: json.load(open(f"{out}/{f}", encoding="utf-8")) for f in os.listdir(out)
    }
    hub = next(v for k, v in files.items() if k.endswith(f"{geo}_hub.json"))
    return hub, files


def _by_path(files):
    return {d["path"]: d for d in files.values() if d.get("path")}


def _themed(hub, lang="ru", geo="gr"):
    """Плитки разделов — те, что ведут в КОНКРЕТНЫЙ раздел.

    ⚠️ Сам `/<lang>/<geo>/s/` — это мостик «Разделы живого опыта», он не плитка раздела.
    Первая версия этого селектора его захватывала, и семь тестов краснели на верном коде.
    """
    pref = f"/{lang}/{geo}/s/"
    return [
        t
        for t in hub["tiles"]
        if t.get("url", "").startswith(pref) and t["url"] != pref
    ]


def _plain(hub, lang="ru", geo="gr"):
    """Карточки разборов, оставшиеся на хабе (без раздела или раздел без страницы)."""
    pref = f"/{lang}/{geo}/"
    return [
        t
        for t in hub["tiles"]
        if t.get("url", "").startswith(pref)
        and not t["url"].startswith(pref + "s/")
        and not t["url"].startswith(pref + "q/")
    ]


def test_hub_gives_sections_not_addresses(tmp_path):
    """Четыре адреса двух разделов → на хабе ДВЕ плитки-ссылки, адресов на хабе нет.

    Это и есть шаг 5. Плитка ведёт на страницу раздела, а не раскрывается.
    """
    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Отказы консульства", 6, "visa-refusals", RU_NAME["visa"]),
            _view("Паромы между островами", 5, "ferries", RU_NAME["transport"]),
            _view("Аренда авто", 4, "car-rental", RU_NAME["transport"]),
        ],
        "shelves": [
            _shelf("visa", RU_NAME["visa"]),
            _shelf("transport", RU_NAME["transport"]),
        ],
    }
    hub, _ = _build(tmp_path, "gr", corpora={"ru": ru})
    tiles = _themed(hub)
    assert len(tiles) == 2, [t.get("title") for t in hub["tiles"]]
    assert {t["url"] for t in tiles} == {"/ru/gr/s/visa/", "/ru/gr/s/transport/"}
    assert not _plain(hub), "адреса разборов остались на хабе"
    assert not any(t.get("links") for t in hub["tiles"]), "вернулось раскрытие плитки"


def test_no_address_is_lost(tmp_path):
    """ИНВАРИАНТ: каждый собранный разбор достижим — со страницы раздела или с хаба.

    Пропавшую ссылку глазами не поймать: плитка выглядит одинаково в обоих случаях.
    """
    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Паромы", 5, "ferries", RU_NAME["transport"]),
            _view("Аренда авто", 4, "car-rental", RU_NAME["transport"]),
            _view("Прочее", 4, "misc"),  # без раздела — тоже обязан быть достижим
        ],
        "shelves": [
            _shelf("visa", RU_NAME["visa"]),
            _shelf("transport", RU_NAME["transport"]),
        ],
    }
    hub, files = _build(tmp_path, "gr", corpora={"ru": ru})
    by_path = _by_path(files)
    reachable = {t["url"] for t in _plain(hub)}
    for t in _themed(hub):
        sec = by_path[t["url"]]
        reachable |= {x["url"] for x in sec.get("tiles") or []}
    pages = {
        p
        for p in by_path
        if p.startswith("/ru/gr/")
        and p != "/ru/gr/"
        and not p.startswith("/ru/gr/s/")
        and not p.startswith("/ru/gr/q/")
    }
    assert reachable == pages, f"достижимы {sorted(reachable)}, собрано {sorted(pages)}"


def test_section_page_keeps_its_tail(tmp_path):
    """На странице раздела разборы СВЕРХУ, хвост НИЖЕ — и хвост никуда не пропадает.

    Юзер решил 13.08: «как хвосты лежат, пусть лежат». Значит нельзя ни выкинуть их, ни
    увести на новый адрес. Хвост без дедуп-групп (живой случай: у `gr` 24 заметки и 0
    групп) обязан печататься пунктами — иначе он исчезает молча.
    """
    ru = {
        "views": [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])],
        "shelves": [_shelf("visa", RU_NAME["visa"], n=4)],
    }
    _hub, files = _build(tmp_path, "gr", corpora={"ru": ru})
    sec = _by_path(files)["/ru/gr/s/visa/"]
    assert sec["template"] == "index.html.j2", sec["template"]
    assert [t["url"] for t in sec["tiles"]] == ["/ru/gr/visa-terms/"]
    assert len(sec.get("questions") or sec.get("faqs") or []) == 4, "хвост потерян"


def test_ceiling_is_taxonomy_not_page_count(tmp_path):
    """Плиток не больше, чем разделов в таксономии, сколько бы страниц ни собралось.

    Цель юзера дословно: «до 15 плиток с внутренностями» — 13 разделов плюс мостики.
    """
    views, shelves = [], []
    for i, (k, nm, _d) in enumerate(tax.SHELVES):
        shelves.append(_shelf(k, nm))
        for j in range(4):  # 52 разбора на 13 разделов
            views.append(_view(f"Тема {i}-{j}", 4 + j, f"t{i}-{j}", nm))
    hub, files = _build(
        tmp_path, "br", corpora={"ru": {"views": views, "shelves": shelves}}
    )
    assert len(_themed(hub, geo="br")) == 13, len(hub["tiles"])
    assert len(hub["tiles"]) <= 15, [t.get("title") for t in hub["tiles"]]
    on_sections = sum(
        len(_by_path(files)[t["url"]]["tiles"]) for t in _themed(hub, geo="br")
    )
    assert on_sections == 52, on_sections


def test_section_without_page_keeps_cards_on_hub(tmp_path):
    """Раздел без своей страницы (хвост тоньше трёх) плитки не получает — вести некуда.

    ⛔ Замер 13.08: таких пар «гео × раздел» 12 из 249 в 9 гео, за ними 23 разбора. Плитка
    в никуда — это 404 прямо из навигации, а мы это уже проходили на пустых гео.
    """
    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Крит", 7, "crete", RU_NAME["tourism"]),
        ],
        "shelves": [_shelf("visa", RU_NAME["visa"])],  # у туризма хвоста нет
    }
    hub, _ = _build(tmp_path, "gr", corpora={"ru": ru})
    assert [t["url"] for t in _themed(hub)] == ["/ru/gr/s/visa/"]
    assert [t["url"] for t in _plain(hub)] == ["/ru/gr/crete/"], hub["tiles"]


def test_view_without_section_stays_reachable(tmp_path):
    """Разбор без раздела остаётся карточкой на хабе.

    Таких 10 на корпус, все со сборной меткой («Прочее», «Общие советы») — брак нарезки.
    Лечится он в карве, а до тех пор адрес обязан быть достижим.
    """
    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Прочие вопросы", 5, "misc"),
        ],
        "shelves": [_shelf("visa", RU_NAME["visa"])],
    }
    hub, _ = _build(tmp_path, "gr", corpora={"ru": ru})
    assert [t["url"] for t in _plain(hub)] == ["/ru/gr/misc/"], hub["tiles"]


def test_bigger_section_first(tmp_path):
    """Порядок детерминирован: крупный раздел выше. Из словаря порядок пришёл бы от
    порядка данных, и хаб перетряхивался бы каждый прогон — лишний дифф в репо страниц.
    """
    ru = {
        "views": [
            _view("Мелочь", 4, "small", RU_NAME["customs"]),
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Отказы", 21, "visa-refusals", RU_NAME["visa"]),
        ],
        "shelves": [
            _shelf("customs", RU_NAME["customs"]),
            _shelf("visa", RU_NAME["visa"]),
        ],
    }
    hub, _ = _build(tmp_path, "gr", corpora={"ru": ru})
    assert [t["url"] for t in _themed(hub)] == ["/ru/gr/s/visa/", "/ru/gr/s/customs/"]


def test_other_language_groups_via_stamped_key(tmp_path):
    """Другой язык берёт раздел из РУССКОГО корпуса по `key`, а имя — из своего корпуса.

    Перевод раздел пока не несёт (замер 13.08: в `out_facet_de` он у 0 разборов из 1851), а
    штамп адреса есть у всех — на нём и держится соединение.
    """
    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Паромы", 5, "ferries", RU_NAME["transport"]),
        ]
    }
    de = {
        "views": [_view("Visafristen", 9, "visa-terms"), _view("Fahren", 5, "ferries")],
        "shelves": [
            _shelf("visa", "Visabestimmungen"),
            _shelf("transport", "Transport und Logistik"),
        ],
    }
    hub, _ = _build(tmp_path, "gr", lang="de", corpora={"ru": ru, "de": de})
    tiles = _themed(hub, lang="de")
    assert {t["title"] for t in tiles} == {
        "Visabestimmungen",
        "Transport und Logistik",
    }, [t.get("title") for t in hub["tiles"]]
    assert all(
        t["icon"] != "•" for t in tiles
    ), "иконка обязана идти от КЛЮЧА, не от имени"


def test_language_without_section_name_stays_flat(tmp_path):
    """Нет имени раздела на языке → хаб остаётся плоским ЦЕЛИКОМ, причина печатается.

    Полу-состояние в этом проекте уже стоило прода. Живой случай ровно такой: языковые
    корпуса стоят на таксономии v0 (9 ключей), пяти новых имён там нет, а `tourism` —
    третий раздел корпуса по массе.

    ⛔ Фикстура нарочно СМЕШАННАЯ: один раздел на языке назван и страницу имеет (визы),
    второй — нет (туризм). Иначе мутация «не выключать группировку» остаётся зелёной: при
    одном безымянном разделе результат совпадает с «плоским» случайно.
    """
    import io
    from contextlib import redirect_stdout

    ru = {
        "views": [
            _view("Крит", 7, "crete", RU_NAME["tourism"]),
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
        ]
    }
    de = {
        "views": [_view("Kreta", 7, "crete"), _view("Visafristen", 9, "visa-terms")],
        "shelves": [_shelf("visa", "Visabestimmungen")],  # имени туризма нет нигде
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        hub, _ = _build(tmp_path, "gr", lang="de", corpora={"ru": ru, "de": de})
    assert not _themed(hub, lang="de"), "часть плиток вышла, часть нет — полумера"
    assert {t["url"] for t in _plain(hub, lang="de")} == {
        "/de/gr/crete/",
        "/de/gr/visa-terms/",
    }, hub["tiles"]
    log = buf.getvalue()
    assert "хаб плоский" in log and "tourism" in log, f"причина не напечатана: {log!r}"


def test_hub_says_what_it_did(tmp_path):
    """Итог печатается: сколько плиток на сколько адресов и сколько осталось карточками.

    Молчаливый шаг — то, из-за чего 33 пустых хаба жили год незамеченными.
    """
    import io
    from contextlib import redirect_stdout

    ru = {
        "views": [
            _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
            _view("Прочее", 4, "misc"),
        ],
        "shelves": [_shelf("visa", RU_NAME["visa"])],
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        _build(tmp_path, "gr", corpora={"ru": ru})
    log = buf.getvalue()
    assert "плиток разделов" in log and "карточками 1" in log, f"не напечатано: {log!r}"


def test_section_page_html_has_tiles_and_tail(tmp_path, monkeypatch):
    """Правило ШАБЛОНА, проверенное настоящим рендером: на странице раздела в HTML есть и
    плитки разборов, и заметки хвоста.

    ⛔ Повод: мутация «шаблон не печатает заметки хвоста» оставалась ЗЕЛЁНОЙ, потому что все
    сторожа смотрели только в JSON. Хвост без дедуп-групп — живой случай (`gr`: 24 заметки,
    0 групп), и молча терять его нельзя: до шаблона данные доходили, а на страницу нет.
    """
    import importlib

    ru = {
        "views": [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])],
        "shelves": [_shelf("visa", RU_NAME["visa"], n=3)],
    }
    _hub, files = _build(tmp_path, "gr", corpora={"ru": ru})
    src = next(
        f"{tmp_path}/out/{k}"
        for k, v in files.items()
        if v.get("path") == "/ru/gr/s/visa/"
    )
    monkeypatch.setenv("PSEO_OUT", str(tmp_path / "html"))
    sys.path.insert(0, str(HERE.parent))
    import render

    importlib.reload(render)
    html = pathlib.Path(render.build(src)).read_text(encoding="utf-8")
    assert "/ru/gr/visa-terms/" in html, "плитки разборов не отрисовались"
    assert "Заметка 0." in html, "заметки хвоста не отрисовались"
