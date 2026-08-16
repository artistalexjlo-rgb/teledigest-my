"""Сторож шага 7: поиск по заголовкам статикой на ВСЕХ страницах + запрос в лог.

Канон §0.12: «по заголовкам, СТАТИКОЙ — индекс кладёт рендер… Ни ключей, ни сервера, ни
продукта» и «поле ввода ставить РАНЬШЕ решения про помощника: чего люди спрашивают, придя
из поиска, мы не знаем — и это, а не выбор архитектуры, главный дефицит».

⛔ Что защищаем:
  1. индекс собирается из ТОГО, ЧТО ОТРЕНДЕРИЛОСЬ, а не из корпуса — иначе поиск ведёт
     в 404 на страницы, которые отсеялись (без адреса, метка не перевелась, гео пустое);
  2. служебные страницы (шлюз, сама страница поиска) в индекс не попадают;
  3. поле есть на КАЖДОЙ странице, а индекс в страницу НЕ инлайнится (352 КБ × 41 630);
  4. страница `/<язык>/find/` существует у каждого языка — иначе поле ведёт в 404;
  5. запрос уходит адресом (`?s=`), потому что именно так он попадает в лог веб-сервера.

⛔ Правила проверены поломкой кода.

Сети, ключей и БД не требует.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import pages as pg  # noqa: E402
import render  # noqa: E402,F401  подменяется в `_fresh_render`: PSEO_OUT читается на импорте
import tail_taxonomy as tax  # noqa: E402

RU_NAME = {k: n for k, n, _ in tax.SHELVES}


def _fresh_render():
    """Перечитать `render`: каталог вывода (`PSEO_OUT`) он берёт на импорте.

    ⛔ Не `importlib.reload`: в ОБЩЕМ прогоне другой сторож выкидывает `render` из
    `sys.modules`, и reload падал с `module render not in sys.modules` — поодиночке
    тесты при этом были зелёные. Импортируем заново и подменяем ссылку в модуле теста.
    """
    global render
    sys.modules.pop("render", None)
    import render as r

    render = r
    return r


def _view(zadacha, n, key, shelf=None):
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
    return {
        "shelf": name,
        "key": key,
        "items": [{"id": f"s{key}{i}", "text": f"Заметка {i}."} for i in range(n)],
    }


def _site(tmp, monkeypatch, views, shelves, extra_pages=()):
    """Собрать гео + служебные страницы и отрендерить ВСЁ, как это делает боевой прогон."""
    data, out = tmp / "data", tmp / "html"
    data.mkdir()
    d = tmp / "out_facet"
    d.mkdir()
    (d / "gr.json").write_text(
        json.dumps(
            {"geo": "gr", "views_by_task": views, "shelves": shelves, "prochee": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, pdata = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp), str(data)
    pg._RU_THEMES.clear()
    pg._THEME_NAMES.clear()
    try:
        pg.build_geo("gr", "ru")
        pg.build_home("ru", ["gr"], {"gr": 1})  # у главной поле СВОЁ, это часть правила
        pg.build_gateway("ru")
        pg.build_find("ru")
    finally:
        pg.BUILT, pg.DATA = built, pdata
    for i, extra in enumerate(extra_pages):
        (data / f"zz_extra{i}.json").write_text(
            json.dumps(extra, ensure_ascii=False), encoding="utf-8"
        )
    monkeypatch.setenv("PSEO_OUT", str(out))
    r = _fresh_render()
    # ⚠️ `BASE` не подменяем: от него берутся i18n, шаблоны и ассеты. Каталог данных
    # передаём параметром — ровно для этого он у `build_all` и появился.
    stat = r.build_all(data_dir=data)
    return stat, out


def test_index_is_built_from_rendered_pages(tmp_path, monkeypatch):
    """Индекс = отрендеренные страницы. Служебных (noindex) в нём нет."""
    views = [
        _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
        _view("Паромы", 5, "ferries", RU_NAME["transport"]),
    ]
    shelves = [
        _shelf("visa", RU_NAME["visa"]),
        _shelf("transport", RU_NAME["transport"]),
    ]
    stat, out = _site(
        tmp_path,
        monkeypatch,
        views,
        shelves,
        extra_pages=[
            # ⛔ Фикстура с ЗАГОЛОВКОМ и `noindex` разом. Без неё мутация «убрать проверку
            # noindex» остаётся зелёной: шлюз и страница поиска и так не имеют h1 в данных,
            # поэтому в индекс не попадают по другой причине, а правило не проверяется.
            {
                "lang": "ru",
                "path": "/ru/sluzhebnaya/",
                "noindex": True,
                "h1": "Служебная страница",
                "template": "index.html.j2",
                "title": "x",
            },
        ],
    )
    idx = json.loads((out / "ru" / "search.json").read_text(encoding="utf-8"))
    titles = {t for t, _ in idx}
    paths = {p for _, p in idx}
    assert "Сроки визы" in titles and "Паромы" in titles, titles
    assert (
        "/ru/go/luky/" not in paths and "/ru/find/" not in paths
    ), "служебное в индексе"
    assert "Служебная страница" not in titles, "страница с noindex попала в индекс"
    assert stat["search_titles"] == len(idx)
    for _t, p in idx:  # ⛔ ни одна запись не ведёт в 404
        assert (out / p.strip("/") / "index.html").exists(), p


def test_dropped_page_never_reaches_the_index(tmp_path, monkeypatch):
    """Отсеянный вид (нет адреса) в индекс не попадает: иначе поиск обещал бы 404."""
    views = [
        _view("Сроки визы", 9, "visa-terms", RU_NAME["visa"]),
        {  # без `key` и с меткой, из которой слаг не выйдет → страницы не будет
            "zadacha": "。。。",
            "items": [{"id": f"x{i}", "text": "Текст."} for i in range(5)],
            "groups": [{"rep": f"x{i}", "ids": [f"x{i}"], "n": 1} for i in range(5)],
            "shelf": RU_NAME["visa"],
        },
    ]
    _stat, out = _site(tmp_path, monkeypatch, views, [_shelf("visa", RU_NAME["visa"])])
    idx = json.loads((out / "ru" / "search.json").read_text(encoding="utf-8"))
    assert "。。。" not in {t for t, _ in idx}, idx


def test_field_on_every_page_and_index_not_inlined(tmp_path, monkeypatch):
    """Поле — на каждой странице, а индекс тянется файлом, а не вшивается в страницу."""
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    _stat, out = _site(tmp_path, monkeypatch, views, [_shelf("visa", RU_NAME["visa"])])
    for rel in (
        "ru/gr/index.html",
        "ru/gr/visa-terms/index.html",
        "ru/gr/s/visa/index.html",
    ):
        html = (out / rel).read_text(encoding="utf-8")
        assert 'id="gq"' in html, f"{rel}: нет поля поиска"
        assert "/ru/search.json" in html, f"{rel}: не указан индекс"
        assert '"Сроки визы","/ru/' not in html, f"{rel}: индекс вшит в страницу"
    home = (out / "ru" / "index.html").read_text(encoding="utf-8")
    assert (
        'id="gq"' not in home
    ), "на главной своё поле по странам — дубля быть не должно"


def test_find_page_exists_and_carries_the_query(tmp_path, monkeypatch):
    """Страница поиска есть, помечена noindex и ищет по тому же индексу.

    Запрос уходит АДРЕСОМ (`?s=`) — только поэтому он и оказывается в логе веб-сервера,
    из которого мы узнаём, что люди спрашивают. Замер 14.08: nginx контейнера пишет
    access.log в `/dev/stdout`, значит запросы видны без своего бэкенда и без монтирования.
    """
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    _stat, out = _site(tmp_path, monkeypatch, views, [_shelf("visa", RU_NAME["visa"])])
    html = (out / "ru" / "find" / "index.html").read_text(encoding="utf-8")
    assert 'name="robots" content="noindex' in html
    assert 'id="results"' in html and "/ru/search.json" in html
    assert 'data-find="/ru/find/"' in html
    # ⛔ Дефект, увиденный на превью: полей было ДВА — из базового шаблона и своё.
    assert html.count('class="search"') == 1, "на странице поиска два поля"
    assert 'id="gq"' not in html, "поле базового шаблона дублирует своё"
    # и заголовок страницы — не подпись поля с многоточием
    assert "<h1>Поиск по заголовкам</h1>" in html, "заголовок из подписи с многоточием"
    js = (HERE.parent / "static" / "base.js").read_text(encoding="utf-8")
    assert 'dataset.find+"?s="+encodeURIComponent' in js, "запрос не уходит адресом"


def test_find_page_for_every_language(tmp_path):
    """Поле на странице ведёт на `/<язык>/find/` — значит она нужна каждому языку."""
    out = tmp_path / "data"
    out.mkdir()
    built, pdata = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), str(out)
    try:
        for lang in ("ru", "de", "th"):
            pg.build_find(lang)
    finally:
        pg.BUILT, pg.DATA = built, pdata
    for lang in ("ru", "de", "th"):
        d = json.loads((out / f"{lang}_find.json").read_text(encoding="utf-8"))
        assert d["path"] == f"/{lang}/find/" and d["noindex"] is True
        assert d["template"] == "find.html.j2"


def test_placeholder_exists_in_every_language():
    """Подпись поля нужна во ВСЕХ языках: иначе на части сайта поле будет без подписи."""
    d = HERE.parent / "i18n"
    langs = sorted(p.stem for p in d.glob("*.json"))
    assert len(langs) >= 14, langs
    for lang in langs:
        t = json.loads((d / f"{lang}.json").read_text(encoding="utf-8"))
        assert t.get("search_ph"), f"{lang}: нет search_ph"
        assert t.get("home_search_none"), f"{lang}: нет строки «ничего не найдено»"


def test_index_is_compressible_and_not_absurdly_big(tmp_path, monkeypatch):
    """Вес индекса — часть решения: он тянется по первому вводу, а не с каждой страницей.

    Замер на настоящей сборке: ru 3124 заголовка = 352 КБ, в gzip 77 КБ. Здесь фикстура
    маленькая, поэтому проверяем не абсолют, а что файл компактный (без отступов) и что
    `application/json` объявлен в gzip_types конфига — иначе nginx отдаст его несжатым.
    """
    import gzip

    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    _stat, out = _site(tmp_path, monkeypatch, views, [_shelf("visa", RU_NAME["visa"])])
    raw = (out / "ru" / "search.json").read_bytes()
    assert b'", "' not in raw and b"\n" not in raw, "индекс с отступами — лишний вес"
    assert len(gzip.compress(raw)) < len(raw) or len(raw) < 200
    # ⛔ Смотрим именно СТРОКУ `gzip_types`, а не файл целиком: первая версия этой проверки
    # ловила слово `application/json` в моём же комментарии выше строки и была зелёной на
    # сломанном конфиге.
    conf = (HERE / "nginx_conf.py").read_text(encoding="utf-8")
    gz = [ln for ln in conf.splitlines() if ln.strip().startswith("gzip_types")]
    assert gz, "в конфиге нет gzip_types вообще"
    assert "application/json" in gz[0], f"индекс поедет несжатым: {gz[0]}"
