"""Сторож шага 6: довод CTA — ПО РАЗДЕЛУ, а дверь в продукт — через ШЛЮЗ.

Канон §0.12: «страница не тупик: блок рядом по теме из той же полки + довод CTA ПО ПОЛКЕ.
Сейчас довод выбирается по хешу пути, поэтому на странице про сроки визы предлагают
„официант не перепутает заказ“». И второе: статистики переходов в Luky не было НИ ОДНОЙ
цифры — считать было нечем.

⛔ Что защищаем:
  1. на визовой странице голосовой довод больше НЕ про официанта;
  2. адресность там, где копирайт её позволяет (туризм, транспорт, покупки, здоровье, жильё);
  3. ВСЕ двери страницы — и кнопка, и маркеры `#luky` в текстах — идут через шлюз с
     `?geo=&shelf=`, иначе переход не посчитан;
  4. сам шлюз ведёт в продукт НАПРЯМУЮ (иначе бесконечная петля на себя) и не лезет в карту.

⛔ Правила проверены поломкой кода, а не только зелёным прогоном.

Сети, ключей и БД не требует.
"""

import importlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import pages as pg  # noqa: E402
import render  # noqa: E402
import tail_taxonomy as tax  # noqa: E402

RU_NAME = {k: n for k, n, _ in tax.SHELVES}
I18N_RU = json.loads((HERE.parent / "i18n" / "ru.json").read_text(encoding="utf-8"))


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


def _build(tmp, geo="gr", lang="ru", views=(), shelves=()):
    out = str(tmp / "out")
    os.makedirs(out, exist_ok=True)
    d = tmp / "out_facet"
    d.mkdir(exist_ok=True)
    (d / f"{geo}.json").write_text(
        json.dumps(
            {
                "geo": geo,
                "views_by_task": list(views),
                "shelves": list(shelves),
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
        pg.build_gateway(lang)
    finally:
        pg.BUILT, pg.DATA = built, data
    return {
        f: json.load(open(f"{out}/{f}", encoding="utf-8")) for f in os.listdir(out)
    }, out


def _pages_by_path(files):
    return {d["path"]: d for d in files.values() if d.get("path")}


def test_shelf_key_lands_on_pages(tmp_path):
    """Без `shelf_key` на странице нечем ни выбрать довод, ни пометить переход."""
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    files, _ = _build(tmp_path, views=views, shelves=[_shelf("visa", RU_NAME["visa"])])
    p = _pages_by_path(files)
    assert p["/ru/gr/visa-terms/"]["shelf_key"] == "visa"
    assert p["/ru/gr/s/visa/"]["shelf_key"] == "visa"


def test_visa_page_no_longer_talks_about_waiter(tmp_path):
    """ЖИВОЙ ДЕФЕКТ из канона: на визовой странице обещали «официант не перепутает заказ».

    Проверяем не строку-заглушку, а РЕАЛЬНЫЙ русский пул: довод обязан быть из
    универсальных, а не из ресторанных.
    """
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    files, _ = _build(tmp_path, views=views, shelves=[_shelf("visa", RU_NAME["visa"])])
    page = _pages_by_path(files)["/ru/gr/visa-terms/"]
    cta = render.build_cta(I18N_RU, page)
    pool = I18N_RU["cta_pools"]["voice"]
    assert "официант" not in cta["voice"], cta["voice"]
    assert cta["voice"] in [pool[i] for i in render.VOICE_ANY], cta["voice"]


def test_addressed_shelves_get_their_own_lines(tmp_path):
    """Где копирайт позволяет — довод адресный: врач для здоровья, таксист для транспорта."""
    views = [
        _view("Врачи и аптеки", 6, "clinics", RU_NAME["health"]),
        _view("Паромы", 5, "ferries", RU_NAME["transport"]),
    ]
    shelves = [
        _shelf("health", RU_NAME["health"]),
        _shelf("transport", RU_NAME["transport"]),
    ]
    files, _ = _build(tmp_path, views=views, shelves=shelves)
    p = _pages_by_path(files)
    pool = I18N_RU["cta_pools"]["voice"]
    v_health = render.build_cta(I18N_RU, p["/ru/gr/clinics/"])["voice"]
    v_transport = render.build_cta(I18N_RU, p["/ru/gr/ferries/"])["voice"]
    assert v_health == pool[8], v_health  # «поговори с врачом»
    assert v_transport in [pool[i] for i in render.VOICE_BY_SHELF["transport"]]


def test_voice_map_indices_exist_in_every_language():
    """Карта индексов — ОДНА на 14 языков, значит опираться она может только на то, что
    есть в каждом пуле. Индекс за границей пула молча дал бы другой язык другой довод.
    """
    langs = sorted(p.stem for p in (HERE.parent / "i18n").glob("*.json"))
    assert len(langs) >= 14, langs
    used = {i for idx in render.VOICE_BY_SHELF.values() for i in idx} | set(
        render.VOICE_ANY
    )
    for lang in langs:
        pool = json.loads(
            (HERE.parent / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        )["cta_pools"]["voice"]
        assert max(used) < len(
            pool
        ), f"{lang}: пул {len(pool)}, а карта ждёт {max(used)}"


def test_every_door_goes_through_the_gateway(tmp_path, monkeypatch):
    """И кнопка, и маркеры `#luky` в текстах ведут через шлюз с гео и разделом.

    ⛔ Иначе переход не посчитан: до шага 6 в тексте стоял прямой адрес продукта.
    ⚠️ Дверей на странице разное число по шаблону: `page.html.j2` печатает только кнопку
    (интро там заменено коротким ответом), `index.html.j2` — ещё и маркер в интро. Поэтому
    проверяем ОБА: на разборе одна дверь, на разделе две, и обе через шлюз.
    """
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    files, out = _build(
        tmp_path, views=views, shelves=[_shelf("visa", RU_NAME["visa"])]
    )
    monkeypatch.setenv("PSEO_OUT", str(tmp_path / "html"))
    importlib.reload(render)
    want = "/ru/go/luky/?geo=gr&amp;shelf=visa"
    seen = {}
    for path in ("/ru/gr/visa-terms/", "/ru/gr/s/visa/"):
        src = next(f"{out}/{k}" for k, v in files.items() if v.get("path") == path)
        html = pathlib.Path(render.build(src)).read_text(encoding="utf-8")
        seen[path] = html.count(want)
        assert 'class="btn" href="https://' not in html, f"{path}: кнопка мимо шлюза"
        assert "href='#luky'" not in html, f"{path}: маркер остался незаменённым"
    assert seen["/ru/gr/visa-terms/"] == 1, seen
    assert seen["/ru/gr/s/visa/"] >= 2, seen  # кнопка + маркер в интро


def test_gateway_page_is_direct_and_not_indexed(tmp_path, monkeypatch):
    """Шлюз ведёт в продукт НАПРЯМУЮ и в карту сайта не идёт.

    Петля на себя означала бы, что человек в продукт не попадает вообще.
    """
    views = [_view("Сроки визы", 9, "visa-terms", RU_NAME["visa"])]
    files, out = _build(
        tmp_path, views=views, shelves=[_shelf("visa", RU_NAME["visa"])]
    )
    gw = _pages_by_path(files)["/ru/go/luky/"]
    assert gw["noindex"] is True
    assert gw["template"] == "go.html.j2"
    monkeypatch.setenv("PSEO_OUT", str(tmp_path / "html"))
    importlib.reload(render)
    assert render.door_url(gw) == render.SITE["cta_luky_url"], "шлюз шлёт на себя"
    assert not render._indexable(gw), "шлюз попал бы в карту сайта"
    src = next(
        f"{out}/{k}" for k, v in files.items() if v.get("path") == "/ru/go/luky/"
    )
    html = pathlib.Path(render.build(src)).read_text(encoding="utf-8")
    assert "location.replace" in html and render.SITE["cta_luky_url"] in html
    assert "/ru/go/luky/?" not in html, "шлюз ссылается на себя же"


def test_gate_survives_the_gateway(tmp_path):
    """Гейт готовности не должен ломаться о шлюз, а он ломался дважды.

    ⛔ 1. Ссылки он резолвил ВМЕСТЕ со строкой запроса: `/ru/go/luky/?geo=gr&shelf=visa`
    искался как каталог с `?` в имени. То есть после шага 6 «битой» стала бы КАЖДАЯ дверь
    сайта — около двух тысяч, и гейт запретил бы публикацию навсегда.
    ⛔ 2. Шлюз тонкий по замыслу (это пересылка, не контент) и попал бы в «пустые».
    Проверяем оба правила на настоящем HTML, а не на выдумке.
    """
    import readycheck

    out = tmp_path / "out"
    (out / "ru" / "gr" / "visa-terms").mkdir(parents=True)
    (out / "ru" / "go" / "luky").mkdir(parents=True)
    body = "<p>" + ("Живой опыт из чатов. " * 40) + "</p>"
    (out / "ru" / "gr" / "visa-terms" / "index.html").write_text(
        "<html><h1>Сроки визы</h1>"
        + body
        + '<a href="/ru/go/luky/?geo=gr&amp;shelf=visa">Luky</a></html>',
        encoding="utf-8",
    )
    (out / "ru" / "go" / "luky" / "index.html").write_text(
        '<html><meta name="robots" content="noindex, nofollow"><h1>Открыть Luky</h1>'
        "<p>Пересылаем…</p></html>",
        encoding="utf-8",
    )
    (out / "sitemap.xml").write_text("<urlset></urlset>", encoding="utf-8")
    pages, broken, empty, moji = readycheck.scan(str(out))
    assert len(pages) == 2, pages
    assert broken == [], broken  # дверь со строкой запроса — не битая ссылка
    assert empty == [], empty  # шлюз тонкий по замыслу, «пустой» он не считается
    assert moji == []


def test_gateway_exists_for_every_built_language(tmp_path):
    """Шлюз нужен КАЖДОМУ языку: страницы ведут на `/<язык>/go/luky/`, и без страницы
    это 404 из каждой кнопки сайта."""
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), out
    try:
        for lang in ("ru", "de", "ar"):
            pg.build_gateway(lang)
    finally:
        pg.BUILT, pg.DATA = built, data
    for lang in ("ru", "de", "ar"):
        d = json.load(open(f"{out}/{lang}_go_luky.json", encoding="utf-8"))
        assert d["path"] == f"/{lang}/go/luky/"
        assert d["title"], f"{lang}: шлюз без заголовка"
