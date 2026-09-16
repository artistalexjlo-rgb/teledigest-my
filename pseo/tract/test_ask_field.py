# -*- coding: utf-8 -*-
"""Сторожа поля-помощника, главной и favicon (17.09).

- одно поле наверху на КАЖДОЙ странице (кроме страницы поиска), нижнего виджета нет;
- контекст страны — английское имя, на главной пусто;
- главная: h1 из i18n, «Популярные» с вайбом/подписью, регионы аккордеоном;
- CTA — одна строка про голос + кнопка, крючков/PS нет;
- переводы симметричны по 14 языкам, имена регионов покрывают все регионы;
- favicon.svg попадает в снимок; правка шаблона перерисовывает страницы.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _render(monkeypatch, tmp_path):
    monkeypatch.setenv("PSEO_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("PSEO_DATA", str(tmp_path / "data"))
    for m in ("render", "config.site"):
        sys.modules.pop(m, None)
    import render  # noqa: E402

    return render


def _page(**kw):
    p = {
        "path": "/ru/gr/visa/visa-documents/",
        "lang": "ru",
        "geo": "gr",
        "geo_name": "Греция",
        "theme": "visa",
        "h1": "Документы на визу",
        "title": "Документы на визу — Греция",
        "meta_desc": "",
        "intent_name": "Документы на визу",
        "faqs": [{"q": "Что нужно", "a": "Паспорт"}],
        "updated": "2026-09-16",
    }
    p.update(kw)
    return p


def test_one_field_on_top_with_country_context(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(_page())
    assert html.count('id="gq"') == 1 and html.count('id="ask"') == 1
    assert 'data-country="Greece"' in html
    assert 'data-page="Документы на визу"' in html
    assert "Найди или спроси…" in html
    assert 'id="askForm"' not in html, "нижнего виджета больше нет"


def test_field_is_on_home_without_country(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(
        _page(
            path="/ru/",
            geo="",
            template="index.html.j2",
            home=True,
            popular=[],
            regions=[],
        )
    )
    assert 'id="ask"' in html and 'data-country=""' in html


def test_find_page_keeps_its_own_field(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(
        _page(path="/ru/find/", template="find.html.j2", noindex=True)
    )
    assert 'id="ask"' not in html and 'id="fq"' in html


def test_home_shows_popular_with_vibe_and_regions(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    page = _page(
        path="/ru/",
        geo="",
        template="index.html.j2",
        home=True,
        popular=[
            {
                "flag": "🇬🇷",
                "name": "Греция",
                "url": "/ru/gr/",
                "n": 40,
                "vibe": "Острова и руины",
            },
            {"flag": "🇦🇪", "name": "ОАЭ", "url": "/ru/ae/", "n": 1, "vibe": ""},
        ],
        regions=[
            {
                "key": "europe",
                "geos": [
                    {
                        "flag": "🇬🇷",
                        "name": "Греция",
                        "url": "/ru/gr/",
                        "n": 40,
                        "vibe": "",
                    }
                ],
            },
            {
                "key": "mideast",
                "geos": [
                    {"flag": "🇦🇪", "name": "ОАЭ", "url": "/ru/ae/", "n": 1, "vibe": ""}
                ],
            },
        ],
    )
    html = render.render_page(page)
    assert "<h1>Куда едешь?</h1>" in html
    assert "Острова и руины · 40" in html, "вайб + число страниц"
    assert "живой опыт · 1" in html, "нет вайба — подпись из i18n"
    assert "Европа" in html and "Ближний Восток" in html
    assert html.index("Греция") < html.index("Европа"), "популярные выше регионов"
    assert "href='#luky'" not in html and "/ru/go/luky/" in html, "интро ведёт в дверь"


def test_luky_plate_is_on_top_with_field_phrases_and_button(monkeypatch, tmp_path):
    """Эскиз 17.09: плашка Luky НАВЕРХУ — поле, крючок, помощник, голос, PS, кнопка;
    итоги (подсказки/ответ) — ПОД плашкой, над содержимым; внизу плашки нет."""
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(_page())
    plate_at = html.index('class="plate cta luky"')
    assert plate_at < html.index("<h1>"), "плашка выше содержимого"
    plate = html[plate_at : html.index("</div></div>", plate_at)]
    assert 'id="gq"' in plate and "Спросить" in plate
    assert "<h2>" in plate and 'class="ps"' in plate
    assert "Спроси Luky —" in plate and "голосовой переводчик" in plate
    assert "Перейти в приложение Luky" in plate
    log_at = html.index('id="askLog"')
    assert plate_at < log_at < html.index("<h1>"), "итоги под плашкой, над содержимым"
    assert html.count('class="plate cta') == 1, "внизу плашки нет"


def test_i18n_keys_are_symmetric_and_regions_named():
    files = sorted((HERE / "i18n").glob("*.json"))
    assert len(files) == 14, [f.name for f in files]
    data = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}
    ref = set(data["en"])
    for lang, d in data.items():
        assert (
            set(d) == ref
        ), f"{lang}: лишние {sorted(set(d) - ref)}, нет {sorted(ref - set(d))}"
        assert set(d["cta_pools"]) == {
            "hook",
            "assistant_lead",
            "assistant",
            "voice_lead",
            "voice",
            "ps",
        }, lang
    for k in (
        "ask_ph",
        "ask_btn",
        "ask_wait",
        "ask_err",
        "home_h1",
        "home_intro",
        "geo_blurb",
        "region_names",
    ):
        assert k in ref, k
    for gone in ("cta_hook", "ask_title", "ask_more", "home_search_ph"):
        assert gone not in ref, gone
    home = json.loads((HERE / "home.json").read_text(encoding="utf-8"))
    for lang, d in data.items():
        assert set(d["region_names"]) >= set(home["region_order"]), lang


def test_site_home_ranks_popular_and_groups_regions(monkeypatch):
    import importlib.util

    # `site` — имя стандартного модуля Python; грузим файл тракта по пути, как соседи
    spec = importlib.util.spec_from_file_location("sitebuild_home", HERE / "site.py")
    site_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site_mod)

    _, page = site_mod.home([("ae", 1), ("gr", 40), ("br", 261), ("xx", 2)], "ru")
    assert page["home"] is True and page["template"] == "index.html.j2"
    assert [c["url"] for c in page["popular"]][:2] == ["/ru/br/", "/ru/gr/"]
    assert page["popular"][1]["vibe"].startswith("Острова")
    keys = [r["key"] for r in page["regions"]]
    assert keys == ["europe", "mideast", "latam", "other"], keys
    assert [
        c["url"] for r in page["regions"] if r["key"] == "other" for c in r["geos"]
    ] == ["/ru/xx/"]


def test_home_prefers_bought_vibes_from_the_volume(tmp_path):
    """Главная читает купленные вайбы с тома (`vibes.json`), сид ru — запасной."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sitebuild_vibes", HERE / "site.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    (tmp_path / "vibes.json").write_text(
        json.dumps({"ru": {"gr": "сид"}, "de": {"gr": "Inseln und Ruinen"}}),
        encoding="utf-8",
    )
    sm.VIBES_FILE = str(tmp_path / "vibes.json")
    _, de = sm.home([("gr", 40)], "de")
    assert de["popular"][0]["vibe"] == "Inseln und Ruinen"
    _, ru = sm.home([("gr", 40)], "ru")
    assert ru["popular"][0]["vibe"] == "сид", "том важнее сида и для ru"
    sm.VIBES_FILE = str(tmp_path / "нет.json")
    _, ru2 = sm.home([("gr", 40)], "ru")
    assert ru2["popular"][0]["vibe"].startswith(
        "Острова"
    ), "нет тома — сид из home.json"


def test_favicon_lands_in_the_snapshot(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir()
    render.build_all()
    assert (
        tmp_path / "out" / "favicon.svg"
    ).is_file(), "favicon 404 с 26.08 — не копировался"


def test_layout_change_forces_rerender(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "ru_gr_visa.json").write_text(
        json.dumps(_page(), ensure_ascii=False), encoding="utf-8"
    )
    st1 = render.build_all()
    st2 = render.build_all()
    assert st1["rendered"] == 1 and st2["rendered"] == 0, (st1, st2)
    monkeypatch.setattr(render, "layout_version", lambda: "changed")
    assert render.build_all()["rendered"] == 1


def test_root_index_picks_browser_language_and_lists_all(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    (tmp_path / "data").mkdir()
    render.build_all()
    html = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    for lang in render.SITE["languages"]:
        assert f'href="/{lang}/"' in html, lang
    assert "navigator.languages" in html and 'location.replace("/en/")' in html
