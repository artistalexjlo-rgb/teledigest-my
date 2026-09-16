# -*- coding: utf-8 -*-
"""Сторож виджета помощника (16.09): есть на страницах страны, нет на служебных,
контекст — английское имя страны; переводы интерфейса симметричны по 14 языкам;
правка шаблона перерисовывает существующие страницы."""

import json
import os
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


def test_widget_on_country_page_with_english_country(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(_page())
    assert 'id="ask"' in html
    assert 'data-country="Greece"' in html, "контекст помощника — английское имя страны"
    assert 'data-page="Документы на визу"' in html
    assert 'data-api="/api/assistant/ask"' in html
    assert "Спроси Luky об этой стране" in html


def test_no_widget_on_pages_without_country(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(
        _page(path="/ru/about/", geo="", template="index.html.j2", body="о проекте")
    )
    assert 'id="ask"' not in html


def test_no_widget_on_find_page(monkeypatch, tmp_path):
    render = _render(monkeypatch, tmp_path)
    html = render.render_page(
        _page(path="/ru/find/", template="find.html.j2", noindex=True)
    )
    assert 'id="ask"' not in html


def test_i18n_keys_are_symmetric_across_languages():
    files = sorted((HERE / "i18n").glob("*.json"))
    assert len(files) == 14, [f.name for f in files]
    keysets = {f.stem: set(json.loads(f.read_text(encoding="utf-8"))) for f in files}
    ref = keysets["en"]
    for lang, ks in keysets.items():
        assert ks == ref, f"{lang}: лишние {sorted(ks - ref)}, нет {sorted(ref - ks)}"
    for k in ("ask_title", "ask_ph", "ask_btn", "ask_wait", "ask_err", "ask_more"):
        assert k in ref


def test_layout_change_forces_rerender(monkeypatch, tmp_path):
    """⛔ Инкремент смотрел только на данные: правка шаблона не перерисовала бы ни одной
    существующей страницы. Отпечаток шаблонов/переводов/статики — часть решения."""
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
    st3 = render.build_all()
    assert st3["rendered"] == 1, "шаблон изменился — страница обязана перерисоваться"
    out = tmp_path / "out"
    assert (out / ".layout_version").read_text(encoding="utf-8") == "changed"
