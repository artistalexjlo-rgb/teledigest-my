# -*- coding: utf-8 -*-
"""Сторож: `render.py::build_all()` не перерисовывает страницу, которой рисоваться
незачем (02.09 — измерено фактом: полный прогон на 46к+ страниц занимал 63с и гонялся
после КАЖДОЙ страны в массовом цикле, 89 из 90 раз впустую).

Три сценария, ОДНО правило: страница остаётся нетронутой, ТОЛЬКО если и её СОБСТВЕННЫЕ
данные не новее HTML, И набор языков её адреса не менялся — иначе языковой свитчер
(hreflang) у соседних страниц протух бы молча, ради экономии на диске.
"""

import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load_render(monkeypatch, out, data_dir, site_config=None):
    if site_config is not None:
        monkeypatch.setenv("PSEO_SITE_CONFIG", site_config)
    else:
        monkeypatch.delenv("PSEO_SITE_CONFIG", raising=False)
    monkeypatch.setenv("PSEO_OUT", out)
    monkeypatch.setenv("PSEO_DATA", data_dir)
    spec = importlib.util.spec_from_file_location(
        "render_incr", os.path.join(HERE, "render.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_site(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "sitebuild_incr", os.path.join(HERE, "site.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _korpus(tmp_path, monkeypatch, site_mod, lang, views):
    built = tmp_path / "built"
    os.makedirs(built / f"out_facet_{lang}", exist_ok=True)
    with open(built / f"out_facet_{lang}" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump({"geo": "gr", "views_by_task": views, "shelves": []}, fh)
    monkeypatch.setattr(site_mod, "BUILT", str(built))
    monkeypatch.setattr(site_mod, "DATA", str(tmp_path / "data"))


def _view(zadacha, adres, tema="visa", n=3, text="advice"):
    return {
        "title": zadacha,
        "theme": tema,
        "slug": adres,
        "branch": adres,
        "part": 1,
        "parts": 1,
        "items": [
            {"id": f"h{i}", "text": f"{text} {i}. tail", "n": 1} for i in range(n)
        ],
    }


def _html_of(out_dir, path_rel):
    return open(os.path.join(out_dir, path_rel), encoding="utf-8").read()


def _mtime_of(out_dir, path_rel):
    return os.path.getmtime(os.path.join(out_dir, path_rel))


PAGE_REL = "ru/gr/visa/visa-documents/index.html"


def test_unchanged_page_is_left_untouched_on_second_run(tmp_path, monkeypatch):
    site_mod = _load_site(monkeypatch)
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "ru",
        [_view("visa documents", "visa-documents")],
    )
    site_mod.build_all("ru")

    out = str(tmp_path / "out")
    data_dir = str(tmp_path / "data")
    render_mod = _load_render(monkeypatch, out, data_dir)

    stat1 = render_mod.build_all()
    assert stat1["rendered"] >= 1
    mtime1 = _mtime_of(out, PAGE_REL)

    time.sleep(0.05)
    stat2 = render_mod.build_all()

    assert stat2["skipped"] >= 1
    mtime2 = _mtime_of(out, PAGE_REL)
    assert mtime2 == mtime1, "непотревоженная страница не должна была переписаться"


def test_changed_data_forces_a_rerender(tmp_path, monkeypatch):
    site_mod = _load_site(monkeypatch)
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "ru",
        [_view("visa documents", "visa-documents")],
    )
    site_mod.build_all("ru")

    out = str(tmp_path / "out")
    data_dir = str(tmp_path / "data")
    render_mod = _load_render(monkeypatch, out, data_dir)
    render_mod.build_all()
    mtime1 = _mtime_of(out, PAGE_REL)
    html1 = _html_of(out, PAGE_REL)

    time.sleep(0.05)
    # реальный сценарий: перевод/сборка переписали корпус свежим текстом
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "ru",
        [_view("visa documents", "visa-documents", text="EDITED advice")],
    )
    site_mod.build_all("ru")

    stat2 = render_mod.build_all()
    assert stat2["rendered"] >= 1
    mtime2 = _mtime_of(out, PAGE_REL)
    html2 = _html_of(out, PAGE_REL)
    assert mtime2 != mtime1
    assert "EDITED advice" in html2
    assert "EDITED advice" not in html1


def test_a_new_language_for_a_tail_rerenders_the_whole_language_group(
    tmp_path, monkeypatch
):
    """Ключевой сценарий: у ru-страницы появляется en-версия того же адреса — сама
    ru-страница СВОИ данные не меняла, но должна перерисоваться, чтобы её hreflang
    узнал про свежий en (иначе свитчер языков молча отстанет)."""
    site_mod = _load_site(monkeypatch)
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "ru",
        [_view("visa documents", "visa-documents")],
    )
    site_mod.build_all("ru")

    out = str(tmp_path / "out")
    data_dir = str(tmp_path / "data")
    render_mod = _load_render(monkeypatch, out, data_dir)
    render_mod.build_all()
    mtime_ru_1 = _mtime_of(out, PAGE_REL)
    html_ru_1 = _html_of(out, PAGE_REL)
    assert 'hreflang="en"' not in html_ru_1, "en ещё не существует — упоминать его рано"

    time.sleep(0.05)
    # en появился — ru-корпус НЕ ТРОГАЕМ вообще, только добавляем en рядом
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "en",
        [_view("visa documents", "visa-documents")],
    )
    site_mod.build_all("en")

    stat2 = render_mod.build_all()
    en_rel = "en/gr/visa/visa-documents/index.html"
    assert os.path.exists(
        os.path.join(out, en_rel)
    ), "новая en-страница обязана появиться"

    mtime_ru_2 = _mtime_of(out, PAGE_REL)
    html_ru_2 = _html_of(out, PAGE_REL)
    assert mtime_ru_2 != mtime_ru_1, "ru должна была перерисоваться из-за нового соседа"
    assert 'hreflang="en"' in html_ru_2, "свитчер обязан узнать про свежий en"
    assert stat2["rendered"] >= 2, "и en, и ru — обе перерисованы в этом прогоне"


def test_manifest_file_does_not_leak_into_readycheck_language_scan(
    tmp_path, monkeypatch
):
    """`.tails.json` лежит в корне OUT — `readycheck.scan()` считает языками только
    директории, файл не должен путаться под ногами."""
    site_mod = _load_site(monkeypatch)
    _korpus(
        tmp_path,
        monkeypatch,
        site_mod,
        "ru",
        [_view("visa documents", "visa-documents")],
    )
    site_mod.build_all("ru")

    out = str(tmp_path / "out")
    data_dir = str(tmp_path / "data")
    render_mod = _load_render(monkeypatch, out, data_dir)
    render_mod.build_all()

    assert os.path.exists(os.path.join(out, ".tails.json"))
    assert not os.path.isdir(os.path.join(out, ".tails.json"))
