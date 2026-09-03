# -*- coding: utf-8 -*-
"""Сторож звена 8, шаг 2 (PLAN.md §3.2): render.py --all ДВАЖДЫ на одних данных даёт
два честных дерева, не одно наполовину.

`test_render_domain.py` проверяет только переключатель SITE-модуля (импорт), не реальный
HTML. Здесь — сквозной прогон: те же `data/*.json` (из настоящего `site.build_all`, не
рукописная фикстура — схему страницы решает сборщик, не угадывание) идут через `render.py`
дважды, с `config.site` и `config.site_ru`. Гейт: домен/CTA в HTML различаются, текст
советов — нет.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name, filename, monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _korpus(tmp_path, monkeypatch, site_mod, views):
    built = tmp_path / "built"
    os.makedirs(built / "out_facet_ru", exist_ok=True)
    import json

    with open(built / "out_facet_ru" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump({"geo": "gr", "views_by_task": views, "shelves": []}, fh)
    monkeypatch.setattr(site_mod, "BUILT", str(built))
    monkeypatch.setattr(site_mod, "DATA", str(tmp_path / "data"))


def _view(zadacha, adres, tema="visa", n=3):
    return {
        "title": zadacha,
        "theme": tema,
        "slug": adres,
        "branch": adres,
        "part": 1,
        "parts": 1,
        "items": [
            {"id": f"h{i}", "text": f"advice {i} about {adres}. tail", "n": 1}
            for i in range(n)
        ],
    }


def test_two_render_passes_over_the_same_data_split_domain_but_keep_the_text(
    tmp_path, monkeypatch
):
    site_mod = _load("sitebuild_two", "site.py", monkeypatch)
    _korpus(
        tmp_path, monkeypatch, site_mod, [_view("visa documents", "visa-documents")]
    )
    site_mod.build_all("ru")

    data_dir = str(tmp_path / "data")
    out_online = str(tmp_path / "out_online")
    out_ru = str(tmp_path / "out_ru")

    render_online = _load(
        "render_online",
        "render.py",
        monkeypatch,
        PSEO_SITE_CONFIG=None,
        PSEO_OUT=out_online,
        PSEO_DATA=data_dir,
    )
    render_online.build_all()

    render_ru = _load(
        "render_ru",
        "render.py",
        monkeypatch,
        PSEO_SITE_CONFIG="config.site_ru",
        PSEO_OUT=out_ru,
        PSEO_DATA=data_dir,
    )
    render_ru.build_all()

    page_rel = "ru/gr/visa/visa-documents/index.html"
    html_online = open(os.path.join(out_online, page_rel), encoding="utf-8").read()
    html_ru = open(os.path.join(out_ru, page_rel), encoding="utf-8").read()

    # домен разъехался туда, куда и должен
    assert "https://info.multyspeak.online" in html_online
    assert "https://info.multyspeak.ru" not in html_online
    assert "https://info.multyspeak.ru" in html_ru
    assert "https://info.multyspeak.online" not in html_ru

    # а текст советов — тот же самый, не пересочинён вторым прогоном
    for i in range(3):
        needle = f"advice {i} about visa-documents. tail"
        assert needle in html_online, html_online
        assert needle in html_ru, html_ru

    # шлюз в продукт тоже ведёт на СВОЙ домен, не общий
    gate_rel = "ru/go/luky/index.html"
    gate_online = open(os.path.join(out_online, gate_rel), encoding="utf-8").read()
    gate_ru = open(os.path.join(out_ru, gate_rel), encoding="utf-8").read()
    assert "https://multyspeak.online" in gate_online
    assert "https://multyspeak.ru" in gate_ru
