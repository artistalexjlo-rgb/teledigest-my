# -*- coding: utf-8 -*-
"""Сторож звена 8 (PLAN.md §3.2): рендер берёт домен из `PSEO_SITE_CONFIG`, не хардкодит.

Один рендер на оба домена невозможен (`templates/base.html.j2` вшивает `site.domain` в
canonical/hreflang каждой страницы) — значит переключатель конфига обязан реально работать,
а `config/site_ru.py` обязан менять ТОЛЬКО домен/CTA, не расходиться с общим конфигом в
остальном (бренд/языки/telegram/год) — иначе два домена одного продукта незаметно разъедутся.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load_render(monkeypatch, site_config=None):
    """Свежая копия render.py — SITE выбирается при exec_module, кэш не годится."""
    if site_config is not None:
        monkeypatch.setenv("PSEO_SITE_CONFIG", site_config)
    else:
        monkeypatch.delenv("PSEO_SITE_CONFIG", raising=False)
    spec = importlib.util.spec_from_file_location(
        "render_test", os.path.join(HERE, "render.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_render_uses_the_online_domain(monkeypatch):
    """Без переменной окружения — прежнее поведение, `.online`."""
    render = _load_render(monkeypatch)
    assert render.SITE["domain"] == "https://info.multyspeak.online"


def test_env_var_switches_render_to_the_ru_domain(monkeypatch):
    """`PSEO_SITE_CONFIG=config.site_ru` даёт `.ru`-домен и свой CTA — звено 8 гоняет
    рендер дважды с разным значением, чтобы получить два дерева из одних данных."""
    render = _load_render(monkeypatch, "config.site_ru")
    assert render.SITE["domain"] == "https://info.multyspeak.ru"
    assert render.SITE["cta_luky_url"] == "https://multyspeak.ru"


def test_ru_config_only_overrides_domain_and_cta():
    """`site_ru.py` не расходится с общим конфигом ни в чём, кроме домена и CTA — иначе
    два домена одного продукта потихоньку разъедутся (бренд, языки, telegram, год)."""
    import config.site as site
    import config.site_ru as site_ru

    diff = {k for k in site.SITE if site.SITE.get(k) != site_ru.SITE.get(k)}
    assert diff == {"domain", "cta_luky_url"}, diff
