"""render.py — Фаза 1: чистый template-fill.

Берёт site-config (config/site.py) + i18n (i18n/{lang}.json) + page-data (json) →
отдаёт готовый HTML. Ноль runtime-логики, ноль обращений к Gemini/Qdrant.

CLI:
    python render.py data/ru_br_finance.json
        → пишет out/<path>/index.html

Смена ссылки/домена/бренда = правка config/site.py + повторный прогон рендера
(обновляет все страницы разом, без квот).
"""

import hashlib
import json
import pathlib
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
from config.site import SITE  # noqa: E402

_env = Environment(
    loader=FileSystemLoader(str(BASE / "templates")),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=False,
    lstrip_blocks=False,
)


def load_i18n(lang: str) -> dict:
    return json.loads((BASE / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def _pick(pool: list, seed: str):
    """Детерминированный выбор из пула по сид-строке (стабильно между сборками,
    варьируется между страницами; PS декоррелирован через свой суффикс)."""
    idx = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16) % len(pool)
    return pool[idx]


def build_cta(t: dict, page: dict) -> dict | None:
    """Собирает CTA-«бутер» из cta_pools: hook + assistant(L1) + voice(L2) + ps(оффтоп).
    Выбор слотов — по пути страницы (варьируем); PS — свой сид (оффтоп, не по теме)."""
    pools = t.get("cta_pools")
    if not pools:
        return None
    key = page.get("path", "")
    return {
        "hook": _pick(pools["hook"], key + "|hook"),
        "assistant_lead": pools["assistant_lead"],
        "assistant": _pick(pools["assistant"], key + "|assistant"),
        "voice_lead": pools["voice_lead"],
        "voice": _pick(pools["voice"], key + "|voice"),
        "ps": _pick(pools["ps"], key + "|ps"),
    }


def render_page(page: dict, lang: str | None = None) -> str:
    lang = lang or page.get("lang", "ru")
    t = load_i18n(lang)
    cta = build_cta(t, page)
    tmpl = _env.get_template(page.get("template", "page.html.j2"))
    html = tmpl.render(site=SITE, t=t, page=page, lang=lang, cta=cta)
    # Маркер #luky в текстах (интро/проза) → реальная дверь в продукт (единый источник — site.py).
    door = f'href="{SITE["cta_luky_url"]}" target="_blank" rel="noopener"'
    return html.replace("href='#luky'", door).replace('href="#luky"', door)


def build(data_path: str) -> pathlib.Path:
    page = json.loads(pathlib.Path(data_path).read_text(encoding="utf-8"))
    html = render_page(page)
    out = BASE / "out" / page["path"].strip("/") / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _indexable(page: dict) -> bool:
    """В sitemap попадает только то, что реально можно индексировать:
    не глобальный draft И не per-page noindex. Это и есть защита домена от
    тонких/непрошедших-гейт страниц (см. BUILDER_RULES / фаза публикации)."""
    return not SITE.get("draft") and not page.get("noindex")


def build_all(lastmod: str = "") -> dict:
    """Рендерит все data/*.json, пишет sitemap.xml (только indexable) + robots.txt.
    lastmod — ISO-дата для <lastmod> (freshness-сигнал); пустая → без тега.
    Возвращает {rendered, indexed, skipped_noindex}."""
    data_dir = BASE / "data"
    urls, n_rendered, n_noindex = [], 0, 0
    for jf in sorted(data_dir.glob("*.json")):
        page = json.loads(jf.read_text(encoding="utf-8"))
        if "path" not in page:
            continue  # не страница (конфиг/фикстура) — пропускаем
        build(str(jf))  # статьи (faqs) + хабы/главная/about (index-шаблон)
        n_rendered += 1
        if _indexable(page):
            urls.append((SITE["domain"] + page["path"], page.get("updated", "")))
        else:
            n_noindex += 1

    lm = f"\n    <lastmod>{lastmod}</lastmod>" if lastmod else ""
    body = "\n".join(f"  <url>\n    <loc>{u}</loc>{lm}\n  </url>" for u, _ in urls)
    sm = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}\n</urlset>\n"
    )
    (BASE / "out" / "sitemap.xml").write_text(sm, encoding="utf-8")
    robots = (
        "User-agent: *\nAllow: /\nDisallow: /landing/\n\n"
        f"Sitemap: {SITE['domain']}/sitemap.xml\n"
    )
    (BASE / "out" / "robots.txt").write_text(robots, encoding="utf-8")
    return {"rendered": n_rendered, "indexed": len(urls), "skipped_noindex": n_noindex}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        lm = sys.argv[2] if len(sys.argv) > 2 else ""
        stat = build_all(lastmod=lm)
        print(
            f"build_all: rendered={stat['rendered']} "
            f"indexed={stat['indexed']} noindex={stat['skipped_noindex']} "
            f"(draft={SITE.get('draft')})"
        )
    else:
        src = sys.argv[1] if len(sys.argv) > 1 else "data/ru_br_finance.json"
        path = build(src)
        print(f"rendered -> {path}")
