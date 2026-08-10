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


# Языки, которые пишутся справа налево. Свойство ПИСЬМЕННОСТИ, а не текста, поэтому
# живёт в коде, а не в i18n: переводчику тут нечего решать. Без `dir="rtl"` арабская
# страница верстается слева направо — пунктуация и цифры встают не на свои места.
RTL_LANGS = {"ar", "he", "fa", "ur"}


def text_dir(lang: str) -> str:
    return "rtl" if lang in RTL_LANGS else "ltr"


# ⛔ hreflang и свитчер обязаны знать, в каких языках страница СУЩЕСТВУЕТ, а не только
# что хвост адреса общий. `shared_tail` отвечал на второй вопрос и этого мало: вид может
# выпасть в отдельном языке (метка не перевелась, после отсева осталось <4 пункта). Замер
# 08.08: 118 битых ссылок — все из hreflang и свитчера, вида «есть в ar, нет в pt».
# Индекс строится ОДИН раз из data/ и хранится в модуле; при рендере одиночного файла
# (render.py <файл>) он пуст, и тогда падаем на прежнее поведение по `shared_tail`.
_PATHS: set[str] = set()


# ⭐ ОБЩИЕ АССЕТЫ (2026-08-09). 72% каждой страницы было одинаковой обвязкой, повторённой
# 41 632 раза: CSS 10.8 КБ + октагон-скрипт 1.7 КБ инлайн в КАЖДОМ файле. Итого 1.50 ГБ,
# из них ~520 МБ — копии одного и того же. Это не только вес репозитория: РФ-сервер собирал
# из него Docker-образ и лёг по памяти, унеся с собой ВПН, переводчик и БД (09.08).
# Теперь один файл на сайт: браузер кеширует его один раз, а не тянет копию на каждой
# странице. Тематический и поисковый скрипты остались инлайн — в них подставляются
# переводы и данные страницы.
# `?v=` — хеш содержимого: адрес меняется при правке ассета, иначе Cloudflare отдавал бы
# старый файл из кеша.
def asset_version() -> str:
    h = hashlib.sha1()
    d = BASE / "static"
    for f in sorted(d.glob("*")) if d.is_dir() else []:
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


def copy_assets() -> int:
    """static/ → out/assets/. Возвращает число файлов."""
    src, dst = BASE / "static", BASE / "out" / "assets"
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*")):
        (dst / f.name).write_bytes(f.read_bytes())
        n += 1
    return n


def index_paths() -> set[str]:
    """Пути всех собранных страниц — по ним и только по ним объявляем альтернативы."""
    global _PATHS
    _PATHS = set()
    for jf in (BASE / "data").glob("*.json"):
        try:
            p = json.loads(jf.read_text(encoding="utf-8")).get("path")
        except Exception:
            continue
        if p:
            _PATHS.add(p)
    return _PATHS


def alt_langs(page: dict) -> list:
    """Языки, в которых ЭТА страница есть. Пустой список = альтернатив не объявляем."""
    if not page.get("shared_tail"):
        return []
    tail = "/".join(page["path"].split("/")[2:])
    if not _PATHS:  # одиночный рендер: индекса нет, доверяем shared_tail как раньше
        return list(SITE["languages"])
    return [x for x in SITE["languages"] if f"/{x}/{tail}" in _PATHS]


def render_page(page: dict, lang: str | None = None) -> str:
    lang = lang or page.get("lang", "ru")
    t = load_i18n(lang)
    cta = build_cta(t, page)
    tmpl = _env.get_template(page.get("template", "page.html.j2"))
    html = tmpl.render(
        site=SITE,
        t=t,
        page=page,
        lang=lang,
        cta=cta,
        text_dir=text_dir(lang),
        alt_langs=alt_langs(page),
        asset_v=asset_version(),
    )
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
    index_paths()  # ДО рендера: hreflang опирается на состав собранного, а не на догадку
    n_assets = copy_assets()  # общие CSS/JS: один файл на сайт вместо копии в странице
    urls, n_rendered, n_noindex = [], 0, 0
    for jf in sorted(data_dir.glob("*.json")):
        page = json.loads(jf.read_text(encoding="utf-8"))
        if "path" not in page:
            continue  # не страница (конфиг/фикстура) — пропускаем
        build(str(jf))  # статьи (faqs) + хабы/главная/about (index-шаблон)
        n_rendered += 1
        if _indexable(page):
            # ⚠️ Именно `updated_iso`, НЕ `updated`: второе — подпись в подвале в формате
            # MM.YYYY («08.2026»), и sitemap такую дату не принимает. Фикстура
            # test_lastmod.py на этом и поймала: в карту уезжало `<lastmod>07.2026</lastmod>`.
            urls.append((SITE["domain"] + page["path"], page.get("updated_iso", "")))
        else:
            n_noindex += 1

    # ⭐ lastmod ПОСТРАНИЧНО (2026-08-07). Было: одна дата из аргумента командной строки на
    # ВСЕ адреса — кто-то вписал `2026-07-06`, и она месяц ехала во все 2185, то есть месяц
    # говорила Google «здесь ничего не менялось». Теперь дату несёт сама страница
    # (`updated_iso`, ставит pages.py и только при РЕАЛЬНОМ изменении содержимого).
    # Аргумент остался запасным: у старых data-файлов поля нет.
    def _lm(iso):
        d = iso or lastmod
        return f"\n    <lastmod>{d}</lastmod>" if d else ""

    body = "\n".join(
        f"  <url>\n    <loc>{u}</loc>{_lm(iso)}\n  </url>" for u, iso in urls
    )
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
    return {
        "rendered": n_rendered,
        "indexed": len(urls),
        "skipped_noindex": n_noindex,
        "assets": n_assets,
    }


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
