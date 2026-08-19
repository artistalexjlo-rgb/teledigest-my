"""render.py — Фаза 1: чистый template-fill.

Берёт site-config (config/site.py) + i18n (i18n/{lang}.json) + page-data (json) →
отдаёт готовый HTML. Ноль runtime-логики, ноль обращений к Gemini/Qdrant.

CLI:
    python render.py data/ru_br_finance.json
        → пишет $PSEO_OUT/<path>/index.html (по умолчанию out/)

Смена ссылки/домена/бренда = правка config/site.py + повторный прогон рендера
(обновляет все страницы разом, без квот).
"""

import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
from html import escape as html_escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
from config.site import SITE  # noqa: E402

# ⭐ ЕДИНСТВЕННОЕ определение каталога вывода. `readycheck.py` и `ship.py` берут его отсюда
# импортом и своего не заводят: до 2026-08-11 `BASE/out` было выписано в трёх местах, и при
# переносе публикации в пульт рендер писал бы в примонтированный каталог, а гейт проверял
# пустой `/app/out` и рапортовал «готово 0» при собранном сайте. Одно знание — одно место.
# PSEO_OUT нужен именно пульту: писать сразу в маунт, а не копировать после рендера.
OUT = pathlib.Path(os.environ.get("PSEO_OUT") or (BASE / "out"))

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


# ── ДОВОД CTA ПО РАЗДЕЛУ (канон §0.12, шаг 6) ──
# Замер до правила: слот голоса выбирался ТОЛЬКО по хешу пути, поэтому на странице про сроки
# визы человеку обещали «и официант не перепутает заказ». Раздел даёт адресность.
# Адресность задаётся картой ИНДЕКСОВ, а не отдельным текстом на каждый язык: пулы CTA в
# 14 языках ПАРАЛЛЕЛЬНЫ по индексу (длину держит `test_lang_complete`). Поэтому одна карта
# работает на все языки, а строки живут там, где и весь копирайт, — в `i18n/<язык>.json`.
VOICE_BY_SHELF = {
    "tourism": [1, 4, 5, 6, 7],  # официант, гид, лобби, дорога, ресепшен
    "transport": [2, 6],  # таксист, спросить дорогу
    "shopping": [3, 9],  # рынок, магазин
    "health": [8],  # врач
    "housing": [0, 7],  # местные, ресепшен
    # Дописаны 14.08 во ВСЕ 14 языков одним порядком (пулы index-параллельны).
    # До этого 8 разделов из 13 брали нейтральную строку, а в них лежит 1 074 разбора
    # из 1 889 — 57% страниц сайта, то есть «нейтрально» было решением по умолчанию для
    # большинства, а не для остатка.
    "visa": [10],  # консульство
    "border": [11],  # погранконтроль
    "docs": [12],  # нотариус и конторы
    "finance": [13],  # банк
    "customs": [14],  # досмотр
    "digital": [15],  # салон связи
    "work": [16],  # работодатель, деканат
    "safety": [17],  # полиция, скорая
}
# Своя строка теперь есть у ВСЕХ 13 разделов. `VOICE_ANY` остался для страниц, у которых
# раздела нет вовсе (сборная метка — брак нарезки, 10 на корпус): там нужна фраза, верная
# всегда, а не случайная из всех — случайная и рождала «официанта» на визовой странице.
VOICE_ANY = [0, 6]  # «объяснись с местными», «спроси дорогу и пойми ответ»


def voice_pool(pools: dict, page: dict) -> list:
    idx = VOICE_BY_SHELF.get(page.get("shelf_key") or "", VOICE_ANY)
    lines = [pools["voice"][i] for i in idx if i < len(pools["voice"])]
    return lines or pools["voice"]  # пул короче ожидаемого → ведём себя как раньше


def door_url(page: dict) -> str:
    """Дверь в продукт — ЧЕРЕЗ ШЛЮЗ `/<язык>/go/luky/`, чтобы переход был посчитан.

    Замер до шага 6: статистики переходов на Luky не было ни одной цифры. Шлюз несёт
    `?geo=&shelf=`, а nginx уже пишет строку запроса и Referer — значит видно, какая страна
    и какой раздел отдают переходы, и своего бэкенда для этого не нужно.

    Сам шлюз ведёт в продукт НАПРЯМУЮ: иначе страница слала бы на саму себя.
    """
    if "/go/luky/" in (page.get("path") or ""):
        return SITE["cta_luky_url"]
    lang = page.get("lang", "ru")
    q = [("geo", page.get("geo") or ""), ("shelf", page.get("shelf_key") or "")]
    tail = urllib.parse.urlencode([(k, v) for k, v in q if v])
    return f"/{lang}/go/luky/" + (f"?{tail}" if tail else "")


def build_cta(t: dict, page: dict) -> dict | None:
    """Собирает CTA-«бутер» из cta_pools: hook + assistant(L1) + voice(L2) + ps(оффтоп).
    Слоты варьируем по пути страницы; голос — по РАЗДЕЛУ; PS — свой сид (оффтоп, не по теме).
    """
    pools = t.get("cta_pools")
    if not pools:
        return None
    key = page.get("path", "")
    return {
        "hook": _pick(pools["hook"], key + "|hook"),
        "assistant_lead": pools["assistant_lead"],
        "assistant": _pick(pools["assistant"], key + "|assistant"),
        "voice_lead": pools["voice_lead"],
        "voice": _pick(voice_pool(pools, page), key + "|voice"),
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
    src, dst = BASE / "static", OUT / "assets"
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*")):
        (dst / f.name).write_bytes(f.read_bytes())
        n += 1
    return n


def index_paths(data_dir=None) -> set[str]:
    """Пути всех собранных страниц — по ним и только по ним объявляем альтернативы.

    `data_dir` нужен сторожу: подменять `BASE` нельзя — от него же берутся i18n, шаблоны
    и ассеты, и тест ломался бы на них, а не проверял правило.
    """
    global _PATHS
    _PATHS = set()
    for jf in pathlib.Path(data_dir or (BASE / "data")).glob("*.json"):
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
    url = door_url(page)  # ОДНА дверь на страницу: и кнопка, и маркеры #luky в текстах
    html = tmpl.render(
        site=SITE,
        t=t,
        page=page,
        lang=lang,
        cta=cta,
        door=url,
        text_dir=text_dir(lang),
        alt_langs=alt_langs(page),
        asset_v=asset_version(),
    )
    # Маркер #luky в текстах (интро/проза) → та же дверь, что у кнопки. Единый источник —
    # `door_url`: раньше здесь стоял прямой адрес продукта, и переходы из текста не считались.
    # ⛔ `&` в атрибуте обязан быть `&amp;`: шаблон экранирует сам, а эта подстановка идёт
    # мимо Jinja, и без экранирования одна и та же ссылка выходила в двух разных видах.
    door = f'href="{html_escape(url)}" target="_blank" rel="noopener"'
    return html.replace("href='#luky'", door).replace('href="#luky"', door)


def build(data_path: str) -> pathlib.Path:
    page = json.loads(pathlib.Path(data_path).read_text(encoding="utf-8"))
    html = render_page(page)
    out = OUT / page["path"].strip("/") / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _indexable(page: dict) -> bool:
    """В sitemap попадает только то, что реально можно индексировать:
    не глобальный draft И не per-page noindex. Это и есть защита домена от
    тонких/непрошедших-гейт страниц (см. BUILDER_RULES / фаза публикации)."""
    return not SITE.get("draft") and not page.get("noindex")


def build_all(lastmod: str = "", data_dir=None) -> dict:
    """Рендерит все data/*.json, пишет sitemap.xml (только indexable) + robots.txt.
    lastmod — ISO-дата для <lastmod> (freshness-сигнал); пустая → без тега.
    data_dir — только для сторожей (боевой прогон берёт `BASE/data`).
    Возвращает {rendered, indexed, skipped_noindex, assets, search_titles}."""
    data_dir = pathlib.Path(data_dir or (BASE / "data"))
    index_paths(
        data_dir
    )  # ДО рендера: hreflang опирается на собранное, а не на догадку
    n_assets = copy_assets()  # общие CSS/JS: один файл на сайт вместо копии в странице
    urls, n_rendered, n_noindex = [], 0, 0
    search = {}  # язык → [[заголовок, адрес], …] для поиска по заголовкам (шаг 7)
    for jf in sorted(data_dir.glob("*.json")):
        page = json.loads(jf.read_text(encoding="utf-8"))
        if "path" not in page:
            continue  # не страница (конфиг/фикстура) — пропускаем
        build(str(jf))  # статьи (faqs) + хабы/главная/about (index-шаблон)
        n_rendered += 1
        # ⭐ ИНДЕКС ПОИСКА СОБИРАЕТСЯ ИЗ ТОГО, ЧТО РЕАЛЬНО ОТРЕНДЕРИЛОСЬ, а не из корпуса:
        # иначе он обещал бы страницы, которые отсеялись (без адреса, метка не перевелась,
        # гео пустое) — то есть поиск вёл бы в 404. Служебные страницы (`noindex`: шлюз,
        # сама страница поиска) в индекс не идут.
        title = page.get("search_title") or page.get("intent_name") or page.get("h1")
        if title and page.get("lang") and not page.get("noindex"):
            search.setdefault(page["lang"], []).append([title, page["path"]])
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
    (OUT / "sitemap.xml").write_text(sm, encoding="utf-8")
    robots = (
        "User-agent: *\nAllow: /\nDisallow: /landing/\n\n"
        f"Sitemap: {SITE['domain']}/sitemap.xml\n"
    )
    (OUT / "robots.txt").write_text(robots, encoding="utf-8")
    # ⭐ ПОИСК ПО ЗАГОЛОВКАМ — ОДИН ФАЙЛ НА ЯЗЫК, тянется по первому нажатию (шаг 7).
    # Замер на настоящей сборке: `ru` — 3124 заголовка, 352 КБ, в gzip 77 КБ (nginx жмёт
    # `application/json`… точнее — жмёт по `gzip_types`, поэтому тип добавлен в конфиг).
    # Инлайнить в каждую страницу нельзя: это те же 352 КБ × 41 630 страниц.
    n_search = 0
    for lang, rows in sorted(search.items()):
        rows.sort(key=lambda r: r[0].lower())
        p = OUT / lang / "search.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        n_search += len(rows)
    return {
        "rendered": n_rendered,
        "indexed": len(urls),
        "skipped_noindex": n_noindex,
        "assets": n_assets,
        "search_titles": n_search,
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
