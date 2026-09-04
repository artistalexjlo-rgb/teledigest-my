"""render.py — Фаза 1: чистый template-fill.

Берёт site-config (config/site.py) + i18n (i18n/{lang}.json) + page-data (json) →
отдаёт готовый HTML. Ноль runtime-логики, ноль обращений к Gemini/Qdrant.

CLI:
    python render.py data/ru_gr_visa.json
        → пишет $PSEO_OUT/<path>/index.html (по умолчанию out/)

Смена ссылки/домена/бренда = правка config/site.py + повторный прогон рендера
(обновляет все страницы разом, без квот).

⛔ Модуль живёт в `pseo/tract/` — переписан заново, не перенесён (юзер 28.08: переезжает
только написанное свежее под план пульта). Снято при переписи: ссылка на `BUILDER_RULES`
(документ в `pseo/legacy/`, не действует) и демо-заглушка `br_finance` из давно прошедшей
эпохи в дефолте CLI.

⭐ ДВА ДОМЕНА, ОДИН РЕНДЕР (звено 8, PLAN.md §3.2). `.ru`-зеркало несёт свой домен и CTA
прямо в HTML (canonical/hreflang) — общий сайт для обоих доменов собрать нельзя. Какой
конфиг брать — говорит `PSEO_SITE_CONFIG` (модуль с `SITE = {...}`, дефолт `config.site`),
второй прогон рендера с `PSEO_SITE_CONFIG=config.site_ru` даёт `.ru`-дерево из тех же
`data/*.json` — данные (звено 5, `site.py`) от домена не зависят.
"""

import hashlib
import importlib
import json
import os
import pathlib
import sys
import urllib.parse
from html import escape as html_escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = pathlib.Path(__file__).parent  # …/pseo/tract — templates/, i18n/, config/,
# static/ переехали сюда же 28.08 (юзер: «одна папка истина — тракт»); ROOT сняли,
# второго уровня над ними больше нет.
sys.path.insert(0, str(HERE))
SITE = importlib.import_module(os.environ.get("PSEO_SITE_CONFIG", "config.site")).SITE

# ⭐ ЕДИНСТВЕННОЕ определение каталога вывода. `readycheck.py` берёт его отсюда импортом
# и своего не заводит: до 2026-08-11 `BASE/out` было выписано в трёх местах, и при переносе
# публикации в пульт рендер писал бы в примонтированный каталог, а гейт проверял пустой
# `/app/out` и рапортовал «готово 0» при собранном сайте. Одно знание — одно место.
# PSEO_OUT нужен именно пульту: писать сразу в маунт, а не копировать после рендера.
OUT = pathlib.Path(os.environ.get("PSEO_OUT") or (HERE / "out"))
# ⭐ И ЕДИНСТВЕННОЕ определение каталога СОБРАННЫХ СТРАНИЦ. `site.py` в него пишет,
# рендер из него читает — переменная одна на обоих, иначе испытательный прогон собрал бы
# сайт из боевого корпуса, а тестовый остался бы лежать нетронутым.
DATA = pathlib.Path(os.environ.get("PSEO_DATA") or (HERE / "data"))
# Картинки — та же папка, что видел site.py при сборке страниц (PSEO_IMAGES), НЕ в git
# (28.08). Рендер их только копирует в OUT — искал и решал, класть ли `page.image`
# в данные, уже site.py.
IMAGES = pathlib.Path(os.environ.get("PSEO_IMAGES") or (HERE / "images"))

_env = Environment(
    loader=FileSystemLoader(str(HERE / "templates")),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=False,
    lstrip_blocks=False,
)


def load_i18n(lang: str) -> dict:
    return json.loads((HERE / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def _pick(pool: list, seed: str):
    """Детерминированный выбор из пула по сид-строке (стабильно между сборками,
    варьируется между страницами; PS декоррелирован через свой суффикс)."""
    idx = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16) % len(pool)
    return pool[idx]


# ── ДОВОД CTA ПО РАЗДЕЛУ ──
# Замер до правила: слот голоса выбирался ТОЛЬКО по хешу пути, поэтому на странице про сроки
# визы человеку обещали «и официант не перепутает заказ». Раздел даёт адресность.
# Адресность задаётся картой ИНДЕКСОВ, а не отдельным текстом на каждый язык: пулы CTA в
# 14 языках ПАРАЛЛЕЛЬНЫ по индексу. Поэтому одна карта работает на все языки, а строки живут
# там, где и весь копирайт, — в `i18n/<язык>.json`. Ключи — THEME_KEYS тракта (tract.py).
VOICE_BY_SHELF = {
    "tourism": [1, 4, 5, 6, 7],  # официант, гид, лобби, дорога, ресепшен
    "transport": [2, 6],  # таксист, спросить дорогу
    "shopping": [3, 9],  # рынок, магазин
    "health": [8],  # врач
    "housing": [0, 7],  # местные, ресепшен
    "visa": [10],  # консульство
    "border": [11],  # погранконтроль
    "docs": [12],  # нотариус и конторы
    "finance": [13],  # банк
    "customs": [14],  # досмотр
    "digital": [15],  # салон связи
    "work": [16],  # работодатель, деканат
    "safety": [17],  # полиция, скорая
}
# `VOICE_ANY` — для страниц, у которых темы нет вовсе (остаток, служебные): фраза, верная
# всегда, а не случайная из всех — случайная рождала «официанта» на визовой странице.
VOICE_ANY = [0, 6]  # «объяснись с местными», «спроси дорогу и пойми ответ»


def voice_pool(pools: dict, page: dict) -> list:
    idx = VOICE_BY_SHELF.get(page.get("theme") or "", VOICE_ANY)
    lines = [pools["voice"][i] for i in idx if i < len(pools["voice"])]
    return lines or pools["voice"]  # пул короче ожидаемого → ведём себя как раньше


def door_url(page: dict) -> str:
    """Дверь в продукт — ЧЕРЕЗ ШЛЮЗ `/<язык>/go/luky/`, чтобы переход был посчитан.

    Шлюз несёт `?geo=&shelf=`, а nginx уже пишет строку запроса и Referer — значит видно,
    какая страна и какой раздел отдают переходы, и своего бэкенда для этого не нужно.
    Сам шлюз ведёт в продукт НАПРЯМУЮ: иначе страница слала бы на саму себя.
    """
    if "/go/luky/" in (page.get("path") or ""):
        return SITE["cta_luky_url"]
    lang = page.get("lang", "ru")
    q = [("geo", page.get("geo") or ""), ("shelf", page.get("theme") or "")]
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
# живёт в коде, а не в i18n: переводчику тут нечего решать.
RTL_LANGS = {"ar", "he", "fa", "ur"}


def text_dir(lang: str) -> str:
    return "rtl" if lang in RTL_LANGS else "ltr"


# hreflang и свитчер обязаны знать, в каких языках страница СУЩЕСТВУЕТ, а не только что
# хвост адреса общий. `shared_tail` отвечает на второй вопрос, и этого мало: вид может
# выпасть в отдельном языке. Индекс строится ОДИН раз из data/ и хранится в модуле; при
# рендере одиночного файла (render.py <файл>) он пуст, и тогда падаем на `shared_tail`.
_PATHS: set[str] = set()


# Общие ассеты — один файл на сайт вместо копии в каждой странице (браузер кеширует один
# раз). `?v=` — хеш содержимого: адрес меняется при правке ассета, иначе кеш отдавал бы
# старый файл.
def asset_version() -> str:
    h = hashlib.sha1()
    d = HERE / "static"
    for f in sorted(d.glob("*")) if d.is_dir() else []:
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


def copy_assets() -> int:
    """static/ → out/assets/. Возвращает число файлов."""
    src, dst = HERE / "static", OUT / "assets"
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*")):
        (dst / f.name).write_bytes(f.read_bytes())
        n += 1
    return n


def copy_images() -> int:
    """IMAGES/*.webp → out/images/. Не в git — папки может не быть вовсе."""
    if not IMAGES.is_dir():
        return 0
    dst = OUT / "images"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(IMAGES.glob("*.webp")):
        (dst / f.name).write_bytes(f.read_bytes())
        n += 1
    return n


def index_paths(data_dir=None) -> set[str]:
    """Пути всех собранных страниц — по ним и только по ним объявляем альтернативы.

    `data_dir` нужен сторожу: подменять `HERE` нельзя — от него же берутся i18n, шаблоны
    и ассеты, и тест ломался бы на них, а не проверял правило. Боевой и испытательный
    прогоны каталог не передают — берут `DATA` (`PSEO_DATA`).
    """
    global _PATHS
    _PATHS = set()
    for jf in pathlib.Path(data_dir or DATA).glob("*.json"):
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


def _tail_of(page: dict) -> str | None:
    """Адрес БЕЗ языка (`ru/gr/visa/x/` → `gr/visa/x/`) — та же формула, что в
    `alt_langs()`, вынесена отдельно: нужна ДО рендера, чтобы решить, надо ли он
    вообще (`_dirty_tails`), а `alt_langs()` смотрит только на неё, HTML не считает.
    """
    if not page.get("shared_tail"):
        return None
    return "/".join(page["path"].split("/")[2:])


_MANIFEST_NAME = ".tails.json"  # OUT/… — не страница, robots/gitignore её не видят


def _load_tail_manifest() -> dict:
    p = OUT / _MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}  # битый манифест — как пустой: следующий прогон перерендерит всё


def _save_tail_manifest(m: dict) -> None:
    (OUT / _MANIFEST_NAME).write_text(
        json.dumps(m, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


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
    # `door_url`, чтобы переходы из текста тоже считались.
    # ⛔ `&` в атрибуте обязан быть `&amp;`: подстановка идёт мимо Jinja, шаблон сам не
    # экранирует то, что вставляется руками, — без этого ссылка выходила в двух видах.
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
    """В sitemap попадает только то, что реально можно индексировать: не глобальный
    draft И не per-page noindex."""
    return not SITE.get("draft") and not page.get("noindex")


def build_all(lastmod: str = "", data_dir=None) -> dict:
    """Рендерит data/*.json — но ТОЛЬКО те, кому это реально нужно (30.08→02.09: полный
    прогон на 46к+ страниц занимал 63с и гонялся после КАЖДОЙ страны в массовом цикле —
    89 из 90 раз впустую, ни одна из строк корпуса не менялась). Пишет sitemap.xml
    (только indexable) + robots.txt — это по-прежнему из ВСЕХ страниц, дёшево (только
    чтение JSON, не Jinja2).

    Страница НЕ рендерится (HTML не трогаем, mtime старый), если ВСЁ верно:
      1. её `data/*.json` не новее уже записанного HTML;
      2. HTML уже существует;
      3. набор языков её адреса (`_tail_of`) не изменился со времени прошлого рендера —
         если перевод только что закрыл дыру для этого адреса на новом языке, ВСЕ его
         языковые версии перерендерятся разом (у них должен обновиться hreflang-свитчер),
         не только свежедобавленная. Это и есть весь смысл: пропускать не «показалось
         дёшево», а доказанно безопасно, свитчер языков не протухнет молча.

    lastmod — ISO-дата для <lastmod> (freshness-сигнал); пустая → без тега.
    data_dir — только для сторожей (прогон берёт `DATA`, см. `PSEO_DATA`).
    Возвращает {rendered, skipped, indexed, skipped_noindex, assets, search_titles}."""
    data_dir = pathlib.Path(data_dir or DATA)
    index_paths(
        data_dir
    )  # ДО рендера: hreflang опирается на собранное, а не на догадку
    n_assets = copy_assets()  # общие CSS/JS: один файл на сайт вместо копии в странице
    copy_images()  # руками положенные картинки (28.08) — не в git, копия вслед за static/

    manifest = _load_tail_manifest()
    current_tails: dict[str, set] = {}
    pages_by_file = {}
    for jf in sorted(data_dir.glob("*.json")):
        try:
            page = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "path" not in page:
            continue  # не страница (конфиг/фикстура) — пропускаем
        pages_by_file[jf] = page
        tail = _tail_of(page)
        if tail is not None:
            current_tails.setdefault(tail, set()).add(page.get("lang"))
    dirty_tails = {
        tail
        for tail, langs in current_tails.items()
        if set(manifest.get(tail) or []) != langs
    }

    urls, n_rendered, n_skipped, n_noindex = [], 0, 0, 0
    search = {}  # язык → [[заголовок, адрес], …] для поиска по заголовкам
    for jf, page in pages_by_file.items():
        out_path = OUT / page["path"].strip("/") / "index.html"
        tail = _tail_of(page)
        stale = (
            not out_path.exists()
            or jf.stat().st_mtime > out_path.stat().st_mtime
            or (tail is not None and tail in dirty_tails)
        )
        if stale:
            build(str(jf))
            n_rendered += 1
        else:
            n_skipped += 1
        # ИНДЕКС ПОИСКА СОБИРАЕТСЯ ИЗ ТОГО, ЧТО РЕАЛЬНО ЕСТЬ, а не из корпуса: иначе он
        # обещал бы страницы, которые отсеялись, — поиск вёл бы в 404. Пропущенный рендер
        # тут не помеха — файл на диске всё равно есть, метаданные те же, что были.
        # Служебные страницы (noindex: шлюз, сама страница поиска) в индекс не идут.
        title = page.get("search_title") or page.get("intent_name") or page.get("h1")
        if title and page.get("lang") and not page.get("noindex"):
            search.setdefault(page["lang"], []).append([title, page["path"]])
        if _indexable(page):
            # Именно `updated_iso`, НЕ `updated`: второе — подпись в подвале в формате
            # MM.YYYY, sitemap такую дату не принимает.
            urls.append((SITE["domain"] + page["path"], page.get("updated_iso", "")))
        else:
            n_noindex += 1

    _save_tail_manifest({t: sorted(langs) for t, langs in current_tails.items()})

    # lastmod ПОСТРАНИЧНО: дату несёт сама страница (`updated_iso`, ставит сборщик и
    # только при РЕАЛЬНОМ изменении содержимого). Аргумент — запасной, у старых файлов
    # без поля.
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
    # ПОИСК ПО ЗАГОЛОВКАМ — один файл на язык, тянется по первому нажатию. Инлайнить в
    # каждую страницу нельзя — вес умножился бы на число страниц.
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
        "skipped": n_skipped,
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
            f"build_all: rendered={stat['rendered']} skipped={stat['skipped']} "
            f"indexed={stat['indexed']} noindex={stat['skipped_noindex']} "
            f"(draft={SITE.get('draft')})"
        )
    elif len(sys.argv) > 1:
        path = build(sys.argv[1])
        print(f"rendered -> {path}")
    else:
        print("нужен путь: python render.py data/<файл>.json  (или --all)")
