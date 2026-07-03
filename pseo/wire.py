"""
wire.py — единый источник правды для навигации: манифест существующих страниц →
деривация хабов, главной и всех soon-флагов. Запускать ПЕРЕД render.py --all.

Цепочка: сканируем data/*.json со статьями (faqs) → {geo: {topic}} → для каждого гео
пишем хаб (тема существует = живая ссылка, нет = «скоро»), главную (страны с хабом),
и нормализуем чипы статей (soon = целевой страницы нет). Ссылка «зажигается» сама,
когда её страница появилась — без хардкода.
"""

import json
import pathlib

BASE = pathlib.Path(__file__).parent
DATA = BASE / "data"

# «СКОРО» обещаем ТОЛЬКО если под тему есть данных ≥ порога (data-backed, реально в очереди).
# Ниже порога — плитки нет (не вешаем ложное обещание). См. BUILD_PLAN «Инвариант навигации».
THRESHOLD = 25


def load_demand():
    f = BASE / "demand.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

GEO_NAME = {"br": "Бразилия", "vn": "Вьетнам", "me": "Черногория", "id": "Индонезия",
            "gr": "Греция", "kr": "Южная Корея", "ph": "Филиппины", "de": "Германия",
            "gb": "Великобритания", "bg": "Болгария", "jp": "Япония", "by": "Беларусь",
            "fr": "Франция", "au": "Австралия", "ar": "Аргентина", "hu": "Венгрия",
            "at": "Австрия", "ru": "Россия", "cl": "Чили", "fi": "Финляндия"}
GEO_FLAG = {"br": "🇧🇷", "vn": "🇻🇳", "me": "🇲🇪", "id": "🇮🇩", "gr": "🇬🇷", "kr": "🇰🇷",
            "ph": "🇵🇭", "de": "🇩🇪", "gb": "🇬🇧", "bg": "🇧🇬", "jp": "🇯🇵", "by": "🇧🇾",
            "fr": "🇫🇷", "au": "🇦🇺", "ar": "🇦🇷", "hu": "🇭🇺", "at": "🇦🇹", "ru": "🇷🇺",
            "cl": "🇨🇱", "fi": "🇫🇮"}

# Каталог тем (slug, icon, title, blurb) — единый порядок хаба/чипов.
TOPICS = [
    ("finance", "💳", "Финансы", "Счёт, наличные, обмен, платежи"),
    ("bureaucracy", "🛂", "Документы и виза", "Визы, ВНЖ, легализация"),
    ("housing", "🏠", "Жильё и аренда", "Где искать, договор, депозит"),
    ("safety", "🛡", "Безопасность", "Районы, на улице, что не делать"),
    ("transport", "🚕", "Транспорт", "Метро, такси, межгород"),
    ("health", "🩺", "Медицина", "Врачи, страховка, аптеки"),
    ("shopping", "🛒", "Покупки и цены", "Что где, цены, рынки"),
]
TOPIC_META = {slug: (ic, t, b) for slug, ic, t, b in TOPICS}


def manifest():
    """{geo: set(topic)} по реально существующим страницам-статьям (с faqs)."""
    have = {}
    for jf in DATA.glob("*.json"):
        p = json.loads(jf.read_text(encoding="utf-8"))
        if "faqs" not in p or "path" not in p:
            continue
        parts = p["path"].strip("/").split("/")  # ru/<geo>/<topic>
        if len(parts) == 3 and parts[0] == "ru":
            have.setdefault(parts[1], set()).add(parts[2])
    return have


def write_hub(geo, topics_have, demand):
    tiles = []
    dgeo = demand.get(geo, {})
    for slug, ic, title, blurb in TOPICS:
        if slug in topics_have:
            soon = False                       # страница есть → живая ссылка
        elif dgeo.get(slug, 0) >= THRESHOLD:
            soon = True                        # данные есть, страницы нет → честное «СКОРО»
        else:
            continue                           # данных нет → не обещаем (анти-протухание)
        tiles.append({
            "icon": ic, "title": title, "blurb": blurb,
            "url": f"/ru/{geo}/{slug}/", "soon": soon,
        })
    hub = {
        "lang": "ru", "geo": geo, "geo_name": GEO_NAME.get(geo, geo),
        "template": "index.html.j2", "path": f"/ru/{geo}/", "updated": "06.2026",
        "title": f"{GEO_NAME.get(geo, geo)}: деньги, документы, жильё — живой опыт · Luky",
        "meta_desc": f"Гайды по {GEO_NAME.get(geo, geo)} из живого опыта чатов сообществ: "
                     "деньги, документы, жильё, безопасность, транспорт. Без воды.",
        "h1": GEO_NAME.get(geo, geo),
        "intro": "Живой опыт тех, кто реально через это прошёл — по делу, без воды. "
                 "Выбери тему, а под свой конкретный случай <strong>спроси Luky</strong>.",
        "list_label": "Темы", "tiles": tiles,
    }
    (DATA / f"ru_{geo}_hub.json").write_text(
        json.dumps(hub, ensure_ascii=False, indent=1), encoding="utf-8")


def write_home(geos, have):
    tiles = [{
        "icon": GEO_FLAG.get(g, "•"), "title": GEO_NAME.get(g, g),
        "blurb": f"{len(have[g])} тем · живой опыт", "url": f"/ru/{g}/",
    } for g in geos]
    home = {
        "lang": "ru", "template": "index.html.j2", "path": "/ru/", "updated": "06.2026",
        "crumb_label": None,
        "title": "Luky — живой опыт по странам: деньги, документы, жильё",
        "meta_desc": "Инфопортал Luky: реальный опыт из чатов сообществ по странам — "
                     "деньги, документы, жильё, безопасность. Без воды, под твой случай.",
        "h1": "Куда едешь?",
        "intro": "Реальный опыт тех, кто уже прошёл через местные непонятки — по делу, без воды. "
                 "Выбери страну, а под свой случай <strong>спроси Luky</strong>.",
        "list_label": "Страны", "tiles": tiles,
    }
    (DATA / "ru_home.json").write_text(
        json.dumps(home, ensure_ascii=False, indent=1), encoding="utf-8")


def normalize_chips(have, demand):
    """Чипы статей: живая если страница есть, «СКОРО» если данные ≥ порога,
    иначе чип УБИРАЕТСЯ (не обещаем без данных — анти-протухание)."""
    for jf in DATA.glob("*.json"):
        p = json.loads(jf.read_text(encoding="utf-8"))
        if "faqs" not in p or "chips" not in p:
            continue
        new = []
        for c in p["chips"]:
            parts = c.get("url", "").strip("/").split("/")
            if len(parts) != 3:
                continue
            geo, topic = parts[1], parts[2]
            exists = topic in have.get(geo, set())
            backed = demand.get(geo, {}).get(topic, 0) >= THRESHOLD
            if exists:
                c["soon"] = False
                new.append(c)
            elif backed:
                c["soon"] = True
                new.append(c)
            # ни страницы, ни данных → чип выкинут
        p["chips"] = new
        jf.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    have = manifest()
    demand = load_demand()
    geos = [g for g in GEO_NAME if g in have]  # стабильный порядок
    for g in geos:
        write_hub(g, have[g], demand)
    write_home(geos, have)
    normalize_chips(have, demand)
    print(f"wired: {len(geos)} гео-хабов + главная (порог данных СКОРО = {THRESHOLD}).")
    for g in geos:
        dgeo = demand.get(g, {})
        soon = [s for s, _, _, _ in TOPICS if s not in have[g] and dgeo.get(s, 0) >= THRESHOLD]
        print(f"  {g}: live={sorted(have[g])} soon(data-backed)={soon}")
