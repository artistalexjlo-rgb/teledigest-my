"""Сторож географии: имя и регион есть у КАЖДОЙ страны, а не у тех, что нам сегодня нужны.

⛔ Зачем именно так. 11.07 я завёл в `pages.py` свои 35 имён и 35 флагов, хотя в репо с 24.04
лежит справочник на 249 стран, а `CLAUDE.md` прямо запрещает хардкод стран. Итог прожил месяц:
на живом сайте 55 позиций в группе «Другие» сырыми кодами (`ae`, `al`, `bo`), Канада и США — в
«Других», а часть чипов вела в 404. Юзер на предложение «дополнить до наших гео» ответил:
«звучит как опять урезанная версия на сейчас» — поэтому проверяем ПОЛНОТУ по справочнику,
а не по корпусу. Ослабленная запись правила и есть механизм регресса (тот же урок, что §0.11).
"""

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PSEO = HERE.parent
sys.path[:0] = [str(HERE)]

import country_codes as ref  # noqa: E402  копия рядом; дословность сторожит test_country_codes_copy
import pages  # noqa: E402


def test_every_country_has_region():
    """Добавили страну в справочник — тест красный, пока ей не назначен регион. Без этого
    новая страна молча уедет в «Другие», как уже было с 214 странами."""
    missing = sorted(g for g in ref.COUNTRIES if g not in pages.CODE2REGION)
    assert not missing, f"страны без региона: {missing}"


def test_no_phantom_codes_in_regions():
    """И наоборот: в раскладке нет кодов, которых в справочнике не существует (я уже вписал
    туда `eh`, которого там нет)."""
    extra = sorted(set(pages.CODE2REGION) - set(ref.COUNTRIES))
    assert not extra, f"кодов нет в справочнике: {extra}"


def test_region_keys_are_words_not_codes():
    """Ключ региона не должен совпадать с кодом страны: `me` — Черногория, `na` — Намибия,
    `af` — Афганистан, `la` — Лаос. Совпадение ключа с кодом путает читателя кода."""
    clash = sorted(k for k in pages.REGION_CODES if k in ref.COUNTRIES)
    assert not clash, f"ключ региона совпал с кодом страны: {clash}"


def test_every_region_named_in_every_buildable_language():
    """Регион без имени печатался бы ключом. Раньше имена были на 4 языках из 14, то есть на
    десяти языках заголовки выпадали на английский."""
    holes = [
        (lang, key)
        for lang in pages.COPY
        for key in pages.REGION_CODES
        if key not in pages.REGION_NAMES.get(lang, {})
    ]
    assert not holes, f"нет имени региона: {holes[:10]}"


def test_no_publishing_geo_falls_back_to_code():
    """Ни одно гео, У КОТОРОГО ЕСТЬ СТРАНИЦЫ, не остаётся без имени ни на одном языке.

    Проверяем именно публикующие, а не все файлы корпуса: `eu`, `uk`, `ua` имеют ноль видов
    и в справочнике их нет (`uk` вообще осколок — Великобритания живёт как `gb` с 53 видами).
    Они уходят на шаге «отключить пустые», и придумывать им имена было бы лечением симптома.
    """
    import json

    built = os.environ.get("BUILT_DIR", str(PSEO / "builder"))
    geos = []
    for p in pathlib.Path(built, "out_facet").glob("*.json"):
        if "," in p.stem:  # мусорные ключи вида «au, nz» — отдельная задача
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(len(v.get("items") or []) >= 4 for v in d.get("views_by_task") or []):
            geos.append(p.stem)
    if not geos:  # на чистой машине корпуса нет — проверять нечего
        return
    # `any` — не страна, а «везде»: человеческое имя ему даёт следующий шаг (ключ в i18n).
    bare = [
        (g, lang)
        for g in geos
        for lang in pages.COPY
        if g != "any" and pages.geo_name(g, lang) == g
    ]
    assert not bare, f"имя не нашлось, печатался бы код: {bare[:10]}"


def test_home_shows_names_not_keys():
    """Главная не печатает ни код страны, ни ключ региона.

    Оба дефекта были найдены прогоном, а не чтением: строка `"oth"` в `home_data` пережила
    переименование ключей и выводилась как «oth», а `any` (не страна) попадала в регионы.
    """
    geos = ["br", "gr", "us", "ca", "eg", "au", "ru", "any"]
    counts = {g: 5 for g in geos}
    for lang in ("ru", "en", "de"):
        _popular, regions, index = pages.home_data(lang, geos, counts)
        keys = set(pages.REGION_CODES)
        assert not [r for r in regions if r["name"] in keys], f"{lang}: регион ключом"
        assert not [
            t for t in index if t["name"] in ref.COUNTRIES and len(t["name"]) == 2
        ], f"{lang}: страна кодом"
        inside = [g["url"] for r in regions for g in r["geos"]]
        assert f"/{lang}/any/" not in inside, "`any` не страна, в регионе ей не место"


def test_flags_come_from_reference():
    """Своей таблицы флагов нет: она совпадала со справочником один в один на всех 35 странах,
    то есть была чистым дублем."""
    assert not hasattr(pages, "GEO_FLAG"), "GEO_FLAG вернулся — это дубль справочника"
    assert pages.geo_flag("br") == ref.COUNTRIES["br"][1]
    assert (
        pages.geo_flag("zz") == "•"
    ), "неизвестный код должен давать заглушку, не падать"


def test_geo_names_is_only_an_override():
    """`GEO_NAMES` имеет право жить, но только как переопределение: в нём локализованные имена
    на 14 языков, которых в справочнике нет. Своей ПОЛНОЙ таблицы стран быть не должно —
    иначе она снова начнёт отставать (было 35 из 249)."""
    ru = pages.GEO_NAMES.get("ru", {})
    assert (
        len(ru) < len(ref.COUNTRIES) / 2
    ), "GEO_NAMES разросся до второй таблицы стран"
    assert (
        pages.geo_name("kr", "ru") == "Южная Корея"
    ), "осознанное переопределение потеряно"
    assert (
        pages.geo_name("al", "ru") == ref.COUNTRIES["al"][0]
    ), "имя должно идти из справочника"
