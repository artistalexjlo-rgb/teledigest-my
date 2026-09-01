# -*- coding: utf-8 -*-
"""ЗВЕНО 6 ПЕРЕВОДЫ: английский корпус → корпус на каждом языке (PLAN.md, звено 6).

ВХОД   `out_facet_en/<гео>.json` — то, что собрало звено 5: страницы веток и мелочь остатка.
ЧТО    рот `translate` переводит тексты советов, рот `labels` — имена веток.
ВЫХОД  `out_facet_<язык>/<гео>.json` той же формы: адреса, ветки и части НЕ трогаются.

⛔ ПОЧЕМУ НОВЫЙ МОДУЛЬ, А НЕ ПРАВКА `facet_lang.py`. Тот переводит С РУССКОГО (структура
была русской до 25.08), ждёт корпус старой схемы (`groups`, `subshelves`, `kratko`) и несёт
`stamp_keys` — штамповку адресов ротом. Адрес теперь рождается в звене 4, русского в
структуре нет, полей этих в новом корпусе нет: править там нечего, все три причины разом.

⛔ РУССКИЙ — ТАКОЙ ЖЕ ЯЗЫК ПЕРЕВОДА (решение юзера 25.08). Отдельного пути к нему нет.
Английский не переводится вовсе: корпус на нём и написан, копия бесплатна.

⛔ ПЕРЕВОДИМ ВСЁ, ЧТО ВИДНО ЧИТАТЕЛЮ — включая мелочь остатка на странице темы. Она
показывается абзацами, и непереведённой осталась бы английской вставкой посреди страницы.

⛔ Рту идут НОМЕРА, не id (сквозное правило PLAN.md): 24-символьный хеш модели копировать
незачем, и на нём она уже врала.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tract  # noqa: E402
from keybroker import any_alive, call  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))  # …/pseo/tract — сам файл тут же
BUILT = os.environ.get("BUILT_DIR", f"{BASE}/builder")

# Пачки: тексты длинные, имена короткие. Числа из старого тракта, они отработали.
TEXT_BATCH = 50
NAME_BATCH = 60
RETRY = 3  # ретрай пачки: битый JSON — флаки, а не исчерпание ключей

LANG_NAME = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "pt": "Portuguese",
    "zh": "Chinese (Simplified)",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "th": "Thai",
    "it": "Italian",
    "hi": "Hindi",
    "tr": "Turkish",
}
LANGS = [
    x for x in LANG_NAME if x != "en"
]  # английский не переводим — он и есть источник

# Имена тринадцати тем на каждом языке. Они одни на весь сайт, поэтому переводятся ОДИН
# раз на язык и лежат отдельно от стран: платить за них в каждом гео незачем.
# `themes.json` РАСТЁТ — звено 6 дописывает купленные языки, как canon.json, поэтому живёт
# на СМОНТИРОВАННОМ томе (BUILT_DIR), а не в образе: иначе редеплой стирал бы покупки
# (28.08, юзер поймал — я скопировал место у СТАТИЧНОГО countries.json, не подумав, что
# этот файл не статичный). Английский источник — сид в git, копируется на том при первом
# касании; дальше том живёт своей жизнью, сид не читается.
SEED_THEMES_FILE = f"{os.path.dirname(os.path.abspath(__file__))}/themes.json"
THEMES_FILE = f"{BUILT}/themes.json"


def _load(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)  # атомарно: оборванный прогон не оставит полфайла


def fingerprint(text):
    """Короткий отпечаток АНГЛИЙСКОГО оригинала.

    По нему видно, что источник переписали и перевод устарел. Без отпечатка устаревший
    перевод жил бы под тем же id вечно и незаметно.
    """
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]


def text_sys(lang):
    return (
        f"Translate each English text into natural {LANG_NAME[lang]}. Preserve ALL facts, "
        "numbers, names, conditions and caveats EXACTLY — add nothing, drop nothing. "
        'Natural target language, not a calque. Input is JSON {"0": english, ...}. '
        'Return STRICT JSON with the SAME numeric keys: {"0": translated, ...}. '
        "Keep every key, translate every value."
    )


def names_sys(lang):
    return (
        f"Translate each English page title into a natural, concise {LANG_NAME[lang]} "
        "guide-section heading (like a table of contents). Keep it short, title-style, "
        'no trailing period. Input is JSON {"0": english, ...}. Return STRICT JSON with '
        'the SAME numeric keys: {"0": translated, ...}. Translate ALL, lose none.'
    )


def _by_batches(pairs, sysprompt, consumer, batch):
    """Пары (ключ, английский текст) → {ключ: перевод}. Рту уходят НОМЕРА пачки.

    Возвращает (перевод, причина остановки). Причина не None — ТОЛЬКО когда мозг
    (`keybroker.any_alive()`) подтвердил, что пул реально мёртв, и прогон честно
    останавливается, не устроив шторма запросов.

    ⛔ 31.08, живой лог: у md пачка №1 (из ~17) провалилась всплеском отказов Google
    (429/5xx/сеть на 4 разных ключах подряд — `call()` внутри уже это корректно
    различает и не путает с исчерпанием), а `_by_batches` хоронил ВЕСЬ ОСТАЛЬНОЙ
    язык — «0 из 415», хотя пул был жив (соседние по времени языки прошли гладко).
    Оценка «жив ли пул» — знание мозга, не присоски: спрашиваем `any_alive()`
    вместо того чтобы гадать по одной неудачной пачке. Пул жив — пропускаем ТОЛЬКО
    эту пачку и идём дальше; мёртв — останавливаемся, как раньше.
    """
    out = {}
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        payload = {str(j): en for j, (_k, en) in enumerate(chunk)}
        res = None
        for _ in range(RETRY):
            res = call(
                json.dumps(payload, ensure_ascii=False), sysprompt, consumer=consumer
            )
            if res is not None:
                break
        if res is None:
            if not any_alive():
                return out, "перевод прерван: пул ключей не отдаёт (мозг подтвердил)"
            print(
                f"  пачка {i // batch + 1} не задалась — пул жив, пропускаем и идём дальше",
                flush=True,
            )
            continue
        for j, (key, _en) in enumerate(chunk):  # сшивка ПО ПОЗИЦИИ
            v = str(res.get(str(j)) or "").strip()
            if v:
                out[key] = v
    return out, None


def theme_names(lang):
    """Имена тринадцати тем на языке. ОДИН путь для любого языка (27.08, без ru-ветки):
    уже есть в файле — берём оттуда, нет — покупаем и дописываем файл.
    """
    vse = _load(THEMES_FILE, {}) or {}
    if not vse.get("en"):
        # Первое касание смонтированного тома: английского источника там ещё нет —
        # копируем сид из git и тут же сохраняем, дальше том самодостаточен.
        vse["en"] = (_load(SEED_THEMES_FILE, {}) or {}).get("en", {})
        _save(THEMES_FILE, vse)
    if vse.get(lang):
        return vse[lang]
    # Источник — АНГЛИЙСКОЕ имя из того же файла: английский тут такой же полноценный
    # язык, как советы в корпусе, а не голый ключ (`visa`, `local_life`).
    en = vse.get("en") or {}
    pairs = [(k, en.get(k) or k.replace("_", " ")) for k in tract.THEME_KEYS]
    mp, stop = _by_batches(pairs, names_sys(lang), "labels", NAME_BATCH)
    if stop:
        print(f"  темы {lang}: {stop}", flush=True)
        return {}
    vse[lang] = mp
    _save(THEMES_FILE, vse)
    print(f"темы {lang}: {len(mp)} имён -> {THEMES_FILE}", flush=True)
    return mp


def corpus(geo, lang="en"):
    return _load(f"{BUILT}/out_facet_{lang}/{geo}.json")


def _all_texts(src):
    """Все тексты гео, которые увидит читатель: советы страниц и мелочь остатка."""
    pairs = []
    for v in src.get("views_by_task") or []:
        pairs += [(it["id"], it.get("text") or "") for it in v.get("items") or []]
    for sh in src.get("shelves") or []:
        pairs += [(it["id"], it.get("text") or "") for it in sh.get("items") or []]
    return pairs


def _ready(old):
    """Что уже переведено и не устарело: {id: (отпечаток, перевод)} и имена веток."""
    teksty, names = {}, {}
    for v in (old or {}).get("views_by_task") or []:
        if v.get("source_title") and v.get("title"):
            names[v["source_title"]] = v["title"]
        for it in v.get("items") or []:
            if it.get("source"):
                teksty[it["id"]] = (it["source"], it.get("text") or "")
    for sh in (old or {}).get("shelves") or []:
        for it in sh.get("items") or []:
            if it.get("source"):
                teksty[it["id"]] = (it["source"], it.get("text") or "")
    return teksty, names


def _retext(items, teksty, source_by_id):
    """Пункты страницы на язык: текст из перевода, отпечаток — от английского оригинала."""
    return [
        {
            **it,
            "text": teksty[it["id"]],
            "source": fingerprint(source_by_id.get(it["id"], "")),
        }
        for it in items or []
        if it["id"] in teksty
    ]


def translate_geo(geo, lang):
    """Один гео на один язык (не `en` — его сразу пишет звено 5, переводить нечего).

    Платим ТОЛЬКО за новое и за переписанное.
    """
    src = corpus(geo)
    if not src:
        print(f"{geo}: английского корпуса нет", flush=True)
        return 0

    pairs = _all_texts(src)
    source_by_id = dict(pairs)
    ready, imena_gotovye = _ready(corpus(geo, lang))
    # Платим за то, чего нет, и за то, чей ОРИГИНАЛ переписали. Остальное уже куплено.
    todo = [(i, en) for i, en in pairs if ready.get(i, ("", ""))[0] != fingerprint(en)]

    imena_en = []
    for v in src.get("views_by_task") or []:
        nm = v.get("title") or ""
        if nm and nm not in imena_en:
            imena_en.append(nm)  # имя ОДНО на ветку: части его делят, платим один раз
    imena_nado = [(n, n) for n in imena_en if n not in imena_gotovye]

    bought, stop = _by_batches(todo, text_sys(lang), "translate", TEXT_BATCH)
    imena_novye, stop_imen = {}, None
    if not stop and imena_nado:
        imena_novye, stop_imen = _by_batches(
            imena_nado, names_sys(lang), "labels", NAME_BATCH
        )

    teksty = {i: t for i, (_h, t) in ready.items() if i in source_by_id}
    teksty.update(bought)
    names = dict(imena_gotovye)
    names.update(imena_novye)

    out = {k: v for k, v in src.items() if k not in ("views_by_task", "shelves")}
    out["lang"] = lang
    out["views_by_task"] = []
    for v in src.get("views_by_task") or []:
        items = _retext(v.get("items"), teksty, source_by_id)
        if not items:
            continue  # страница без единого переведённого пункта — не страница
        nv = dict(v)
        # ⛔ Английское имя остаётся ПОЛЕМ: по нему находится готовый перевод на следующем
        # прогоне, и по нему же видно, что имя в звене 4 переписали.
        nv["source_title"] = v.get("title") or ""
        nv["title"] = names.get(nv["source_title"], nv["source_title"])
        nv["items"] = items
        out["views_by_task"].append(nv)
    out["shelves"] = []
    for sh in src.get("shelves") or []:
        items = _retext(sh.get("items"), teksty, source_by_id)
        if items:
            out["shelves"].append({**sh, "items": items})
    _save(f"{BUILT}/out_facet_{lang}/{geo}.json", out)
    theme_names(lang)  # имена тем — один раз на язык, отдельным файлом

    trouble = stop or stop_imen
    print(
        f"{geo} {lang}: советов {len(teksty)} из {len(pairs)} (куплено {len(bought)}), "
        f"имён {len(names)} из {len(imena_en)}" + (f" ⛔ {trouble}" if trouble else ""),
        flush=True,
    )
    return len(bought)


def translate_all(geo, langs=None):
    """Все языки одного гео. СТОП проверяется между языками — оплаченное уже на диске."""
    for lang in langs or LANGS:
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп перед языком {lang}", flush=True)
            break
        translate_geo(geo, lang)


if __name__ == "__main__":
    _geo = sys.argv[1] if len(sys.argv) > 1 else ""
    _langs = [x for x in sys.argv[2:] if x in LANG_NAME] or None
    if not _geo:
        print("нужен гео: python translation.py <гео> [языки]")
    else:
        translate_all(_geo, _langs)
