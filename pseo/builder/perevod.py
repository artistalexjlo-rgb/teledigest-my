# -*- coding: utf-8 -*-
"""ЗВЕНО 6 ПЕРЕВОДЫ: английский корпус → корпус на каждом языке (PLAN.md, звено 6).

ВХОД   `out_facet/<гео>.json` — то, что собрало звено 5: страницы веток и мелочь остатка.
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

import tail_taxonomy as tax  # noqa: E402
from keybroker import call  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/pseo
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
TEMY_FILE = f"{BUILT}/temy.json"


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


def otpechatok(text):
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


def _pachkami(pary, sysprompt, consumer, batch):
    """Пары (ключ, английский текст) → {ключ: перевод}. Рту уходят НОМЕРА пачки.

    Возвращает (перевод, причина остановки). Причина не None — пул ключей не отдаёт, и
    прогон честно останавливается, не устроив шторма запросов.
    """
    out = {}
    for i in range(0, len(pary), batch):
        chunk = pary[i : i + batch]
        payload = {str(j): en for j, (_k, en) in enumerate(chunk)}
        res = None
        for _ in range(RETRY):
            res = call(
                json.dumps(payload, ensure_ascii=False), sysprompt, consumer=consumer
            )
            if res is not None:
                break
        if res is None:
            return out, "перевод прерван: пул ключей не отдаёт (исчерпание или сбой)"
        for j, (key, _en) in enumerate(chunk):  # сшивка ПО ПОЗИЦИИ
            v = str(res.get(str(j)) or "").strip()
            if v:
                out[key] = v
    return out, None


def temy(lang):
    """Имена тринадцати тем на языке. Один раз на язык, дальше берётся готовое."""
    vse = _load(TEMY_FILE, {}) or {}
    if lang == "ru":  # имена тем в таксономии УЖЕ русские — переводить нечего
        return {k: n for k, n, _d in tax.SHELVES}
    if vse.get(lang):
        return vse[lang]
    # ⛔ Источник — АНГЛИЙСКИЙ ключ темы (`visa`, `local_life`), а не русское имя из
    # таксономии: рот в этом звене переводит с английского, и второго направления у нас нет.
    pary = [(k, k.replace("_", " ")) for k, _n, _d in tax.SHELVES]
    mp, stop = _pachkami(pary, names_sys(lang), "labels", NAME_BATCH)
    if stop:
        print(f"  темы {lang}: {stop}", flush=True)
        return {}
    vse[lang] = mp
    _save(TEMY_FILE, vse)
    print(f"темы {lang}: {len(mp)} имён -> {TEMY_FILE}", flush=True)
    return mp


def korpus(geo, lang=""):
    d = "out_facet" if not lang or lang == "en" else f"out_facet_{lang}"
    return _load(f"{BUILT}/{d}/{geo}.json")


def _vse_teksty(src):
    """Все тексты гео, которые увидит читатель: советы страниц и мелочь остатка."""
    pary = []
    for v in src.get("views_by_task") or []:
        pary += [(it["id"], it.get("text") or "") for it in v.get("items") or []]
    for sh in src.get("shelves") or []:
        pary += [(it["id"], it.get("text") or "") for it in sh.get("items") or []]
    return pary


def _gotovoe(old):
    """Что уже переведено и не устарело: {id: (отпечаток, перевод)} и имена веток."""
    teksty, imena = {}, {}
    for v in (old or {}).get("views_by_task") or []:
        if v.get("src_name") and v.get("zadacha"):
            imena[v["src_name"]] = v["zadacha"]
        for it in v.get("items") or []:
            if it.get("src"):
                teksty[it["id"]] = (it["src"], it.get("text") or "")
    for sh in (old or {}).get("shelves") or []:
        for it in sh.get("items") or []:
            if it.get("src"):
                teksty[it["id"]] = (it["src"], it.get("text") or "")
    return teksty, imena


def _perelozhit(items, teksty, ist):
    """Пункты страницы на язык: текст из перевода, отпечаток — от английского оригинала."""
    return [
        {
            **it,
            "text": teksty[it["id"]],
            "src": otpechatok(ist.get(it["id"], "")),
        }
        for it in items or []
        if it["id"] in teksty
    ]


def perevedi(geo, lang):
    """Один гео на один язык. Платим ТОЛЬКО за новое и за переписанное."""
    src = korpus(geo)
    if not src:
        print(f"{geo}: английского корпуса нет", flush=True)
        return 0
    if lang == "en":
        _save(f"{BUILT}/out_facet_en/{geo}.json", src)
        print(f"{geo} en: копия оригинала, ключей 0", flush=True)
        return 0

    pary = _vse_teksty(src)
    ist = dict(pary)
    staroe, imena_gotovye = _gotovoe(korpus(geo, lang))
    # Платим за то, чего нет, и за то, чей ОРИГИНАЛ переписали. Остальное уже куплено.
    nado = [(i, en) for i, en in pary if staroe.get(i, ("", ""))[0] != otpechatok(en)]

    imena_en = []
    for v in src.get("views_by_task") or []:
        nm = v.get("zadacha") or ""
        if nm and nm not in imena_en:
            imena_en.append(nm)  # имя ОДНО на ветку: части его делят, платим один раз
    imena_nado = [(n, n) for n in imena_en if n not in imena_gotovye]

    novye, stop = _pachkami(nado, text_sys(lang), "translate", TEXT_BATCH)
    imena_novye, stop_imen = {}, None
    if not stop and imena_nado:
        imena_novye, stop_imen = _pachkami(
            imena_nado, names_sys(lang), "labels", NAME_BATCH
        )

    teksty = {i: t for i, (_h, t) in staroe.items() if i in ist}
    teksty.update(novye)
    imena = dict(imena_gotovye)
    imena.update(imena_novye)

    out = {k: v for k, v in src.items() if k not in ("views_by_task", "shelves")}
    out["lang"] = lang
    out["views_by_task"] = []
    for v in src.get("views_by_task") or []:
        items = _perelozhit(v.get("items"), teksty, ist)
        if not items:
            continue  # страница без единого переведённого пункта — не страница
        nv = dict(v)
        # ⛔ Английское имя остаётся ПОЛЕМ: по нему находится готовый перевод на следующем
        # прогоне, и по нему же видно, что имя в звене 4 переписали.
        nv["src_name"] = v.get("zadacha") or ""
        nv["zadacha"] = imena.get(nv["src_name"], nv["src_name"])
        nv["items"] = items
        out["views_by_task"].append(nv)
    out["shelves"] = []
    for sh in src.get("shelves") or []:
        items = _perelozhit(sh.get("items"), teksty, ist)
        if items:
            out["shelves"].append({**sh, "items": items})
    _save(f"{BUILT}/out_facet_{lang}/{geo}.json", out)
    temy(lang)  # имена тем — один раз на язык, отдельным файлом

    beda = stop or stop_imen
    print(
        f"{geo} {lang}: советов {len(teksty)} из {len(pary)} (куплено {len(novye)}), "
        f"имён {len(imena)} из {len(imena_en)}" + (f" ⛔ {beda}" if beda else ""),
        flush=True,
    )
    return len(novye)


def perevedi_vse(geo, langs=None):
    """Все языки одного гео. СТОП проверяется между языками — оплаченное уже на диске."""
    for lang in langs or LANGS:
        if os.path.exists("RUNNER_STOP"):
            print(f"  стоп перед языком {lang}", flush=True)
            break
        perevedi(geo, lang)


if __name__ == "__main__":
    _geo = sys.argv[1] if len(sys.argv) > 1 else ""
    _langs = [x for x in sys.argv[2:] if x in LANG_NAME] or None
    if not _geo:
        print("нужен гео: python perevod.py <гео> [языки]")
    else:
        perevedi_vse(_geo, _langs)
