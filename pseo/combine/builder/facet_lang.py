"""facet_lang.py <geo> <lang> — переукладка facet-структуры на язык.
  lang=en → текст = оригинал ai_lesson (английский, бесплатно), метки RU→EN.
  иначе   → текст = перевод ai_lesson→lang (платно), метки RU→lang.
Только виды ≥4 фактов (что станут страницами) — так и стоимость ограничена сама собой.
Пейсинг/резерв/429/кап — внутри keybroker.call (сосок мозга), отдельный runner не нужен.

Запуск: facet_lang.py br es   → out_facet_es/br.json
"""

import json
import os
import re
import sqlite3
import sys

from keybroker import call

DB = "/home/teledigest/data/messages_fts.db"
HERE = "/root/pseo_builder"
# Ни квоты (мозг), ни окна (коэкзистенцию держит резерв мозга) — рот просто зовёт call,
# тот отдаёт None на капе → гео откладывается сам.


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
ROL = {
    "en": {
        "цель": "goal",
        "требование": "requirement",
        "обход": "workaround",
        "обстоятельство": "context",
    },
    "es": {
        "цель": "objetivo",
        "требование": "requisito",
        "обход": "alternativa",
        "обстоятельство": "contexto",
    },
    "pt": {
        "цель": "objetivo",
        "требование": "requisito",
        "обход": "alternativa",
        "обстоятельство": "contexto",
    },
}
# роли (4 фикс-значения) для прочих языков — фолбэк на английские (не блокируем перевод)


def _atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _has_cyr(s):
    return bool(re.search("[а-яёА-ЯЁ]", s or ""))


def labels_sys(lang):
    return (
        f"Translate each Russian task/topic label into a natural, concise {LANG_NAME[lang]} "
        "guide-section heading (like a table of contents). Keep it short, title-style. "
        'Return STRICT JSON: {"map": {"<ru label>": "<heading>", ...}}. Translate ALL, lose none.'
    )


def text_sys(lang):
    return (
        f"Translate each English text into natural {LANG_NAME[lang]}. Preserve ALL facts, numbers, "
        "names, conditions and caveats EXACTLY — add nothing, drop nothing. Natural target language, "
        "not a calque. Input is JSON {id: english}. Return STRICT JSON {id: translated}. Keep all ids."
    )


def kratko_sys(lang):
    # kratko генерится по-русски (dedup.py --kratko) → переводим RU→lang, В Т.Ч. для en
    return (
        f"Translate each Russian summary into natural {LANG_NAME[lang]}. Preserve ALL facts, "
        "numbers, names and caveats EXACTLY — add nothing, drop nothing. "
        "Input is JSON {id: russian}. Return STRICT JSON {id: translated}. Keep all ids."
    )


def translate_kratko(kr_by_key, lang):
    """{key: ru_kratko} → {key: translated}. Непереведённое просто выпадает (блок скрыт)."""
    out = {}
    items = list(kr_by_key.items())
    for i in range(0, len(items), 30):
        batch = dict(items[i : i + 30])
        r = call(
            json.dumps(batch, ensure_ascii=False),
            kratko_sys(lang),
            consumer="translate",
        )
        for k, v in (r or {}).items():
            if v and v.strip() and not _has_cyr(v):
                out[k] = v.strip()
    return out


def carry_groups(src_view, kept_ids, by_id_text):
    """Перенести дедуп-группы в перевод: id-состав языконезависим. Муха без перевода
    выпадает из группы; репрезентант без перевода → самый богатый переведённый в группе;
    пустая группа выпадает. n НЕ пересчитываем — счётчик подтверждений это факт ДАННЫХ,
    а не того, что удалось перевести."""
    out = []
    for g in src_view.get("groups") or []:
        ids = [i for i in g["ids"] if i in kept_ids]
        if not ids:
            continue
        rep = (
            g["rep"]
            if g["rep"] in kept_ids
            else max(ids, key=lambda i: len(by_id_text[i]))
        )
        out.append({"rep": rep, "ids": ids, "n": g["n"]})
    return out


def is_fresh(path):
    """Файл в НОВОМ формате (несёт groups)? Старый формат = пересобрать (укладка 0.10)."""
    try:
        old = json.load(open(path, encoding="utf-8"))
        vs = old.get("views_by_task", [])
        return (not vs) or any("groups" in v for v in vs)
    except Exception:
        return False


def translate_labels(labels, lang):
    """RU→lang, батчи по 60, ретраи пока остаются непереведённые (кириллица в значении)."""
    uniq = sorted(set(labels))
    mp = {}
    todo = list(uniq)
    for _ in range(4):
        if not todo:
            break
        for i in range(0, len(todo), 60):
            out = call(
                json.dumps(todo[i : i + 60], ensure_ascii=False),
                labels_sys(lang),
                consumer="labels",
            )
            for k, v in ((out or {}).get("map") or {}).items():
                if v and v.strip() and not _has_cyr(v):
                    mp[k] = v.strip()
        todo = [x for x in uniq if x not in mp]
    return {x: mp.get(x, x) for x in uniq}


def translate_texts(id_text, lang):
    """{id: english} → {id: target}. Батчи по 50 (канон 4.1: запрос ≈10К ток = 60% окна модели).
    Возвращает (out, complete): complete=False если мозг отдал None (кап рта / исчерпание ключей /
    429) — тогда гео НЕ пишем, доделаем позже (без шторма).
    """
    items = list(id_text.items())
    out = {}
    for i in range(0, len(items), 50):
        batch = dict(items[i : i + 50])
        r = call(
            json.dumps(batch, ensure_ascii=False), text_sys(lang), consumer="translate"
        )
        if (
            r is None
        ):  # None = исчерпание ключей/провал вызова (НЕ пустой батч!) → стоп, incomplete
            return (
                out,
                False,
            )  # НЕ проглатываем провал в `or {}` — иначе пишем пустышку как «done»
        for k, v in r.items():
            if v and v.strip() and not _has_cyr(v):
                out[k] = v.strip()
    return out, True


def run(geo, lang):
    out_path = f"{HERE}/out_facet_{lang}/{geo}.json"
    if os.path.exists(out_path):
        if is_fresh(out_path):
            print(f"{geo}/{lang}: уже готов (новый формат), скип", flush=True)
            return True
        print(f"{geo}/{lang}: старый формат (без groups) — пересборка", flush=True)
    src = json.load(open(f"{HERE}/out_facet/{geo}.json", encoding="utf-8"))
    views = [
        v for v in src["views_by_task"] if len(v["items"]) >= 4
    ]  # только страничные

    ids = {it["id"] for v in views for it in v["items"]}
    con = sqlite3.connect(DB)
    q = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT id, ai_lesson FROM extracted_patterns WHERE id IN ({q})", tuple(ids)
    ).fetchall()
    con.close()
    en_text = {r[0]: (r[1] or "").strip() for r in rows}

    if lang == "en":
        text = en_text
    else:
        text, complete = translate_texts(
            en_text, lang
        )  # ПЛАТНАЯ часть, квоту держит мозг
        if (
            not complete
        ):  # мозг отдал None (кап/429) → гео НЕ пишем, доделаем позже (без шторма)
            print(
                f"{geo}/{lang}: кап мозга — стоп, гео отложен (перевёл {len(text)})",
                flush=True,
            )
            return False

    label_map = translate_labels([v["zadacha"] for v in views], lang)
    rol = ROL.get(lang, ROL["en"])  # прочие языки — англ. роли (не блокируем)

    out_views = []
    kr_src = {}  # индекс out_view → ru-kratko (переведём батчем ниже)
    for v in views:
        lbl = label_map.get(v["zadacha"], v["zadacha"])
        if _has_cyr(lbl):
            continue  # метка не перевелась → не плодим кириллический URL
        items = []
        for it in v["items"]:
            t = text.get(it["id"])
            if not t:
                continue  # текста нет/не перевёлся → выкинуть муху
            items.append(
                {
                    "id": it["id"],
                    "text": t,
                    "sushnosti": [
                        {"imya": e["imya"], "rol": rol.get(e["rol"], e["rol"])}
                        for e in it.get("sushnosti") or []
                    ],
                    "mesto": it.get("mesto"),
                    "uslovie": it.get("uslovie"),
                }
            )
        if len(items) >= 4:  # после отсева мог упасть ниже порога
            tv = {"zadacha": lbl, "items": items}
            kept = {it["id"] for it in items}
            by_text = {it["id"]: it["text"] for it in items}
            # укладка 0.10: группы дедупа языконезависимы (id-состав) — несём сквозь перевод
            if v.get("groups"):
                tv["groups"] = carry_groups(v, kept, by_text)
            if v.get("kratko"):
                kr_src[str(len(out_views))] = v["kratko"]
            out_views.append(tv)

    # короткий ответ: ru-выжимка → перевод батчем (не перевёлся → блок скрыт шаблоном)
    if kr_src:
        kr_tr = translate_kratko(kr_src, lang)
        for k, val in kr_tr.items():
            out_views[int(k)]["kratko"] = val

    # КОРЕНЬ бага «пустой файл»: RU-гео ИМЕЕТ ≥4-виды, а перевод дал 0 → это ПРОВАЛ (429/сдох),
    # НЕ писать пустышку (иначе done-по-факту-файла → пропущен навсегда). На ретрай.
    if views and not out_views:
        print(
            f"{geo}/{lang}: RU={len(views)} видов, перевод дал 0 — ПРОВАЛ, НЕ пишем (ретрай)",
            flush=True,
        )
        return False
    # views пусто (гео реально тонкий) → пустой файл легитимен (нечего переводить), пишем.
    page = {"geo": geo, "views_by_task": out_views, "entity_index": {}}
    d = f"{HERE}/out_facet_{lang}"
    os.makedirs(d, exist_ok=True)
    _atomic(f"{d}/{geo}.json", page)
    print(
        f"{geo}/{lang}: {len(out_views)} видов, "
        f"{sum(len(v['items']) for v in out_views)} мух → {d}/{geo}.json",
        flush=True,
    )
    return True  # явный успех (было: падал в None → exit 3 на каждой записи)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: facet_lang.py <geo> <lang>")
    ok = run(sys.argv[1], sys.argv[2])
    sys.exit(
        0 if ok else 3
    )  # 3 = перевод провалился (мозг на капе / 429); драйвер досыпает, не штормит
