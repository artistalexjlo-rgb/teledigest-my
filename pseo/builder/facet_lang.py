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
from runner import (
    in_window,
)  # окно экстракции (часы); КВОТУ держит мозг keybroker, не pool_state

DB = "/home/teledigest/data/messages_fts.db"
HERE = "/root/pseo_builder"


def budget_ok():
    """Только окно экстракции (часы). Квоту/пул теперь держит мозг keybroker (сосок call
    отдаёт None на капе → рот сам отложит гео). Второго квота-источника (pool_state/gemini_quota) НЕТ.
    """
    return not in_window()


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
    """{id: english} → {id: target}. Батчи по 50 (канон 4.1: запрос ≈10К ток = 60% окна).
    Возвращает (out, complete): complete=False если остановились (окно экстракции ИЛИ мозг отдал
    None на капе) — тогда гео НЕ пишем, доделаем в другой день (без шторма).
    """
    items = list(id_text.items())
    out = {}
    for i in range(0, len(items), 50):
        if (
            not budget_ok()
        ):  # только окно экстракции → стоп, гео отложить (кап держит мозг ниже)
            return out, False
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
        print(f"{geo}/{lang}: уже готов, скип", flush=True)
        return True
    if (
        lang != "en" and not budget_ok()
    ):  # не начинаем гео в окне экстракции (кап держит мозг)
        print(f"{geo}/{lang}: окно экстракции — отложен", flush=True)
        return False
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
        ):  # окно экстракции ИЛИ мозг на капе → гео НЕ пишем, доделаем позже (без шторма)
            print(
                f"{geo}/{lang}: окно/кап мозга — стоп, гео отложен (перевёл {len(text)})",
                flush=True,
            )
            return False

    label_map = translate_labels([v["zadacha"] for v in views], lang)
    rol = ROL.get(lang, ROL["en"])  # прочие языки — англ. роли (не блокируем)

    out_views = []
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
            out_views.append({"zadacha": lbl, "items": items})

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
    sys.exit(0 if ok else 3)  # 3 = стоп по бюджету/окну (драйвер досыпает, не штормит)
