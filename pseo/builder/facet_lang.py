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
        'not a calque. Input is JSON {"0": english, "1": english, ...}. Return STRICT JSON with the '
        'SAME short numeric keys: {"0": translated, ...}. Keep every key, translate every value.'
    )


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
    Возвращает (out, reason): reason=None — всё ок; иначе строка-причина остановки.

    call() отдаёт None и на РАЗОВОМ сбое парса (модель вернула битый JSON — флаки), и на
    реальном исчерпании ключей. Раньше ЛЮБОЙ None рубил весь гео с надписью «кап» — одна
    кривая пачка = 0 переводов и ложь про кап (юзер 07-22). Теперь пачку РЕТРАИМ: флаки-
    ответ на повторе обычно проходит; если ключи реально стынут — call вернёт None быстро
    (без похода в Google), и мы честно остановимся, потратив ~0 лишних запросов.
    """
    items = list(id_text.items())  # [(настоящий_хэш_id, english), ...]
    out = {}
    for i in range(0, len(items), 50):
        chunk = items[i : i + 50]  # держим ПОРЯДОК — по нему сошьём назад
        # ⭐ модели даём ПОРЯДКОВЫЕ "0".."49", НЕ 24-символьный хэш (образец carve в
        # facet.py). Копировать длинный хэш 50 раз — не её задача, на ней и врала id
        # (факт 07-22). Короткий индекс скопировать легко; хэш живёт снаружи.
        payload = {str(j): txt for j, (_id, txt) in enumerate(chunk)}
        r = None
        for _ in range(3):  # 1 + 2 ретрая пачки на транзиентный сбой (парс/сеть)
            r = call(
                json.dumps(payload, ensure_ascii=False),
                text_sys(lang),
                consumer="translate",
            )
            if r is not None:
                break
        if r is None:  # три раза подряд None → пул реально не отдаёт, стоп (без шторма)
            return out, "перевод прерван: пул ключей не отдаёт (исчерпание/сбой)"
        for j, (real_id, _txt) in enumerate(chunk):  # сшивка ПО ПОЗИЦИИ
            v = r.get(str(j))
            if v and str(v).strip() and not _has_cyr(str(v)):
                out[real_id] = str(v).strip()
    return out, None


def add_kratko(geo, lang):
    """Синтез коротких ответов по ГОТОВОМУ языковому файлу — логика и промпт живут в
    dedup.kratko_lang (одно место на все языки, включая ru). Сбой не роняет перевод."""
    try:
        import dedup

        cwd = os.getcwd()
        os.chdir(HERE)  # dedup работает относительными путями out_facet_<lang>/
        try:
            return dedup.kratko_lang(geo, lang)
        finally:
            os.chdir(cwd)
    except Exception as e:
        print(f"{geo}/{lang}: kratko не сделан ({type(e).__name__}: {e})", flush=True)
        return 0


def run(geo, lang):
    out_path = f"{HERE}/out_facet_{lang}/{geo}.json"
    if os.path.exists(out_path):
        if is_fresh(out_path):
            # ⛔ НЕ досинтезировать тут kratko: это РАЗОВЫЙ ретрофит по старому материалу,
            # ему не место в постоянном пути (иначе каждый заход сканирует всё старое —
            # «разовое исправление навсегда», юзер 07-22). Ретрофит = отдельная команда
            # `dedup.py --kratko-lang <geo> <lang>`, прогоняется один раз.
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
        text, reason = translate_texts(
            en_text, lang
        )  # ПЛАТНАЯ часть, квоту держит мозг
        if reason:  # пул не отдал (после ретраев) → гео НЕ пишем, доделаем позже
            print(
                f"{geo}/{lang}: {reason}; гео отложен (успел {len(text)})",
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
            tv = {"zadacha": lbl, "items": items}
            kept = {it["id"] for it in items}
            by_text = {it["id"]: it["text"] for it in items}
            # укладка 0.10: группы дедупа языконезависимы (id-состав) — несём сквозь перевод
            if v.get("groups"):
                tv["groups"] = carry_groups(v, kept, by_text)
            out_views.append(tv)

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
    # КОРОТКИЙ ОТВЕТ синтезируется ЗДЕСЬ, из только что записанных абзацев этого языка —
    # не переводится с русского (до 07-22 было так, плашка могла разъехаться с текстом).
    add_kratko(geo, lang)
    return True  # явный успех (было: падал в None → exit 3 на каждой записи)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: facet_lang.py <geo> <lang>")
    ok = run(sys.argv[1], sys.argv[2])
    sys.exit(
        0 if ok else 3
    )  # 3 = перевод провалился (мозг на капе / 429); драйвер досыпает, не штормит
