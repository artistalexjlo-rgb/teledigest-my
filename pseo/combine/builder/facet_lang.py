"""facet_lang.py <geo> <lang> — переукладка facet-структуры на язык.
  lang=en → текст = оригинал ai_lesson (английский, бесплатно), метки RU→EN.
  иначе   → текст = перевод ai_lesson→lang (платно), метки RU→lang.
Только виды ≥4 фактов (что станут страницами) — так и стоимость ограничена сама собой.
Пейсинг/резерв/429/кап — внутри keybroker.call (сосок мозга), отдельный runner не нужен.

Запуск: facet_lang.py br es   → out_facet_es/br.json
"""

import glob
import json
import os
import re
import sqlite3
import sys

import tail_taxonomy as _tax
from dedup import BRANCH_ITEM_MIN  # порог пунктов в ветви — ОДНО место, не копия
from keybroker import call
from slugs import slug  # тот же слаг, что строит адреса при сборке

DB = "/home/teledigest/data/messages_fts.db"
HERE = "/root/pseo_builder"
# русское имя полки → латинский ключ. Тот же источник, что у pages.py (SHELF_KEY):
# ключ должен совпадать во всех языках, иначе адреса полок разъедутся и hreflang соврёт.
SHELF_KEY = {name: key for key, name, _ in _tax.SHELVES}
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
    "de": {
        "цель": "Ziel",
        "требование": "Voraussetzung",
        "обход": "Umweg",
        "обстоятельство": "Umstand",
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


def carry_subs(src, kept_ids, label_map):
    """Перенести ВЕТВЛЕНИЕ (`subshelves`) в перевод: состав ветви — id, они языко-независимы,
    переводится только имя ветви.

    ⭐ ЗАЧЕМ (2026-08-07): перевод ветвление не нёс ВООБЩЕ, а `pages.py` строит хаб с ветками
    именно по `subshelves`. То есть языки собирались бы простынями — при том что русский уже
    разрезан. И хуже: `is_fresh` про ветви не знал, файл считался готовым навсегда, и
    исправить это было бы нечем, кроме ручного сноса файлов.

    Муха без перевода выпадает; ветвь, в которой осталось меньше `BRANCH_ITEM_MIN` пунктов,
    выпадает целиком. Меньше двух ветвей — ветвление теряет смысл, отдаём None: страница
    соберётся обычной, а не хабом с одной плиткой.

    ⛔ Порог берётся ИЗ `dedup`, а не литералом. Первая версия этой функции (929afae)
    выбрасывала ветвь только когда она ПУСТА — то есть завела вторую, более слабую копию
    правила «ветка-страница от 4 пунктов», и ветвь, потерявшая пункты при переводе,
    становилась в языке тощей страницей при соблюдённом правиле в русском.
    """
    out = []
    for sub in src.get("subshelves") or []:
        reps = [r for r in sub.get("reps") or [] if r in kept_ids]
        if len(reps) < BRANCH_ITEM_MIN:
            continue
        name = label_map.get(sub["name"], sub["name"])
        if _has_cyr(name):
            continue  # имя не перевелось → кириллический URL ветви, не плодим
        t = {"name": name, "reps": reps}
        if sub.get("key"):
            t["key"] = sub["key"]  # адрес ветви ОДИН на все языки (канон §0.11)
        out.append(t)
    return out if len(out) >= 2 else None


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
    """Файл в НОВОМ формате? Старый = пересобрать (укладка 0.10).

    Два признака: `groups` в видах и КЛЮЧ `shelves` на верхнем уровне.

    ⛔ Про полки — грабля, на которой это молча не сработало бы: до 2026-07-27 перевод
    полки не нёс вообще, а проверка смотрела только на `groups`. Значит все 13 языков
    уже лежат «готовыми» и были бы скипнуты — полок в них не появилось бы никогда, а
    прогон отрапортовал бы «готово», не сделав ничего.
    ⛔ Требуем именно КЛЮЧ, а не непустой список: у тонкого гео полок честно нет, и
    проверка «полки непусты» гоняла бы его на перевод вечно.
    """
    try:
        old = json.load(open(path, encoding="utf-8"))
        vs = old.get("views_by_task", [])
        return (
            ((not vs) or any("groups" in v for v in vs))
            and "shelves" in old
            and old.get("branches_carried") is True
        )
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


def stamp_keys(geo):
    """Проштамповать АДРЕСА в РУССКИЙ файл гео: `key` каждому страничному виду и каждой ветви.

    ⭐ ПРАВИЛО (канон §0.11, слова юзера): `/<язык>/<страна>/` + ОДИНАКОВЫЙ английский хвост.
    Ключ = слаг АНГЛИЙСКОЙ метки, поэтому `/ru/br/money/` и `/zh/br/money/` — один хвост.

    Почему в русский файл: адрес принадлежит СОДЕРЖИМОМУ, а не переводу. Пока слаг считался
    в `pages.py` от локализованной метки, у каждого языка выходил свой адрес — отсюда и
    свитчер в 404, и врущий hreflang, и вырождение всех нелатинских адресов в один «tema».
    Почему здесь: перевод меток RU→EN умеет только этот модуль.

    ⛔ Файл ДОПИСЫВАЕТСЯ, а не пересобирается: в нём лежат `groups`, `kratko` и `subshelves`,
    стоившие прогонов. ⛔ Уникальность ключей внутри гео обеспечивается ЗДЕСЬ: `slug()`
    схлопывает разные метки в один хвост, а уникализации адресов в `pages.py` нет нигде —
    совпавшие адреса молча перезаписывали бы страницы друг друга.

    Идемпотентна: уже проштампованные узлы в перевод не идут. Возвращает число новых ключей.
    """
    fn = f"{HERE}/out_facet/{geo}.json"
    d = json.load(open(fn, encoding="utf-8"))
    views = [v for v in d.get("views_by_task") or [] if len(v.get("items") or []) >= 4]
    # (узел, поле-с-меткой): у вида метка в `zadacha`, у ветви — в `name`
    nodes = [(v, "zadacha") for v in views]
    for src in list(views) + list(d.get("shelves") or []):
        nodes += [(sub, "name") for sub in (src.get("subshelves") or [])]
    todo = [(x, f) for x, f in nodes if not x.get("key")]
    if not todo:
        print(f"{geo}: адреса на месте, скип", flush=True)
        return 0
    labels = sorted({x[f] for x, f in todo})
    en = translate_labels(labels, "en")
    used = {x["key"] for x, _ in nodes if x.get("key")}  # уже занятые — не трогаем
    n = 0
    for x, f in todo:
        base = slug(en.get(x[f], ""))
        k, i = base, 1
        while k in used:  # хвост уже занят другой меткой → -2, -3, …
            i += 1
            k = f"{base}-{i}"
        used.add(k)
        x["key"] = k
        n += 1
    _atomic(fn, d)
    print(
        f"{geo}: адресов проштамповано {n} из {len(nodes)} узлов (видов {len(views)})",
        flush=True,
    )
    return n


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
    # ⭐ ПОЛКИ ТОЖЕ ПЕРЕВОДИМ (2026-07-27). Раньше набор собирался ТОЛЬКО из страничных
    # видов, поэтому текстов хвоста в переводе не было и собрать полку было не из чего:
    # 392 полки существовали лишь по-русски. Раскладка (какая муха в какой полке) от
    # языка НЕ зависит — она посчитана в ru-файле и просто переносится; платим только за
    # тексты хвоста.
    shelves = src.get("shelves") or []

    ids = {it["id"] for v in views for it in v["items"]}
    ids |= {it["id"] for sh in shelves for it in sh["items"]}
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
        if reason:  # пул не отдал (после ретраев внутри translate_texts) → гео НЕ пишем
            # маркер НЕУДАЧ → цикл перепройдёт фазу (транзиент пула), 3 раза не вышло →
            # стоп-машина. В fails писать НЕКУДА: языкового файла не создаём (иначе
            # пустышка встанет как «готово»), поэтому сигнал только маркером в вывод.
            print(
                f"⚠️ НЕУДАЧ: {geo}/{lang} {reason}; гео отложен (успел {len(text)})",
                flush=True,
            )
            return False

    # Имена полок — в ТУ ЖЕ пачку меток: их ≤9 на гео (глобальная таксономия), отдельный
    # вызов ради них был бы лишним запросом на каждое гео×язык.
    shelf_names = sorted({sh["shelf"] for sh in shelves})
    # Имена ВЕТВЕЙ — туда же: отдельный вызов на каждое гео×язык был бы лишним запросом.
    sub_names = sorted(
        {
            sub["name"]
            for src in list(views) + list(shelves)
            for sub in (src.get("subshelves") or [])
        }
    )
    label_map = translate_labels(
        [v["zadacha"] for v in views] + shelf_names + sub_names, lang
    )
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
            if v.get("key"):
                # АДРЕС страницы. Английский и ОДИН на все языки (канон §0.11): хвост
                # `/ru/br/money/` = `/zh/br/money/`. Без него слаг считался бы от
                # локализованной метки — и адреса разъезжались бы по языкам, а на
                # нелатинице вырождались в один «tema» на всё гео.
                tv["key"] = v["key"]
            kept = {it["id"] for it in items}
            by_text = {it["id"]: it["text"] for it in items}
            # укладка 0.10: группы дедупа языконезависимы (id-состав) — несём сквозь перевод
            if v.get("groups"):
                tv["groups"] = carry_groups(v, kept, by_text)
            subs = carry_subs(v, kept, label_map)
            if subs:
                tv["subshelves"] = subs
            elif v.get("branch_tried"):
                tv["branch_tried"] = True  # цельная и по-русски — незачем звать снова
            out_views.append(tv)

    # ПОЛКИ: раскладка уже посчитана в ru-файле и от языка не зависит — переносим её,
    # подставляя переведённые тексты. Ключ полки несём ОТДЕЛЬНО от имени: URL строится
    # в pages.py через SHELF_KEY, а он смотрит по РУССКОМУ имени — от переведённого он
    # промахнётся и слепит слаг из перевода, разный в каждом языке (а на нелатинских —
    # и вовсе мусорный). Ключ латинский и общий, поэтому hreflang сойдётся 1:1.
    out_shelves = []
    for sh in shelves:
        s_items = []
        for it in sh["items"]:
            t = text.get(it["id"])
            if not t:
                continue  # текста нет/не перевёлся → выкинуть муху (как в видах)
            s_items.append(
                {
                    "id": it["id"],
                    "text": t,
                    "sushnosti": [
                        {"imya": e["imya"], "rol": rol.get(e["rol"], e["rol"])}
                        for e in it.get("sushnosti") or []
                    ],
                    "mesto": it.get("mesto"),
                    "uslovie": it.get("uslovie"),
                    # тип НЕ переводим: pages.py мапит его в css-ключ по русскому имени
                    # (TYPE_KEY), а локализация чипов — отдельная, пока не сделанная тема
                    "type": it.get("type"),
                }
            )
        if not s_items:
            continue
        tsh = {
            "shelf": label_map.get(sh["shelf"], sh["shelf"]),
            "key": SHELF_KEY.get(sh["shelf"], ""),
            "items": s_items,
        }
        kept = {i["id"] for i in s_items}
        if sh.get("groups"):
            tsh["groups"] = carry_groups(
                sh, kept, {i["id"]: i["text"] for i in s_items}
            )
        subs = carry_subs(sh, kept, label_map)
        if subs:
            tsh["subshelves"] = subs
        elif sh.get("branch_tried"):
            tsh["branch_tried"] = True
        out_shelves.append(tsh)

    # КОРЕНЬ бага «пустой файл»: RU-гео ИМЕЕТ ≥4-виды, а перевод дал 0 → это ПРОВАЛ (429/сдох),
    # НЕ писать пустышку (иначе done-по-факту-файла → пропущен навсегда). На ретрай.
    if views and not out_views:
        print(
            f"{geo}/{lang}: RU={len(views)} видов, перевод дал 0 — ПРОВАЛ, НЕ пишем (ретрай)",
            flush=True,
        )
        return False
    # views пусто (гео реально тонкий) → пустой файл легитимен (нечего переводить), пишем.
    page = {
        "geo": geo,
        "views_by_task": out_views,
        "shelves": out_shelves,  # ключ пишем ВСЕГДА, даже пустым — по нему is_fresh
        # Признак формата: файл несёт ветвление. Требуется is_fresh — иначе уже лежащие
        # файлы (собранные до 07.08) считались бы готовыми навсегда и ветвей не получили.
        "branches_carried": True,
        "entity_index": {},
    }
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
    if len(sys.argv) >= 3 and sys.argv[1] == "--stamp-keys":
        # адреса штампуются ОДИН раз на гео и служат всем языкам
        gs = (
            [
                os.path.basename(p)[:-5]
                for p in sorted(glob.glob(f"{HERE}/out_facet/*.json"))
            ]
            if sys.argv[2] == "--all"
            else sys.argv[2:]
        )
        total = sum(stamp_keys(g) for g in gs)
        print(f"ИТОГО адресов проштамповано: {total}", flush=True)
        sys.exit(0)
    if len(sys.argv) < 3:
        sys.exit("usage: facet_lang.py <geo> <lang> | --stamp-keys <geo|--all>")
    ok = run(sys.argv[1], sys.argv[2])
    sys.exit(
        0 if ok else 3
    )  # 3 = перевод провалился (мозг на капе / 429); драйвер досыпает, не штормит
