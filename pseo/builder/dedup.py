"""dedup.py — укладка facet-коллектива, шаг 1: дедуп почти-дублей внутри carve-вида.

Мухи одного вида часто повторяют один факт («без резидентства сложно» ×4, «старые
купюры USD» ×6 — br). Дедуп сжимает повторы в группу: один абзац (самый богатый
текст) + честный счётчик подтверждений. Разные факты остаются порознь.

Метод (контроль глазами 2026-07-19, 4 вида br / ~120 мух): agglomerative
average-link по bge-косинусу, порог 0.86. Single-link отвергнут (цепочки: 16/17
мух в один ком на 0.80); 0.83 груб (финтех слипся с трад. банками), 0.90 слеп
(очевидные ×4 дубли не слились). Вектора — ГОТОВЫЕ, из local_vec.db свипера
(doc_id = id мухи, 1024-dim bge-m3, нормируем сами) — шаг полностью keyless.

Опционально --kratko: короткий ответ страницы (LLM-выжимка СТРОГО из текстов
топ-групп, режим переводчик — ничего нового). Рот kratko через сосок мозга.

Запуск (VPS, после facet): /root/embed_ab/venv/bin/python dedup.py <geo|--all> [--kratko]
Пишет обратно в out_facet/<geo>.json (атомарно): view.groups [+ view.kratko].
Идемпотентен: пересчёт groups дёшев и детерминирован; kratko не перегенерится.
"""

import glob
import json
import os
import sqlite3
import sys

import numpy as np

VEC_DB = os.environ.get("LOCAL_VEC_DB", "/root/embed_ab/local_vec.db")
OUT = "out_facet"
THR = (
    0.86  # порог avg-link (контроль 2026-07-19); не крутить без нового контроля глазами
)
PAGE_MIN = 4  # kratko только видам-страницам (гейт страниц в pages.py тот же)
KRATKO_TOP = (
    12  # сколько топ-абзацев кормим выжимке (окно соска ~16.6К ток — с запасом)
)

# Выжимка делается НА ЯЗЫКЕ ТЕХ ЖЕ АБЗАЦЕВ, что стоят на странице под плашкой.
# ⛔ НЕ переводить готовую русскую выжимку на другие языки (так было до 07-22): плашка
# сверху и советы ниже приходили разными путями и могли разъехаться, а на языке, которого
# никто не читает, это не поймать. Синтез из видимого текста делает согласованность
# СВОЙСТВОМ КОНСТРУКЦИИ, а не предметом проверки.
LANG_NAME = {
    "ru": "русском",
    "en": "English",
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


def kratko_sys(lang="ru"):
    if lang == "ru":
        tail = "Естественный русский, без воды и без «в чатах говорят»."
    else:
        tail = (
            f"Write the answer in natural {LANG_NAME.get(lang, lang)} — the SAME language "
            "as the advice above. No filler, no «people in chats say»."
        )
    return (
        "Ты СЖИМАЕШЬ готовые советы в короткий ответ, НЕ автор. Ниже советы одной темы. "
        "Напиши «короткий ответ» страницы: 2-3 предложения, ТОЛЬКО факты из советов ниже "
        "(самые подтверждённые/практичные), НИЧЕГО не добавлять, не выдумывать, не обобщать "
        f"сверх написанного. {tail}\n"
        'СТРОГО JSON: {"kratko": "<2-3 предложения>"}'
    )


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_vecs(ids):
    """id мухи → нормированный вектор. Мухи без вектора (свипер не догнал) — нет в dict."""
    c = sqlite3.connect(VEC_DB)
    out = {}
    for s in range(0, len(ids), 900):  # лимит переменных sqlite
        chunk = ids[s : s + 900]
        q = ",".join("?" * len(chunk))
        for did, blob in c.execute(
            f"SELECT doc_id, v FROM vec WHERE doc_id IN ({q})", chunk
        ):
            a = np.frombuffer(blob, dtype=np.float32)
            out[did] = a / (np.linalg.norm(a) or 1.0)
    c.close()
    return out


def avg_link(sim, thr):
    """Агломеративный average-link: мёржим пару кластеров с максимальной средней
    связью, пока она >= порога. Single-link отвергнут контролем (цепочки)."""
    cls = [[i] for i in range(sim.shape[0])]
    while len(cls) > 1:
        best, bi, bj = -1.0, -1, -1
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                s = float(np.mean(sim[np.ix_(cls[i], cls[j])]))
                if s > best:
                    best, bi, bj = s, i, j
        if best < thr:
            break
        cls[bi] += cls.pop(bj)
    return cls


def group_view(view, vv):
    """Вид → groups: [{rep, ids, n}], сортировка n↓ потом длина rep-текста↓.
    Репрезентант = самый богатый (длинный) текст группы. Мухи без вектора —
    каждая своей группой (не судим — не теряем)."""
    items = view["items"]
    by_id = {it["id"]: it for it in items}
    with_vec = [it for it in items if it["id"] in vv]
    no_vec = [it for it in items if it["id"] not in vv]
    groups = []
    if len(with_vec) >= 2:
        m = np.stack([vv[it["id"]] for it in with_vec])
        for c in avg_link(m @ m.T, THR):
            ids = [with_vec[i]["id"] for i in c]
            rep = max(ids, key=lambda i: len(by_id[i]["text"]))
            groups.append({"rep": rep, "ids": ids, "n": len(ids)})
    else:
        groups = [{"rep": it["id"], "ids": [it["id"]], "n": 1} for it in with_vec]
    groups += [{"rep": it["id"], "ids": [it["id"]], "n": 1} for it in no_vec]
    groups.sort(key=lambda g: (-g["n"], -len(by_id[g["rep"]]["text"])))
    return groups


def kratko_for(view, lang="ru"):
    """LLM-выжимка короткого ответа из топ-групп ЭТОГО ЖЕ файла (тексты репрезентантов) —
    значит на языке файла. None = инфра-сбой/невалид — страница выйдет без блока, не блокер.
    """
    from keybroker import call  # импорт тут: кластеризация остаётся keyless

    by_id = {it["id"]: it for it in view["items"]}
    tops = [
        by_id[g["rep"]]["text"]
        for g in view["groups"][:KRATKO_TOP]
        if g["rep"] in by_id
    ]
    if not tops:
        return None
    out = call(
        json.dumps(tops, ensure_ascii=False), kratko_sys(lang), consumer="kratko"
    )
    k = (out or {}).get("kratko")
    return k.strip() if isinstance(k, str) and k.strip() else None


def kratko_lang(geo, lang):
    """Короткий ответ для ЯЗЫКОВОГО файла: синтез из абзацев этого файла, на его языке.
    Зовётся после сборки out_facet_<lang>/<geo>.json. Идемпотентно (готовые не трогает).
    """
    fn = f"out_facet_{lang}/{geo}.json"
    d = json.load(open(fn, encoding="utf-8"))
    need = [
        v for v in d.get("views_by_task", []) if v.get("groups") and not v.get("kratko")
    ]
    if not need:
        print(f"{geo}/{lang}: kratko на месте, скип", flush=True)
        return 0
    print(f"{geo}/{lang}: нужно kratko: {len(need)}", flush=True)
    n_k = 0
    for v in need:
        k = kratko_for(v, lang)
        if k:
            v["kratko"] = k
            n_k += 1
            if n_k % 5 == 0:
                _atomic_json(fn, d)  # чекпоинт: падение не съест сделанное
        print(f"  {geo}/{lang}: kratko {n_k}/{len(need)}", flush=True)
    _atomic_json(fn, d)
    print(f"{geo}/{lang}: kratko +{n_k}", flush=True)
    return n_k


# ── ВЕТВЛЕНИЕ ПОЛОК (штатный шаг комбайна, канон юзера 2026-07-21: полка-гигант — это
# не гармошка на 400 пунктов, а ХАБ с ветками-страницами; делает конвейер, не руки).
BRANCH_MIN = 90  # полка > стольких групп → ветвим (иначе обычная полка-страница)
BRANCH_BATCH = 90  # реп-текстов в запрос (окно соска)

BRANCH_SYS = (
    "Ниже советы ОДНОЙ широкой полки гайда (id: текст). Разбей их на 4-9 ПОД-ТЕМ — "
    "каждая станет отдельной страницей. Правила: под-тема = конкретная задача/объект "
    "(«Такси и Uber», «Аэропорт GRU: трансферы»), НЕ тип совета; похожие советы — в одну "
    "под-тему; каждый совет РОВНО в одной; охвати ВСЕ; названия короткие, из содержимого.\n"
    'СТРОГО JSON: {"subs":[{"name":"<название>","ids":["0",...]}]}'
)
BRANCH_MERGE_SYS = (
    "Ниже названия под-тем одной полки, полученные кусками — среди них есть дубли одной "
    "и той же темы разными словами. Сведи к 4-9 КАНОНИЧЕСКИМ под-темам: каждому входному "
    "названию — ровно одно каноническое (можно совпадающее с входным). Не выдумывай новых тем.\n"
    'СТРОГО JSON: {"map":{"<входное>":"<каноническое>",...}}'
)


def branch_shelf(shelf):
    """Полка-гигант → subshelves: [{name, reps}]. Двухпроходно: батч-carve по текстам
    репрезентантов → слияние имён батчей в канонические (1 вызов). Сбой → None (полка
    остаётся как есть, не блокер). Непокрытые репы → без ветки (останутся на хабе)."""
    from keybroker import call

    by_id = {it["id"]: it for it in shelf["items"]}
    reps = [g["rep"] for g in shelf["groups"]]
    raw = {}  # имя-из-батча → [rep...]
    for s in range(0, len(reps), BRANCH_BATCH):
        chunk = reps[s : s + BRANCH_BATCH]
        idx = {str(j): by_id[r]["text"] for j, r in enumerate(chunk)}
        out = call(json.dumps(idx, ensure_ascii=False), BRANCH_SYS, consumer="carve")
        for sub in (out or {}).get("subs") or []:
            name = (sub.get("name") or "").strip()
            ids = [
                chunk[int(i)]
                for i in (sub.get("ids") or [])
                if isinstance(i, str) and i.isdigit() and int(i) < len(chunk)
            ]
            if name and ids:
                raw.setdefault(name, []).extend(ids)
    if len(raw) < 2:
        return None
    mp = {n: n for n in raw}
    if len(raw) > 9:  # батчи наплодили дубли имён → слить в канонические
        out = call(
            json.dumps(sorted(raw), ensure_ascii=False),
            BRANCH_MERGE_SYS,
            consumer="carve",
        )
        for k, v in ((out or {}).get("map") or {}).items():
            if k in raw and v and v.strip():
                mp[k] = v.strip()
    merged = {}
    for name, ids in raw.items():
        merged.setdefault(mp[name], []).extend(ids)
    seen = set()
    subs = []
    for name, ids in sorted(merged.items(), key=lambda x: -len(x[1])):
        uniq = [i for i in ids if i not in seen]
        seen.update(uniq)
        if len(uniq) >= 4:  # ветка-страница от 4 пунктов (гейт как у страниц)
            subs.append({"name": name, "reps": uniq})
    return subs if len(subs) >= 2 else None


def run(geo, llm=False):
    fn = f"{OUT}/{geo}.json"
    d = json.load(open(fn, encoding="utf-8"))
    views = d.get("views_by_task", [])
    shelves = d.get("shelves", [])  # хвост-антологии (полка×тип) — та же укладка
    all_ids = [it["id"] for c in (views, shelves) for v in c for it in v["items"]]
    vv = load_vecs(list(set(all_ids)))
    n_groups = n_dups = n_k = 0
    for v in views:
        v["groups"] = group_view(v, vv)
        n_groups += len(v["groups"])
        n_dups += len(v["items"]) - len(v["groups"])
        if llm and len(v["items"]) >= PAGE_MIN and not v.get("kratko"):
            k = kratko_for(v)
            if k:
                v["kratko"] = k
                n_k += 1
    n_sdups = n_b = 0
    for sv in shelves:  # полкам kratko не даём: антология разнородна, «ответа» нет
        sv["groups"] = group_view(sv, vv)
        n_sdups += len(sv["items"]) - len(sv["groups"])
        # ветвление полки-гиганта — штатный шаг комбайна (идемпотентно: не пере-жжём)
        if llm and len(sv["groups"]) > BRANCH_MIN and not sv.get("subshelves"):
            subs = branch_shelf(sv)
            if subs:
                sv["subshelves"] = subs
                n_b += 1
    _atomic_json(fn, d)
    print(
        f"{geo}: видов {len(views)}, групп {n_groups}, схлопнуто дублей {n_dups}"
        + (f", полок {len(shelves)} (дублей {n_sdups})" if shelves else "")
        + (f", kratko +{n_k}" if llm else "")
        + (f", веток у {n_b} полок" if llm and n_b else ""),
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: dedup.py <geo|--all> [--llm]  (--kratko = алиас; llm-шаги: короткие ответы + ветвление полок)"
        )
        sys.exit(1)
    if sys.argv[1] == "--kratko-lang":  # синтез kratko по готовому языковому файлу
        kratko_lang(sys.argv[2], sys.argv[3])
        sys.exit(0)
    llm = "--kratko" in sys.argv or "--llm" in sys.argv
    if sys.argv[1] == "--all":
        geos = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{OUT}/*.json"))
    else:
        geos = [sys.argv[1]]
    for g in geos:
        run(g, llm=llm)
