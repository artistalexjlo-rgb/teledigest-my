# -*- coding: utf-8 -*-
"""ВЕКТОРА И БЛИЗОСТЬ: готовые bge-вектора свипера + кластеризация. Ключей не тратит.

⭐ ЗАЧЕМ ОТДЕЛЬНЫМ МОДУЛЕМ (2026-08-24, заказ юзера). Новому тракту от `dedup.py` нужны
только эти три функции — чтение векторов и группировка. Остальные 400 строк того файла —
рты отменённой схемы (`kratko`, ветвление, `run`), и пока всё лежало вместе, каждое обращение
к векторам открывало файл со старыми приёмами перед глазами.

Вектора считает свипер (bge-m3, локальная модель) и кладёт в `local_vec.db` — здесь только
чтение. `dedup.py` берёт эти же имена отсюда: второй копии нет.
"""

import os
import sqlite3

import numpy as np

VEC_DB = os.environ.get("LOCAL_VEC_DB", "/root/embed_ab/local_vec.db")


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


def groups_all(ids, vv, thr):
    """Схлопывание почти-копий на БОЛЬШОМ множестве (гео целиком, звено 2).

    Возвращает список списков id: [[рядом стоящие…], …]. Мухи без вектора не судим —
    каждая идёт своей группой.

    ⭐ ПОЧЕМУ НЕ ЗОВЁМ `avg_link` НА ВСЁМ СРАЗУ. Он перебирает все пары кластеров на каждом
    слиянии: на виде из 30 мух это незаметно, на гео из 765 — порядка 10^7 переборов, часы.
    Здесь сначала бьём множество на КОМПОНЕНТЫ по рёбрам «похожи не меньше порога», и
    average-link считаем ВНУТРИ компоненты. Это не приближение, а тот же результат: средняя
    связь не выше максимальной, поэтому две группы без единого ребра ≥ порога слиться не
    могут. Компоненты на рабочем пороге мелкие, и перебор внутри них дёшев.
    """
    have = [i for i in ids if i in vv]
    if len(have) < 2:
        return [[i] for i in ids]
    m = np.stack([vv[i] for i in have])
    sim = m @ m.T
    parent = list(range(len(have)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(*np.where(np.triu(sim >= thr, k=1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for i in range(len(have)):
        comps.setdefault(find(i), []).append(i)

    out = []
    for comp in comps.values():
        if len(comp) == 1:
            out.append([have[comp[0]]])
            continue
        sub = sim[np.ix_(comp, comp)]
        for c in avg_link(sub, thr):
            out.append([have[comp[i]] for i in c])
    out += [[i] for i in ids if i not in vv]
    return out
