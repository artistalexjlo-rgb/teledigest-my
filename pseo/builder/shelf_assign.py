"""shelf_assign.py — полка (тема) у каждого ВИДА. Без ключей, по готовой разметке хвоста.

Зачем. Ось адресации у нас — формулировка метки задачи, и уровня темы в фактовом контуре нет
вообще: у Греции 62 вида при 8 настоящих темах («сроки» ×4, «финансы» ×7, «жильё и маршрут»
×9), а хаб страны вываливает 63 ссылки плоским списком. Плитки на хабе, довод CTA по теме и
блок «рядом по теме» — всё стоит на одном отсутствующем поле. Канон: §0.12.

⛔ Своей таблицы слов не заводим. Рот `assign` уже разложил ХВОСТ по девяти полкам
таксономии — это готовая разметка, за неё заплачено. Берём её как обучающие примеры: центр
полки = средний вектор её хвостовых мух, вид кладём в ближайшую полку. Вектора — ГОТОВЫЕ, из
`local_vec.db` свипера (bge-m3), то есть шаг полностью keyless и повторяемый.

Метод проверен замером ДО реализации (2026-08-12), контроль глазами на Греции: 49 визовых
ломтиков сошлись в «Визовые процедуры», паромы и аренда авто — в «Транспорт», Крит с
ресторанами — в «Работа, учёба, быт», банковские карты — в «Финансы». По корпусу разложилось
1882 вида из 1889 (семь без векторов — свипер не догнал).
⚠️ Известный недостаток набора из девяти полок: отдых и еда падают в «Работа, учёба, быт».
Лечится добавлением полки в таксономию и ПЕРЕсчётом — раскладка дешёвая и повторяемая.

Запуск (VPS, где лежат вектора):
    /root/embed_ab/venv/bin/python shelf_assign.py --all [--dry]
Пишет `view["shelf"]` в `out_facet/<geo>.json` (атомарно). Идемпотентен: пересчёт даёт то же.
"""

import glob
import json
import os
import sqlite3
import sys
import tempfile

OUT = os.environ.get("OUT_FACET", "out_facet")
VEC_DB = os.environ.get("LOCAL_VEC_DB", "/root/embed_ab/local_vec.db")
PAGE_MIN = 4  # то же число, что у гейта страниц в pages.py: тоньше — не страница
MIN_EXAMPLES = 5  # меньше примеров — центр полки ненадёжен, полку не строим


def load_vecs(ids, db=None):
    """id мухи → нормированный вектор. Ровно как в dedup.py: тот же формат, та же нормировка."""
    import numpy as np

    c = sqlite3.connect(db or VEC_DB)
    out = {}
    ids = list(ids)
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


def centroids(shelf_ids, vecs):
    """Полка → её центр. Полки с малым числом примеров ОТБРАСЫВАЕМ, а не достраиваем:
    ненадёжный центр стягивает на себя чужие виды, и это не видно в числах."""
    import numpy as np

    names, cent, thin = [], [], {}
    for name, ids in sorted(shelf_ids.items()):
        vs = [vecs[i] for i in ids if i in vecs]
        if len(vs) < MIN_EXAMPLES:
            thin[name] = len(vs)
            continue
        c = np.mean(vs, axis=0)
        names.append(name)
        cent.append(c / (np.linalg.norm(c) or 1.0))
    return names, (np.array(cent) if cent else np.zeros((0, 0))), thin


def assign(views, names, cent, vecs):
    """[(ключ_вида, [id мух])] → {ключ_вида: (полка, близость)}.

    ЧИСТАЯ функция: ни базы, ни файлов — поэтому метод проверяется сторожами на синтетических
    векторах, без `local_vec.db` (он живёт только на VPS).
    Вид без векторов остаётся БЕЗ полки: молча приписывать ему ближайшую — врать.
    """
    import numpy as np

    out = {}
    for key, ids in views:
        vs = [vecs[i] for i in ids if i in vecs]
        if not vs or not len(names):
            out[key] = (None, 0.0)
            continue
        q = np.mean(vs, axis=0)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = cent @ q
        j = int(np.argmax(sims))
        out[key] = (names[j], round(float(sims[j]), 3))
    return out


def _atomic(path, data):
    """Запись через temp+rename: прерванный прогон не оставляет полуфайла (правило проекта)."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def collect(files):
    """Из файлов гео: разметка хвоста (полка → id) и виды-страницы (ключ → id мух)."""
    shelf_ids, views, per_geo = {}, [], {}
    for f in files:
        geo = os.path.basename(f)[:-5]
        if "," in geo:  # мусорные ключи гео («au, nz») — не наша задача, см. шаг 8
            continue
        d = json.load(open(f, encoding="utf-8"))
        for sh in d.get("shelves") or []:
            name = sh.get("shelf")
            if not name:
                continue
            shelf_ids.setdefault(name, []).extend(
                it["id"] for it in (sh.get("items") or []) if it.get("id")
            )
        vs = []
        for i, v in enumerate(d.get("views_by_task") or []):
            items = v.get("items") or []
            if len(items) >= PAGE_MIN:
                key = (geo, i)
                views.append((key, [it["id"] for it in items if it.get("id")]))
                vs.append(i)
        per_geo[geo] = (f, vs)
    return shelf_ids, views, per_geo


def run(paths, dry=False):
    files = sorted(paths)
    shelf_ids, views, per_geo = collect(files)
    need = {i for ids in shelf_ids.values() for i in ids} | {
        i for _, ids in views for i in ids
    }
    vecs = load_vecs(need)
    names, cent, thin = centroids(shelf_ids, vecs)
    got = assign(views, names, cent, vecs)

    print(f"полок с центром: {len(names)} из {len(shelf_ids)}; видов: {len(views)}")
    if thin:
        print(f"  ⚠️ полок без центра (мало примеров): {thin}")
    print(f"  векторов: {len(vecs)} из {len(need)}")
    n_none = sum(1 for v in got.values() if not v[0])
    print(f"  разложено: {len(got) - n_none}, без полки (нет векторов): {n_none}")
    dist = {}
    for sh, _ in got.values():
        dist[sh] = dist.get(sh, 0) + 1
    for k, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {n:5}  {k}")
    if dry:
        print("DRY — в файлы не писали")
        return got

    changed = 0
    for geo, (f, idxs) in per_geo.items():
        d = json.load(open(f, encoding="utf-8"))
        touched = False
        for i in idxs:
            sh, sim = got.get((geo, i), (None, 0.0))
            v = d["views_by_task"][i]
            if sh and (v.get("shelf") != sh or v.get("shelf_sim") != sim):
                v["shelf"], v["shelf_sim"] = sh, sim
                touched = True
        if touched:
            _atomic(f, d)
            changed += 1
    print(f"файлов обновлено: {changed}")
    return got


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv
    paths = (
        sorted(glob.glob(f"{OUT}/*.json"))
        if not args or "--all" in sys.argv
        else [f"{OUT}/{g}.json" for g in args]
    )
    run(paths, dry=dry)
