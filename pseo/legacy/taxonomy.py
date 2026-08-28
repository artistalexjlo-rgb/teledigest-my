"""
taxonomy.py — этап-2 индекса (концептуально в свипере): раскладывает мух по
КАНОНИЧЕСКИМ категориям (WikiVoyage × expat) + секциям. Детерминированно (bge-m3,
без Gemini). Пишет таблицу tax(pattern_id → geo → category → section), которую
ЧИТАЕТ билдер. Билдер сам НЕ кластерует — организация живёт тут.

Канон — фиксированный список (не выводим геометрией — мировая практика, велосипед
изобретён). Назначение мух — семантикой (чинит грязный тег, разбивает Travel-помойку).

Запуск (VPS): /root/embed_ab/venv/bin/python taxonomy.py [--geo br,vn]
"""

import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DB = "/home/teledigest/data/messages_fts.db"
VEC = "/root/embed_ab/local_vec.db"
TAXDB = "/root/pseo_builder/taxonomy.db"
MIN_CAT = 12  # категория гео с <12 мухами — тонко, не строим (парк на уровне индекса)

# Канон: категория → описание для семантического назначения (bge-m3). WikiVoyage×expat.
CANON = {
    "documents": "visas, residence permit VNZH, CPF tax id, legalization apostille of documents, immigration paperwork, entry requirements, passport",
    "money": "banking, opening a bank account, currency exchange, cards, payments PIX, money transfers, taxes",
    "housing": "renting an apartment, real estate, rental contract, deposit, where to live, neighborhoods",
    "health": "healthcare, doctors, medical insurance, pharmacies, vaccination yellow fever, hospitals",
    "transport": "metro subway, city buses, taxi, ride-sharing Uber 99, intercity bus tickets, airport transfer, renting and driving a car, transit fares",
    "safety": "crime, safe areas, scams fraud, what not to do, personal security on the street",
    "connectivity": "SIM card, mobile internet, phone plan, eSIM, staying connected",
    "shopping": "buying goods, prices, markets, electronics, groceries, where to shop",
    "food": "restaurants, local food, eating out, food delivery, cuisine",
    "attractions": "sightseeing, tourist attractions, what to see and do, tours, museums, landmarks, beaches, trip itinerary and route planning",
    "culture": "language barrier, local customs, etiquette, communication, adapting to local life",
    "work": "employment, jobs, freelancing, remote work, business, work taxes",
    "education": "schools, universities, courses, language learning",
    "shipping": "sending postal parcels and packages, courier delivery of physical items",
}


def load_vectors():
    """Все bge-m3 векторы в память ОДНИМ запросом (per-fly запросы тормозят)."""
    v = sqlite3.connect(VEC)
    d = {}
    for did, blob in v.execute("SELECT doc_id, v FROM vec WHERE dim=1024"):
        d[did] = np.frombuffer(blob, dtype=np.float32)
    v.close()
    return d


def main():
    only = None
    if "--geo" in sys.argv:
        only = set(sys.argv[sys.argv.index("--geo") + 1].split(","))
    model = SentenceTransformer("BAAI/bge-m3")
    cats = list(CANON)
    C = model.encode([CANON[c] for c in cats], normalize_embeddings=True)
    vecs = load_vectors()
    m = sqlite3.connect(DB)
    geos = [
        r[0]
        for r in m.execute(
            "SELECT DISTINCT country FROM extracted_patterns WHERE country!='any' AND country IS NOT NULL"
        )
    ]
    if only:
        geos = [g for g in geos if g in only]

    tax = sqlite3.connect(TAXDB)
    tax.execute(
        "CREATE TABLE IF NOT EXISTS tax(pattern_id TEXT PRIMARY KEY, geo TEXT, "
        "category TEXT, section INT, ts TEXT)"
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = 0
    for geo in geos:
        ids = [
            r[0]
            for r in m.execute(
                "SELECT id FROM extracted_patterns WHERE country=? AND ai_lesson IS NOT NULL "
                "AND length(ai_lesson)>140",
                (geo,),
            )
            if r[0] in vecs
        ]
        if len(ids) < MIN_CAT:
            continue
        V = np.vstack([vecs[i] for i in ids])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        assign = np.argmax(V @ C.T, axis=1)
        for ci, cat in enumerate(cats):
            sel = [j for j in range(len(ids)) if assign[j] == ci]
            if len(sel) < MIN_CAT:
                continue  # тонкая категория — не в индекс (парк на уровне индекса)
            sv = V[sel]
            k = max(1, min(8, len(sel) // 12))
            lab = (
                KMeans(n_clusters=k, n_init=3, random_state=0).fit(sv).labels_
                if k > 1
                else [0] * len(sel)
            )
            for jj, sidx in enumerate(sel):
                tax.execute(
                    "INSERT OR REPLACE INTO tax VALUES(?,?,?,?,?)",
                    (ids[sidx], geo, cat, int(lab[jj]), ts),
                )
            total += len(sel)
    tax.commit()
    # сводка geo→category→count для wire.py (навигация из таксономии, на десктоп)
    import json as _json

    summ = {}
    for geo, cat, cnt in tax.execute(
        "SELECT geo, category, COUNT(*) FROM tax GROUP BY geo, category"
    ):
        summ.setdefault(geo, {})[cat] = cnt
    with open("/root/pseo_builder/taxonomy_summary.json", "w", encoding="utf-8") as f:
        _json.dump(summ, f, ensure_ascii=False)
    print(f"taxonomy: {total} мух разложено, гео {len(geos)}")
    print("топ гео×категория:")
    for geo, cat, cnt in tax.execute(
        "SELECT geo,category,COUNT(*) c FROM tax GROUP BY geo,category ORDER BY c DESC LIMIT 18"
    ):
        print(f"  {geo}/{cat}: {cnt}")
    tax.close()


if __name__ == "__main__":
    main()
