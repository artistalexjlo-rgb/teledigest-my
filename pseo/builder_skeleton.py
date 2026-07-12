"""
pSEO builder SKELETON — reference, NOT wired yet.
Источник: C:\\Users\\Servak\\Downloads\\builder.py (от помощника, 2026-06-14).
Сохранён сюда, чтобы не потерялся. Реализация — отдельный пакет teledigest-my/pseo/.

ИНВАРИАНТЫ (не трогать при реализации — см. memory/roadmap_pseo_builder.md):
1. Таксономия append-only, page-ID детерминированы, URL стабильны.
   Кластеризация (HDBSCAN) — ТОЛЬКО офлайн-discovery новых интентов, НЕ page-ID.
2. Изолировано от горячего пути бота: свой пакет, свой крон, свой деплой (CF Pages).
3. embed() — ТОЛЬКО через наш rate-limited путь (gemini_brain compute_*_v2,
   use_persistent_quota=True). Прямой Gemini = риск GCP-suspend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

# --- config -----------------------------------------------------------------

OUT = pathlib.Path("out")
N_MIN = 4  # min real Q&A in a bucket to publish
C_MIN = 0.55  # min avg cosine of members to centroid (bucket tightness)
NEAR_DUP_TAU = 0.90  # >= this vs an existing page summary -> merge/skip

# fixed intent taxonomy: seed phrases embedded ONCE, used to assign QA -> intent
INTENTS = {
    "visa": "visa residency permit rules requirements",
    "housing": "rent apartment long term housing prices",
    "coworking": "coworking space wifi desk monthly",
    "banking": "open bank account card transfers",
    "transport": "scooter rental driving license transport",
}

# demand-gated languages per geo (do NOT render every lang everywhere)
LANGS_BY_GEO = {
    "phuket": ["ru", "en"],
    "uruguay": ["ru", "en", "pt", "es"],
    "montevideo": ["ru", "en", "pt", "es"],
}

# --- data seams (wire to your stack) ----------------------------------------


def embed(text: str) -> list[float]:
    """Gemini embedding-2, 1536-dim. STUB."""
    ...


def qdrant_search(
    collection: str, vector: list[float], top_k: int, flt: dict | None = None
):
    """STUB. Return [(id, score, payload), ...]."""
    ...


def qdrant_upsert(collection: str, point_id: str, vector: list[float], payload: dict):
    """STUB."""
    ...


def load_qa(geo: str) -> list[dict]:
    """All Q&A rows for a geo: {'id','text','vector','intent'? ,...}. STUB."""
    ...


def extract_facts(qa_rows: list[dict], langs: list[str]) -> list[dict]:
    """
    Gemini structured output -> language-neutral fact records, with value
    translations cached for `langs` HERE (not at render time):
      {'subject','attribute','value', 'translations': {lang: str},
       'source_ids':[...], 'date':..., 'value_tag':'commodity'|'hot_intent'}
    STUB.
    """
    ...


# --- taxonomy assignment (stable, deterministic) ----------------------------


def cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb + 1e-9)


INTENT_VECS = {k: embed(v) for k, v in INTENTS.items()}  # embedded once


def assign_intent(qa_vec: list[float]) -> str:
    return max(INTENT_VECS, key=lambda k: cosine(qa_vec, INTENT_VECS[k]))


def centroid(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


# --- gating -----------------------------------------------------------------


def coherence(vecs: list[list[float]], c: list[float]) -> float:
    return sum(cosine(v, c) for v in vecs) / len(vecs)


def is_near_dup(summary_vec: list[float]) -> str | None:
    hits = qdrant_search("page_summaries", summary_vec, top_k=1)
    if hits and hits[0][1] >= NEAR_DUP_TAU:
        return hits[0][0]  # existing page id to merge into
    return None


# --- rendering --------------------------------------------------------------

env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())


def fact_hash(facts: list[dict]) -> str:
    blob = json.dumps(facts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def render_page(
    geo: str, intent: str, lang: str, facts: list[dict], related: list[dict]
):
    # localize: pure dict lookup, zero runtime MT
    loc_facts = [{**f, "value": f["translations"].get(lang, f["value"])} for f in facts]
    alts = [{"lang": l, "href": f"/{l}/{geo}/{intent}/"} for l in LANGS_BY_GEO[geo]]
    html = env.get_template("page.html.j2").render(
        lang=lang,
        geo=geo,
        intent=intent,
        facts=loc_facts,
        related=related,
        hreflangs=alts,
        jsonld=build_qapage_jsonld(loc_facts),  # Schema.org QAPage
    )
    p = OUT / lang / geo / intent / "index.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")


def build_qapage_jsonld(facts: list[dict]) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "QAPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f["attribute"],
                "acceptedAnswer": {"@type": "Answer", "text": f["value"]},
            }
            for f in facts
        ],
    }
    return json.dumps(data, ensure_ascii=False)


def related_buckets(centroid_vec: list[float], self_id: str, k: int = 6) -> list[dict]:
    hits = qdrant_search("bucket_centroids", centroid_vec, top_k=k + 1)
    return [
        {"geo": h[2]["geo"], "intent": h[2]["intent"]} for h in hits if h[0] != self_id
    ][:k]


# --- main pipeline ----------------------------------------------------------


def build():
    seen_hashes = load_built_hashes()  # {bucket_id: fact_hash} from last run. STUB.
    sitemap: dict[str, list[str]] = {}

    for geo in LANGS_BY_GEO:
        qa = load_qa(geo)
        for row in qa:
            row["intent"] = assign_intent(row["vector"])

        for intent in INTENTS:
            bucket = [r for r in qa if r["intent"] == intent]
            if len(bucket) < N_MIN:
                continue
            vecs = [r["vector"] for r in bucket]
            c = centroid(vecs)
            if coherence(vecs, c) < C_MIN:
                continue

            facts = extract_facts(bucket, LANGS_BY_GEO[geo])
            h = fact_hash(facts)
            bucket_id = f"{geo}:{intent}"

            # incremental: skip unchanged buckets
            if seen_hashes.get(bucket_id) == h:
                continue

            summary_vec = embed(" ".join(f["value"] for f in facts))
            dup = is_near_dup(summary_vec)
            if dup:
                merge_into(dup, facts)  # STUB
                continue

            qdrant_upsert(
                "bucket_centroids", bucket_id, c, {"geo": geo, "intent": intent}
            )
            qdrant_upsert(
                "page_summaries", bucket_id, summary_vec, {"geo": geo, "intent": intent}
            )

            related = related_buckets(c, bucket_id)
            for lang in LANGS_BY_GEO[geo]:
                render_page(geo, intent, lang, facts, related)
                sitemap.setdefault(lang, []).append(f"/{lang}/{geo}/{intent}/")

            seen_hashes[bucket_id] = h

    write_sitemaps(sitemap)
    save_built_hashes(seen_hashes)  # STUB


# --- sitemaps (sharded per lang) --------------------------------------------


def write_sitemaps(sitemap: dict[str, list[str]]):
    BASE = "https://example.com"
    index = []
    for lang, urls in sitemap.items():
        body = "".join(f"<url><loc>{BASE}{u}</loc></url>" for u in urls)
        (OUT / f"sitemap_{lang}.xml").write_text(
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</urlset>',
            encoding="utf-8",
        )
        index.append(f"<sitemap><loc>{BASE}/sitemap_{lang}.xml</loc></sitemap>")
    (OUT / "sitemap.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f'{"".join(index)}</sitemapindex>',
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
