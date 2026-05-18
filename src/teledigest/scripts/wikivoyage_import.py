#!/usr/bin/env python3
"""
wikivoyage_import.py — Import WikiVoyage country pages into Firestore.

Why:
    Chats give fresh, operational lifehacks but a country has to accumulate
    chat data before МОЗГ can answer anything about it. For brand-new
    countries (or thin ones), there's nothing. WikiVoyage gives a stable
    community-curated baseline — visa info, currency, tourist sights, safety
    warnings — that fills the cold-start gap.

How it works:
    1. Walk the MediaWiki category tree starting at Category:<Country>,
       collecting all destination pages (cities, regions, natural features,
       themes). Stop at MAX_DEPTH to avoid going too deep into street-level
       detail that hurts retrieval quality.
    2. For each page, fetch the wikitext via MediaWiki API.
    3. Parse with mwparserfromhell:
       - Listing templates ({{see}}, {{do}}, {{eat}}, {{drink}}, {{sleep}},
         {{buy}}, {{listing}}) -> one pattern per template, with structured
         fields (name, address, price, hours, content).
       - Section blocks (==Get in==, ==Money==, ==Stay safe==, etc.) ->
         one pattern per paragraph, tag mapped from section heading.
    4. Write to Firestore collection `wikivoyage_base` (separate from
       `wisdom_base` so the chat layer and wiki layer can be refreshed
       independently). Deterministic doc IDs make re-runs idempotent.

Pattern schema in wikivoyage_base:
    {
      title: str,               # listing name or section title
      country: str,             # iso2 lowercase
      tag: str,                 # mapped from wiki section, same vocab as chat-мух
      instruction: str,         # English fact text — Live API translates at query time
      source: 'wikivoyage',
      sourceTitle: str,         # wiki page name e.g. "Bangkok"
      sourceUrl: str,           # https://en.wikivoyage.org/wiki/Bangkok
      importedAt: timestamp,
    }

Usage (inside container or anywhere teledigest is installed):
    python -m teledigest.scripts.wikivoyage_import --country th
    python -m teledigest.scripts.wikivoyage_import --country br --max-depth 1
    python -m teledigest.scripts.wikivoyage_import --country th --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import time
from pathlib import Path

import requests

from teledigest.config import init_config, log

WIKI_API = "https://en.wikivoyage.org/w/api.php"
WIKI_PAGE_BASE = "https://en.wikivoyage.org/wiki/"
USER_AGENT = "teledigest-bot/0.1 (https://github.com/artistalexjlo-rgb/teledigest-my)"

# Be polite to the wiki API. 1.5s pause keeps us well under rate limits.
REQUEST_PAUSE_S = 1.5

# Country code -> WikiVoyage category name. WikiVoyage uses English country
# names verbatim as category roots (Category:Thailand, Category:Brazil).
# Re-export from country_codes — single source of truth shared with the
# migration and Apps Script. Membership in this dict also gates which
# countries the wiki importer accepts (see arg parser below).
from teledigest.country_codes import (  # noqa: E402,F401
    COUNTRY_NAMES_EN as COUNTRY_WIKI_NAME,
)

# Wiki section heading -> our chat-мух tag vocabulary. Keeps wisdom_base
# and wikivoyage_base searchable with the same tag filters.
SECTION_TAG_MAP: dict[str, str] = {
    "understand": "Culture",
    "history": "Culture",
    "regions": "Travel",
    "cities": "Travel",
    "other destinations": "Travel",
    "get in": "Bureaucracy",
    "by plane": "Transport",
    "by train": "Transport",
    "by bus": "Transport",
    "by car": "Transport",
    "by boat": "Transport",
    "get around": "Transport",
    "talk": "Language",
    "see": "Travel",
    "do": "Travel",
    "buy": "Shopping",
    "money": "Finance",
    "costs": "Finance",
    "eat": "Food",
    "drink": "Food",
    "sleep": "Accommodation",
    "learn": "Education",
    "work": "Work",
    "stay safe": "Safety",
    "stay healthy": "Health",
    "respect": "Culture",
    "connect": "Telecom",
    "cope": "Bureaucracy",
    "go next": "Travel",
}

# Listing template name -> tag. Templates carry pre-structured fields.
LISTING_TAG_MAP: dict[str, str] = {
    "see": "Travel",
    "do": "Travel",
    "eat": "Food",
    "drink": "Food",
    "sleep": "Accommodation",
    "buy": "Shopping",
    "listing": "Travel",  # generic
    "marker": "Travel",  # also a listing-like template
}


# --- Wiki API ---------------------------------------------------------------


def _api(session: requests.Session, **params) -> dict:
    """One MediaWiki API call. JSON format, polite pause after.
    Retries on 429 with exponential backoff (30s, 60s, 120s)."""
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    for attempt in range(4):
        resp = session.get(WIKI_API, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 30 * (2**attempt)
            log.warning(
                "WikiVoyage 429 rate limit — waiting %ds (attempt %d/4)",
                wait,
                attempt + 1,
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_PAUSE_S)
        return resp.json()
    resp.raise_for_status()  # raise after exhausting retries
    return resp.json()


def list_category_members(
    session: requests.Session,
    category: str,
    max_depth: int,
) -> list[str]:
    """
    Walk Category:<category> recursively up to max_depth and return all
    page titles (namespace 0). Subcategory traversal is breadth-first so
    we can hit max_depth cleanly.
    """
    visited_cats: set[str] = set()
    visited_pages: set[str] = set()
    queue: list[tuple[str, int]] = [(category, 0)]

    while queue:
        cat, depth = queue.pop(0)
        if cat in visited_cats:
            continue
        visited_cats.add(cat)

        cont = ""
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat}",
                "cmlimit": 500,
                "cmtype": "page|subcat",
            }
            if cont:
                params["cmcontinue"] = cont
            data = _api(session, **params)
            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                title = m["title"]
                ns = m["ns"]
                if ns == 0:
                    visited_pages.add(title)
                elif ns == 14:  # subcategory
                    subcat = title.removeprefix("Category:")
                    if depth + 1 <= max_depth:
                        queue.append((subcat, depth + 1))
            cont = data.get("continue", {}).get("cmcontinue", "")
            if not cont:
                break

    return sorted(visited_pages)


def fetch_wikitext(session: requests.Session, title: str) -> str | None:
    """Return raw wikitext of a page, or None if missing."""
    data = _api(
        session,
        action="query",
        prop="revisions",
        titles=title,
        rvprop="content",
        rvslots="main",
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None
    revs = page.get("revisions", [])
    if not revs:
        return None
    return revs[0].get("slots", {}).get("main", {}).get("content")


# --- Parsing ---------------------------------------------------------------


def _norm_section(name: str) -> str:
    return name.strip().lower()


def _normalize_text(s: str) -> str:
    """Strip wiki-markup leftovers, collapse whitespace."""
    s = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", s)  # [[Link|Label]] -> Label
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)  # [[Plain]] -> Plain
    s = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", s)  # [url label] -> label
    s = re.sub(r"\[https?://\S+\]", "", s)  # bare [url] -> drop
    s = re.sub(r"\{\{[^}]+\}\}", "", s)  # leftover templates
    s = re.sub(r"'''([^']+)'''", r"\1", s)  # bold
    s = re.sub(r"''([^']+)''", r"\1", s)  # italic
    s = re.sub(r"<[^>]+>", "", s)  # html tags
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _template_field(template, name: str) -> str:
    """Return the trimmed string value of a named template parameter, or ''."""
    if not template.has(name):
        return ""
    return _normalize_text(str(template.get(name).value))


def _build_listing_pattern(
    template,
    country: str,
    page_title: str,
    idx: int,
) -> dict | None:
    """Convert a {{see/do/eat/...}} template into a pattern dict."""
    tname = str(template.name).strip().lower()
    tag = LISTING_TAG_MAP.get(tname, "Travel")

    name = _template_field(template, "name") or _template_field(template, "alt")
    if not name:
        return None  # untitled listing — skip
    address = _template_field(template, "address")
    hours = _template_field(template, "hours")
    price = _template_field(template, "price")
    content = _template_field(template, "content")
    phone = _template_field(template, "phone")

    # Quality gate: require either substantive prose OR multiple
    # structured fields. Filters out skeletons like
    # "Jomtien Bowl :: 8 lanes. Hours: Daily, 10:00-02:00."
    # while keeping useful budget-hostel listings with address+price+context.
    has_substantive_content = content and len(content) >= 30
    structured_field_count = sum(bool(x) for x in (address, hours, price, phone))
    if not has_substantive_content and structured_field_count < 2:
        return None

    # Build a single instruction sentence/paragraph.
    parts = [content] if content else []
    if address:
        parts.append(f"Address: {address}.")
    if hours:
        parts.append(f"Hours: {hours}.")
    if price:
        parts.append(f"Price: {price}.")
    if phone:
        parts.append(f"Phone: {phone}.")
    instruction = " ".join(parts).strip()
    if not instruction:
        return None

    return {
        "title": name,
        "country": country,
        "tag": tag,
        "instruction": instruction,
        "_seed_index": idx,
    }


def _build_section_patterns(
    section_name: str,
    paragraphs: list[str],
    country: str,
    page_title: str,
    base_idx: int,
) -> list[dict]:
    """One pattern per non-trivial paragraph in a section, tag from section."""
    norm = _norm_section(section_name)
    tag = SECTION_TAG_MAP.get(norm, "Travel")
    out: list[dict] = []
    for j, p in enumerate(paragraphs):
        clean = _normalize_text(p)
        # Drop very short fragments — bullets without prose, single words.
        if len(clean) < 60:
            continue
        out.append(
            {
                "title": f"{page_title}: {section_name.strip()}",
                "country": country,
                "tag": tag,
                "instruction": clean,
                "_seed_index": base_idx + j,
            }
        )
    return out


def parse_page(
    wikitext: str,
    country: str,
    page_title: str,
) -> list[dict]:
    """Return list of pattern dicts ready for Firestore."""
    import mwparserfromhell as mw

    parsed = mw.parse(wikitext)
    patterns: list[dict] = []
    idx = 0

    # Pass 1: listing templates anywhere in the page.
    for tpl in parsed.filter_templates():
        tname = str(tpl.name).strip().lower()
        if tname in LISTING_TAG_MAP:
            p = _build_listing_pattern(tpl, country, page_title, idx)
            if p:
                patterns.append(p)
                idx += 1

    # Pass 2: section text blocks. Use mwparserfromhell's get_sections to
    # walk top-level sections (==Header==). We skip the lead (untitled
    # section) and listing-only sections since those are covered by Pass 1.
    for section in parsed.get_sections(levels=[2], include_lead=False):
        heading_nodes = section.filter_headings()
        if not heading_nodes:
            continue
        section_name = str(heading_nodes[0].title).strip()
        # Take section as raw text. _normalize_text below strips templates
        # ({{...}}) via regex, so we don't need to mutate the parse tree
        # (which was fragile — Wikicode.remove can ValueError on nested
        # templates whose reference is in a different slice).
        text = str(section)
        # Remove the heading line itself from body text
        text = re.sub(r"^==[^=]+==\s*", "", text, count=1)
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        patterns.extend(
            _build_section_patterns(
                section_name,
                paragraphs,
                country,
                page_title,
                idx,
            )
        )
        idx += len(paragraphs)

    return patterns


# --- Firestore writer -------------------------------------------------------


def _build_firestore_client():
    """Reuse OAuth creds + Firestore project from channel_poster wiring."""
    from teledigest.channel_poster import _build_firestore_client as _bld

    return _bld()


def _doc_id(country: str, page_title: str, idx: int) -> str:
    seed = f"wikivoyage:{country}:{page_title}:{idx}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def write_patterns(
    db, country: str, page_title: str, patterns: list[dict]
) -> tuple[int, int]:
    """Returns (written, skipped). Skipped = already-exists (idempotent).

    Computes text-embedding-004 vectors for new docs in batches so vector
    search works immediately after import (no separate backfill needed).
    Falls back gracefully if embedding API is unavailable.
    """
    if not patterns:
        return 0, 0
    # Cipher-fix: write to wikivoyage_base with v2 unified-text embedding.
    # Bot reads same collection. Backup of pre-migration state lives in
    # wikivoyage_v1_backup.
    coll = db.collection("wikivoyage_base")
    now = dt.datetime.now(dt.timezone.utc)
    url = WIKI_PAGE_BASE + page_title.replace(" ", "_")

    # --- Phase 1: determine which docs are new (need write) ---
    new_docs: list[tuple[str, dict]] = []  # (doc_id, pattern)
    skipped = 0
    for p in patterns:
        idx = p.pop("_seed_index")
        doc_id = _doc_id(country, page_title, idx)
        ref = coll.document(doc_id)
        snap = ref.get()
        if snap.exists:
            skipped += 1
            continue
        new_docs.append((doc_id, p))

    if not new_docs:
        return 0, skipped

    # --- Phase 2: compute embeddings for new docs (v2 cipher) ---
    from teledigest.country_codes import country_full_name_en
    from teledigest.gemini_brain import (
        _EMBEDDING_MODEL_TAG_V2,
        compute_document_embeddings_v2,
    )

    country_full = country_full_name_en(country)

    def _embed_text(p: dict) -> str:
        parts = [country_full]
        title = (p.get("title") or "").strip()
        tag = (p.get("tag") or "").strip()
        instr = (p.get("instruction") or "").strip()
        if title:
            parts.append(title)
        if tag:
            parts.append(tag)
        if instr:
            parts.append(instr)
        return ". ".join(parts)

    texts = [_embed_text(nd) for _, nd in new_docs]
    try:
        embeddings = compute_document_embeddings_v2(texts)
    except Exception as emb_err:
        log.warning(
            "write_patterns: embedding batch failed (%s) — writing without vectors",
            emb_err,
        )
        embeddings = [None] * len(new_docs)

    # --- Phase 3: write new docs with embeddings ---
    written = 0
    for (doc_id, p), emb, text in zip(new_docs, embeddings, texts):
        instr = p.get("instruction") or ""
        payload: dict = {
            **p,
            "source": "wikivoyage",
            "sourceTitle": page_title,
            "sourceUrl": url,
            "importedAt": now,
            "embedded_text": text,
            "embedding_model": _EMBEDDING_MODEL_TAG_V2,
            "text_length": len(instr),
            "needs_chunking": len(instr) > 500,
        }
        if emb is not None:
            from google.cloud.firestore_v1.vector import Vector

            payload["embedding"] = Vector(emb)
        coll.document(doc_id).set(payload)
        written += 1

    return written, skipped


# --- Main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--country",
        required=True,
        help="ISO 3166-1 alpha-2 country code (lowercase), "
        "e.g. 'th'. Must have an entry in COUNTRY_WIKI_NAME.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Subcategory recursion depth (default 2). "
        "0 = only the top country page's direct members.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on number of pages to process (0 = all). "
        "Useful for first dry-run sanity checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk + parse + count, do not write Firestore",
    )
    parser.add_argument(
        "--config",
        default="/config/teledigest.conf",
        help="Path to teledigest.conf (default container path)",
    )
    args = parser.parse_args()

    country = args.country.lower()
    if country not in COUNTRY_WIKI_NAME:
        log.error(
            "Country '%s' not in COUNTRY_WIKI_NAME map. "
            "Add it to wikivoyage_import.py and retry.",
            country,
        )
        return 2
    wiki_name = COUNTRY_WIKI_NAME[country]

    init_config(Path(args.config))

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    log.info(
        "WikiVoyage import: country=%s wiki_category=%s max_depth=%d",
        country,
        wiki_name,
        args.max_depth,
    )

    pages = list_category_members(session, wiki_name, args.max_depth)
    log.info(
        "Found %d destination pages in Category:%s tree (depth<=%d).",
        len(pages),
        wiki_name,
        args.max_depth,
    )
    if args.limit and len(pages) > args.limit:
        log.info("Limiting to first %d pages for this run.", args.limit)
        pages = pages[: args.limit]

    if args.dry_run:
        for p in pages:
            print(f"WOULD FETCH: {p}")
        log.info("Dry run — would process %d pages.", len(pages))
        return 0

    db = _build_firestore_client()
    total_written = 0
    total_skipped = 0
    total_patterns = 0
    failed_pages: list[str] = []
    skipped_empty: list[str] = []
    for i, title in enumerate(pages, start=1):
        try:
            wt = fetch_wikitext(session, title)
            if not wt:
                log.warning("[%d/%d] %s: no wikitext, skipping", i, len(pages), title)
                skipped_empty.append(title)
                continue
            patterns = parse_page(wt, country, title)
            written, skipped = write_patterns(db, country, title, patterns)
            total_written += written
            total_skipped += skipped
            total_patterns += len(patterns)
            log.info(
                "[%d/%d] %s -> %d patterns (wrote=%d skipped=%d)",
                i,
                len(pages),
                title,
                len(patterns),
                written,
                skipped,
            )
        except Exception as e:
            log.exception("[%d/%d] %s: parse/write failed: %s", i, len(pages), title, e)
            failed_pages.append(title)

    log.info(
        "Import complete: country=%s pages_in_tree=%d "
        "pages_parsed_ok=%d pages_failed=%d pages_no_wikitext=%d "
        "patterns_total=%d wrote_new=%d skipped_existing=%d",
        country,
        len(pages),
        len(pages) - len(failed_pages) - len(skipped_empty),
        len(failed_pages),
        len(skipped_empty),
        total_patterns,
        total_written,
        total_skipped,
    )
    if failed_pages:
        log.warning(
            "Failed pages (re-run will retry these): %s", ", ".join(failed_pages)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
