"""
gemini_brain.py — МОЗГ via Gemini Live API + Firestore wisdom_base.

Flow:
1. Query Firestore wisdom_base: pull the top-N most recent docs across
   ALL countries. No country filter on the database side — Gemini sees
   broad context and picks the relevant facts itself from the query.
2. Format instruction fields as numbered context (each entry shows its
   country/tag so the model can ground references).
3. Open Gemini Live API session, send system prompt + context + question.
4. Receive streamed text, concatenate, return as Russian answer to user.
5. On any Live API failure → fallback to sync Gemini → return empty so
   caller can fall back to DeepSeek+SQLite path.

Why no country filter:
- Earlier version filtered Firestore by country derived from chat tag.
  Real chats (e.g. luky_channel) are not always 1-1 with a country.
  Asking "как получить CPF" in an AR-tagged chat hid all BR facts.
- Gemini is perfectly capable of reading "CPF" or "пикс" and answering
  from the right-country facts in the context. We just need to give it
  enough context — that's what the limit bump is for.

Auth: GEMINI_API_KEY from env or [gemini] api_key in config.
"""

from __future__ import annotations

import asyncio

from .config import get_config, log

_BRAIN_SYSTEM = """\
You are an assistant bot for an expat community chat called "МОЗГ".
Answer the user's question using ONLY the facts in the knowledge base below.

The knowledge base mixes facts from many countries. Each entry has a
[n. title | tag | country | source] header so you can tell where it applies
and where it came from. There are two sources:
- "База данных" — facts collected from real user experience and discussion.
- "WikiVoyage" — community-curated travel encyclopedia (stable baseline).

When citing a fact, prefer wording like "по WikiVoyage..." for wiki entries
and neutral language for "База данных" entries. If both sources agree —
just answer directly without citing.

Behavior:
- Answer in Russian, conversational and informal.
- Be specific: include exact prices, addresses, timelines, service names, steps.
- KEEP IT SHORT: 2-4 sentences MAX. The user is on a phone — no walls of text.
- If the user needs depth, finish with: "хочешь подробнее — спроси конкретнее".
- If this is a follow-up turn (the conversation already has prior exchanges)
  — use that context. Don't ask the user to repeat themselves.
- If the question is ambiguous (could apply to multiple countries / multiple
  scenarios in the knowledge base): list the most likely interpretations
  in 1-2 sentences each and append: "уточни, что из этого тебе нужно".
  Do NOT pick one arbitrarily.
- If the knowledge base really has nothing — reply exactly:
  "в базе пока нет информации по этому вопросу"
- Plain text only — no Markdown, no bullet symbols, no formatting.
- Do NOT invent or guess information.
"""


def _build_firestore_client():
    """Build a Firestore client using service account."""
    from .google_auth import build_firestore_client

    return build_firestore_client()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_EMBEDDING_MODELS = ["gemini-embedding-2", "gemini-embedding-001"]
_EMBEDDING_DIM = 768

# --- v2 (cipher-fix) ---
# Single canonical model + native 3072 dim + task_type asymmetric.
# See roadmap_embedding_cipher_fix.md for why these constants matter.
_EMBEDDING_MODEL_V2 = "gemini-embedding-2"
# Firestore vector index hard-caps at 2048 dim. embedding-2 native is 3072
# but supports MRL truncation to 1536 (or 768) without retraining. 1536 is
# the largest supported MRL point that fits Firestore.
_EMBEDDING_DIM_V2 = 1536
_EMBEDDING_MODEL_TAG_V2 = "gemini-embedding-2-1536"


def _get_embedding_api_key() -> str:
    """Return Gemini API key from config or env (single-key path, legacy)."""
    import os

    cfg = get_config()
    return str(
        getattr(cfg.gemini, "api_key", None) or os.environ.get("GEMINI_API_KEY", "")
    )


def _get_embedding_api_keys() -> list[str]:
    """Return all available Gemini API keys for embedding.

    Sources, in order of preference:
      1. GEMINI_API_KEYS env var (comma-separated) — for multi-key migration
      2. GEMINI_API_KEY env var or [gemini] api_key in config — single key

    Each free-tier key has its own daily quota on gemini-embedding-2 (~30K RPD),
    so rotating across N keys multiplies throughput linearly during bulk
    re-embedding without paid tier.
    """
    import os

    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    if raw_multi:
        keys = [k.strip() for k in raw_multi.split(",") if k.strip()]
        if keys:
            return keys
    single = _get_embedding_api_key()
    return [single] if single else []


# Round-robin pointer for the multi-key pool. Initialized lazily on first
# use to a random index, so concurrent instances (e.g. live bot + migration
# script) don't collide on the same key at start.
_key_rr_idx: int | None = None

# Per-key cooldown bookkeeping. When Gemini responds with retry_after we
# remember the wake-time and skip the key (no HTTP call) until then. Avoids
# the self-inflicted pulemyot where each 429 ticks the RPM counter and
# *extends* the cooldown that Gemini already set.
_key_cooldown_until: dict[int, float] = {}


def _ensure_rr_idx_initialized(pool_size: int) -> None:
    """Pick a random starting index on first call. Idempotent."""
    global _key_rr_idx
    if _key_rr_idx is None and pool_size > 0:
        import random

        _key_rr_idx = random.randint(0, pool_size - 1)
    elif _key_rr_idx is None:
        _key_rr_idx = 0


def _is_key_available(idx: int) -> bool:
    """True if key is not in a known cooldown window."""
    import time as _time

    return _time.time() >= _key_cooldown_until.get(idx, 0.0)


def _cool_key(idx: int, seconds: float) -> None:
    """Mark a key as cooling for the next `seconds`. Adds a small buffer
    so we don't try exactly at expiry and re-trigger another 429."""
    import time as _time

    _key_cooldown_until[idx] = _time.time() + max(seconds + 1.0, 2.0)


def _soonest_wake() -> float:
    """Seconds until the earliest-cooling key becomes available. 0 if any
    key is already available or none are tracked."""
    import time as _time

    if not _key_cooldown_until:
        return 0.0
    now = _time.time()
    return max(0.0, min(_key_cooldown_until.values()) - now)


# How long to sleep between attempts within ONE sweep so a burst of 7 keys
# doesn't fire in 1.2 seconds and ping-pong RPM counters.
_INTRA_SWEEP_GAP_S = 1.0
# When Gemini returns 429 without an explicit retry_after, assume RPM
# bucket needs ~30s to drain. Conservative default.
_DEFAULT_COOLDOWN_S = 30.0


def _compute_embedding(text: str, model_idx: int = 0) -> list[float] | None:
    """Compute embedding vector for a single text. On 429 falls back to the
    sibling embedding model so a single retrieval doesn't fail just because
    one model's daily quota is exhausted."""
    from google import genai

    api_key = _get_embedding_api_key()
    if not api_key:
        log.warning("МОЗГ embeddings: no GEMINI_API_KEY")
        return None
    client = genai.Client(api_key=api_key)

    models_to_try = [
        _EMBEDDING_MODELS[model_idx % len(_EMBEDDING_MODELS)],
        _EMBEDDING_MODELS[(model_idx + 1) % len(_EMBEDDING_MODELS)],
    ]
    for model in models_to_try:
        try:
            result = client.models.embed_content(
                model=model,
                contents=text,
                config={"output_dimensionality": _EMBEDDING_DIM},
            )
            embeddings = result.embeddings
            if not embeddings:
                return None
            return list(embeddings[0].values or [])
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                log.warning("МОЗГ: %s quota exhausted — trying next", model)
                continue
            log.warning("МОЗГ: embed_content failed: %s", e)
            return None
    log.warning("МОЗГ: all embedding models exhausted")
    return None


def compute_embeddings_batch(
    texts: list[str], model_idx: int = 0
) -> list[list[float] | None]:
    """Compute embeddings for a batch of texts.

    Alternates between two embedding models (model_idx % 2).
    On 429 (quota exhausted) automatically falls back to the other model.
    """
    if not texts:
        return []

    from google import genai

    api_key = _get_embedding_api_key()
    if not api_key:
        return [None] * len(texts)
    client = genai.Client(api_key=api_key)

    # Try preferred model first, then fall back to the other one on 429.
    models_to_try = [
        _EMBEDDING_MODELS[model_idx % len(_EMBEDDING_MODELS)],
        _EMBEDDING_MODELS[(model_idx + 1) % len(_EMBEDDING_MODELS)],
    ]
    for model in models_to_try:
        try:
            result = client.models.embed_content(
                model=model,
                contents=texts,  # type: ignore[arg-type]
                config={"output_dimensionality": _EMBEDDING_DIM},
            )
            embeddings = result.embeddings or []
            return [list(e.values or []) for e in embeddings]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                log.warning("МОЗГ: model %s quota exhausted — trying next", model)
                continue
            log.warning("МОЗГ: batch embed failed (model=%s): %s", model, e)
            return [None] * len(texts)

    log.warning("МОЗГ: all embedding models exhausted quota")
    return [None] * len(texts)


# ---------------------------------------------------------------------------
# Embeddings v2 — single canonical model + task_type asymmetric (cipher fix)
# ---------------------------------------------------------------------------


def _embed_v2(
    texts: list[str],
    task_type: str,
    dim: int = _EMBEDDING_DIM_V2,
) -> list[list[float] | None]:
    """Compute embeddings via the canonical v2 setup.

    task_type: "RETRIEVAL_QUERY" for user queries,
               "RETRIEVAL_DOCUMENT" for indexed documents.
    Asymmetric task_type is what makes short queries land near long structured
    docs — without it the embedder treats both sides symmetrically and recall
    on short-query / long-doc pairs collapses.
    """
    if not texts:
        return []
    if task_type not in ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"):
        raise ValueError(
            f"task_type must be RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT, got {task_type!r}"
        )

    from google import genai
    from google.genai import types as genai_types

    keys = _get_embedding_api_keys()
    if not keys:
        log.warning("v2 embed: no API keys (GEMINI_API_KEY/GEMINI_API_KEYS missing)")
        return [None] * len(texts)

    # Config MUST be the typed EmbedContentConfig — passing a dict silently
    # drops `task_type` (verified on VPS: cos(same text as QUERY vs DOCUMENT)
    # returned 1.0000, meaning task_type was ignored). Known SDK quirk.
    cfg = genai_types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=dim,
    )

    # Lazily build one Client per key (round-robin across calls).
    clients = {k: genai.Client(api_key=k) for k in keys}
    _ensure_rr_idx_initialized(len(keys))
    global _key_rr_idx

    import re as _re
    import time as _time

    _CODE_RE = _re.compile(r"\b(429|400|401|403)\b|RESOURCE_EXHAUSTED")
    # Parse "retry in N.Ns" hint embedded in SDK exception message.
    _RETRY_RE = _re.compile(r"retry in ([\d.]+)s", _re.I)

    # gemini-embedding-2 via google-genai SDK: passing a list as `contents`
    # is interpreted as a single multi-part document, not a batch — it returns
    # one vector regardless of list size. So call per-text individually.
    out: list[list[float] | None] = []
    for text in texts:
        vec: list[float] | None = None
        last_err: Exception | None = None
        fatal_error = False  # non-quota exception → no retries on this text
        deadline = _time.time() + 60.0  # single-text deadline
        attempts_this_text = 0
        while _time.time() < deadline and vec is None and not fatal_error:
            any_available = False
            made_attempt = False
            for offset in range(len(keys)):
                idx = (_key_rr_idx + offset) % len(keys)
                if not _is_key_available(idx):
                    continue
                any_available = True
                if attempts_this_text > 0:
                    _time.sleep(_INTRA_SWEEP_GAP_S)
                attempts_this_text += 1
                made_attempt = True
                try:
                    result = clients[keys[idx]].models.embed_content(
                        model=_EMBEDDING_MODEL_V2,
                        contents=text,
                        config=cfg,
                    )
                    _key_rr_idx = (idx + 1) % len(keys)
                    embeddings = result.embeddings or []
                    candidate = list(embeddings[0].values or []) if embeddings else None
                    if candidate is None:
                        last_err = RuntimeError(
                            f"empty embeddings on 200 OK (key #{idx})"
                        )
                        continue
                    vec = candidate
                    break
                except Exception as e:
                    _key_rr_idx = (idx + 1) % len(keys)
                    last_err = e
                    err = str(e)
                    if _CODE_RE.search(err):
                        m = _RETRY_RE.search(err)
                        wait = float(m.group(1)) if m else _DEFAULT_COOLDOWN_S
                        _cool_key(idx, wait)
                        log.warning(
                            "v2 embed: key #%d %s (cooling %.1fs)",
                            idx,
                            err.split("\n")[0][:120],
                            wait,
                        )
                        continue
                    # Non-quota error — abandon this text entirely.
                    fatal_error = True
                    break
            if vec is not None or fatal_error:
                break
            if not any_available:
                wait = _soonest_wake()
                if wait <= 0:
                    continue
                log.warning(
                    "v2 embed: all keys cooling, sleeping %.1fs", min(wait, 90.0)
                )
                _time.sleep(min(wait, 90.0))
            elif not made_attempt:
                break
        if vec is None and last_err is not None:
            log.warning(
                "v2 embed failed (task=%s, text_len=%d): %s",
                task_type,
                len(text),
                last_err,
            )
        out.append(vec)
    return out


def compute_query_embedding_v2(
    text: str, dim: int = _EMBEDDING_DIM_V2
) -> list[float] | None:
    """Embed a single user query for retrieval (task_type=RETRIEVAL_QUERY)."""
    if not text or not text.strip():
        return None
    results = _embed_v2([text], "RETRIEVAL_QUERY", dim)
    return results[0] if results else None


def compute_document_embeddings_v2(
    texts: list[str], dim: int = _EMBEDDING_DIM_V2
) -> list[list[float] | None]:
    """Embed a batch of documents for indexing (task_type=RETRIEVAL_DOCUMENT)."""
    return _embed_v2(texts, "RETRIEVAL_DOCUMENT", dim)


def _parse_retry_after_seconds(body_text: str) -> float | None:
    """Extract 'Please retry in N.Ns' hint from Gemini 429 error body."""
    import re

    m = re.search(r"retry in ([\d.]+)s", body_text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _embed_v2_rest_batch(
    texts: list[str],
    task_type: str,
    dim: int,
    api_key: str,
    model: str = _EMBEDDING_MODEL_V2,
    timeout: int = 60,
) -> tuple[int, list[list[float] | None] | None, float | None]:
    """Single :batchEmbedContents REST call carrying N texts.

    Why direct REST: google-genai SDK's embed_content with contents=[t1, t2]
    silently concatenates into ONE multi-part doc and returns ONE vector
    (verified on VPS). The real "N texts → N vectors in one HTTP call" lives
    only on the batchEmbedContents REST endpoint. One such call costs 1 RPD,
    so batching 100 texts per call effectively multiplies free-tier capacity
    by 100x.

    Returns (http_status, list-of-vectors-or-None-per-text, retry_after_seconds).
    status<0 on network failure with vectors=None. retry_after_seconds is set
    when Gemini's 429 body carries 'Please retry in N.Ns' (RPM bucket refill).
    """
    import requests as _req

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":batchEmbedContents?key={api_key}"
    )
    body = {
        "requests": [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": dim,
            }
            for t in texts
        ]
    }
    try:
        resp = _req.post(url, json=body, timeout=timeout)  # type: ignore[arg-type]
    except Exception as exc:
        log.warning("v2 batch REST: network error: %s", exc)
        return -1, None, None
    if resp.status_code != 200:
        retry_after = _parse_retry_after_seconds(resp.text)
        return resp.status_code, None, retry_after
    data = resp.json()
    embeds = data.get("embeddings", [])
    out: list[list[float] | None] = []
    for emb in embeds:
        vals = emb.get("values") or []
        out.append(list(vals) if vals else None)
    # Defend against mismatched length — pad/truncate to match input.
    while len(out) < len(texts):
        out.append(None)
    return 200, out[: len(texts)], None


def _estimate_tokens(text: str) -> int:
    """Rough token count for a text. Gemini's tokenizer averages ~4 chars
    per token on mixed text; we divide by 3 to overestimate so a chunk
    stays under TPM rather than burst over it. False positives (chunk
    smaller than necessary) cost a bit of throughput; false negatives
    (chunk over TPM cap) cost the whole chunk via 429."""
    return max(1, len(text) // 3)


def _split_by_token_budget(
    texts: list[str],
    token_budget: int,
    max_count: int,
) -> list[list[str]]:
    """Pack `texts` into chunks where sum(tokens) <= token_budget and
    len(chunk) <= max_count. Documents that alone exceed budget go in
    their own single-element chunk (Gemini will either accept or reject,
    but won't poison other texts)."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        tk = _estimate_tokens(t)
        if tk >= token_budget:
            if cur:
                chunks.append(cur)
                cur, cur_tokens = [], 0
            chunks.append([t])
            continue
        if cur_tokens + tk > token_budget or len(cur) >= max_count:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += tk
    if cur:
        chunks.append(cur)
    return chunks


def _process_one_chunk(
    chunk: list[str],
    dim: int,
    keys: list[str],
) -> list[list[float] | None] | None:
    """Send one chunk through the multi-key pool.

    Two cooperating mechanisms protect us from the self-inflicted RPM
    burnout we observed in earlier logs:

    1. Per-key cooldown — if a key returned 429 recently, we skip it
       without an HTTP call until its wake-time. Avoids ticking its RPM
       counter again (which would extend the cooldown Gemini already set).
    2. Intra-sweep gap — 1 second between actual HTTP attempts within one
       sweep. Spreads a 7-key sweep over ~7 seconds instead of ~1.2, so
       retry_after windows on early keys can drain before we reach them
       in the next sweep.

    Returns the embedding list on success, or None if no key in the pool
    can serve the chunk after waking up cooldowns. Caller can then split.
    """
    import time as _time

    _ensure_rr_idx_initialized(len(keys))
    global _key_rr_idx

    # Hard cap on total wait inside one chunk. Without this, a fully
    # exhausted pool would loop indefinitely.
    deadline = _time.time() + 180.0

    last_nonretriable_status: int | None = None
    attempt_count = 0  # actual HTTP attempts (not skipped-cooling)

    while _time.time() < deadline:
        any_available = False
        for offset in range(len(keys)):
            idx = (_key_rr_idx + offset) % len(keys)
            if not _is_key_available(idx):
                continue
            any_available = True
            if attempt_count > 0:
                _time.sleep(_INTRA_SWEEP_GAP_S)
            attempt_count += 1
            status, result, retry_after = _embed_v2_rest_batch(
                chunk, "RETRIEVAL_DOCUMENT", dim, keys[idx]
            )
            # Advance RR pointer per attempt, win or lose — keeps load
            # spread even when keys are unevenly available.
            _key_rr_idx = (idx + 1) % len(keys)
            if status == 200 and result is not None:
                return result
            if status in (400, 401, 403, 429) or status == -1:
                cool_for = (
                    retry_after if retry_after is not None else _DEFAULT_COOLDOWN_S
                )
                _cool_key(idx, cool_for)
                log.warning(
                    "v2 batch: key #%d status=%s retry_after=%s (cooling %.1fs)",
                    idx,
                    status,
                    retry_after,
                    cool_for,
                )
                continue
            # Non-retriable per-key (e.g. 5xx). Note and abandon — caller
            # can split-half.
            log.warning("v2 batch: key #%d non-retriable status=%s", idx, status)
            last_nonretriable_status = status
            return None

        if not any_available:
            wait = _soonest_wake()
            if wait <= 0:
                # Race: cooldown expired between checks. Try again.
                continue
            wait = min(wait, 90.0)
            log.warning("v2 batch: all keys cooling, sleeping %.1fs to wake-up", wait)
            _time.sleep(wait)
            # After waking up, loop back to try again.

    log.warning(
        "v2 batch: chunk deadline reached (last non-retriable status=%s)",
        last_nonretriable_status,
    )
    return None


def compute_document_embeddings_v2_batch(
    texts: list[str],
    dim: int = _EMBEDDING_DIM_V2,
    token_budget: int = 8000,
    max_count: int = 30,
    inter_chunk_sleep: float = 6.0,
) -> list[list[float] | None]:
    """Bulk-embed documents via :batchEmbedContents REST (1 HTTP call → N vectors).

    Chunks are sized by **token budget**, not document count — wiki docs
    vary 200..2000 chars and a flat count-based limit blows past the 30K
    TPM cap on long-doc-heavy batches. Token budget 8000 leaves room for
    transient bursts and stays well under TPM.

    Pacing: inter_chunk_sleep seconds between chunks keeps total
    throughput well under 100 RPM per key even on a single-key pool.

    Resilience: if a chunk fails its full sweep budget, retried as two
    halves recursively. A single bad document gets isolated and skipped,
    rest of the batch continues.

    Returns embeddings in input order; None for any text that could not
    be embedded (typically the bad-doc case after split-half).
    """
    if not texts:
        return []

    keys = _get_embedding_api_keys()
    if not keys:
        log.warning("v2 batch: no API keys (GEMINI_API_KEY/GEMINI_API_KEYS missing)")
        return [None] * len(texts)

    import time as _time

    chunks = _split_by_token_budget(texts, token_budget, max_count)
    log.info(
        "v2 batch: %d texts → %d chunks (budget=%d tokens, max=%d count)",
        len(texts),
        len(chunks),
        token_budget,
        max_count,
    )

    out: list[list[float] | None] = []
    for chunk_idx, chunk in enumerate(chunks):
        result = _process_one_chunk(chunk, dim, keys)

        if result is None and len(chunk) > 1:
            # Split-half rescue: maybe one doc poisons the batch.
            log.warning(
                "v2 batch: chunk %d/%d (%d texts) failed, splitting in half",
                chunk_idx + 1,
                len(chunks),
                len(chunk),
            )
            mid = len(chunk) // 2
            left = compute_document_embeddings_v2_batch(
                chunk[:mid], dim, token_budget, max_count, inter_chunk_sleep
            )
            right = compute_document_embeddings_v2_batch(
                chunk[mid:], dim, token_budget, max_count, inter_chunk_sleep
            )
            result = left + right

        if result is None:
            result = [None] * len(chunk)
        out.extend(result)

        if chunk_idx < len(chunks) - 1 and inter_chunk_sleep > 0:
            _time.sleep(inter_chunk_sleep)

    return out


def _fetch_by_vector(
    db,
    collection: str,
    query_vector: list[float],
    limit: int,
    source_label: str,
) -> list[dict]:
    """Vector search via Firestore find_nearest. Falls back to recency on error."""
    try:
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector

        docs = (
            db.collection(collection)
            .find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=limit,
            )
            .stream()
        )
        results = []
        for d in docs:
            data = d.to_dict()
            if not data:
                continue
            data["_source"] = source_label
            results.append(data)
        return results
    except Exception as e:
        log.warning(
            "МОЗГ: vector search on %s failed (%s) — falling back to recency",
            collection,
            e,
        )
        return []


def _fetch_wisdom_and_wiki(
    query: str = "",
    wisdom_limit: int = 150,
    wiki_limit: int = 50,
) -> list[dict]:
    """
    Fetch top-N docs from BOTH wisdom_base (chat-mined facts) and
    wikivoyage_base (wiki-imported facts).

    When `query` is provided (non-empty) AND an embedding can be computed,
    uses vector search (find_nearest COSINE) — returns semantically relevant
    docs regardless of insertion time.

    Falls back to recency sort (order_by createdAt/importedAt) when:
    - query is empty string
    - GEMINI_API_KEY is missing
    - embedding API call fails
    - Firestore vector index not yet deployed

    Each doc is tagged with `_source` ("База данных" or "WikiVoyage") so
    the formatter can mark its origin in the context Gemini sees.

    Token math: ~150 wisdom + ~50 wiki = ~200 docs × ~200 chars ≈ 40K chars
    ≈ ~12K tokens. Fits Live API 65K TPM budget with headroom.
    """
    from google.cloud import firestore as fs

    cfg = get_config()
    if not cfg.google.firestore_project_id:
        log.warning("Gemini МОЗГ: firestore_project_id not configured — skipping.")
        return []

    try:
        db = _build_firestore_client()
    except Exception as e:
        log.error("Gemini МОЗГ: Firestore init failed: %s", e)
        return []

    # Try to compute embedding for vector search.
    # Must use the v2 query embedder (1536 dim, task_type=RETRIEVAL_QUERY) —
    # the Firestore vector indexes are built for 1536-dim vectors, and the
    # legacy _compute_embedding returned 768-dim symmetric vectors. Calling
    # the legacy path against the new index threw silent dim-mismatch
    # exceptions that fell back to recency sort — MOZG looked like it
    # "worked" but was returning the newest 200 docs regardless of relevance.
    query_vector: list[float] | None = None
    if query:
        query_vector = compute_query_embedding_v2(query)
        if query_vector:
            log.info("МОЗГ: using vector search for query %r", query[:60])
        else:
            log.info("МОЗГ: embedding failed — falling back to recency sort")

    results: list[dict] = []

    # 1. wisdom_base — chat-mined facts
    if query_vector:
        results.extend(
            _fetch_by_vector(
                db,
                cfg.google.assistant_collection,
                query_vector,
                wisdom_limit,
                "База данных",
            )
        )
        # If vector search returned nothing (e.g. index not ready), fall back
        if not results:
            log.info("МОЗГ: vector search returned 0 wisdom docs — recency fallback")
            query_vector = None  # triggers recency path below for both collections

    if not query_vector:
        try:
            docs = (
                db.collection(cfg.google.assistant_collection)
                .order_by("createdAt", direction=fs.Query.DESCENDING)
                .limit(wisdom_limit)
                .stream()
            )
            for d in docs:
                data = d.to_dict()
                if not data:
                    continue
                data["_source"] = "База данных"
                results.append(data)
        except Exception as e:
            log.error("Gemini МОЗГ: wisdom_base query failed: %s", e)

    # 2. wikivoyage_base — wiki-imported facts
    # Optional: if collection name is configurable add to GoogleConfig later.
    if query_vector:
        wiki_results = _fetch_by_vector(
            db,
            "wikivoyage_base",
            query_vector,
            wiki_limit,
            "WikiVoyage",
        )
        results.extend(wiki_results)
    else:
        try:
            docs = (
                db.collection("wikivoyage_base")
                .order_by("importedAt", direction=fs.Query.DESCENDING)
                .limit(wiki_limit)
                .stream()
            )
            for d in docs:
                data = d.to_dict()
                if not data:
                    continue
                data["_source"] = "WikiVoyage"
                results.append(data)
        except Exception as e:
            # Wiki collection may not exist yet / index missing. Soft fail —
            # wisdom_base alone still works.
            log.warning("Gemini МОЗГ: wikivoyage_base query skipped (%s)", e)

    return results


# Back-compat alias for callers that still expect the old name.
def _fetch_wisdom(limit: int = 200) -> list[dict]:
    return _fetch_wisdom_and_wiki(query="", wisdom_limit=limit, wiki_limit=0)


def _format_context(docs: list[dict]) -> str:
    """Format docs as numbered context for Gemini.

    Header shape: [n. title | tag | country | source]
    Source is "База данных" or "WikiVoyage" — explicit so the model can
    attribute citations and weigh recency vs encyclopedic baseline.
    """
    parts = []
    idx = 1
    for doc in docs:
        instruction = (doc.get("instruction") or "").strip()
        if not instruction:
            continue
        title = (doc.get("title") or "").strip()
        tag = (doc.get("tag") or "").strip()
        country = (doc.get("country") or "").strip()
        source = (doc.get("_source") or "База данных").strip()
        header = f"[{idx}. {title} | {tag} | {country} | {source}]"
        parts.append(f"{header}\n{instruction}")
        idx += 1
    return "\n\n".join(parts)


async def _ask_live_api(
    prompt: str,
    model_name: str,
    api_key: str,
    history: list[dict] | None = None,
) -> str:
    """
    Single-shot Q&A via Gemini Live API.

    Opens a session, sends the prompt as a complete turn, receives the model's
    response chunks, returns the concatenated text. Closes the session.

    Why bother with Live for a single-shot exchange:
    - Live API's free-tier quota is "Unlimited RPD" (vs 500/day on
      flash-lite-preview shared between МОЗГ and Apps Script extraction).
      Each МОЗГ question = one Live session, doesn't burn the per-day cap.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # Format mirrored from a known-working production setup (Node.js
    # @google/genai voice translator) for the same model. Key points:
    #
    # - gemini-3.1-flash-live-preview is an AUDIO-output model. Asking for
    #   [Modality.TEXT] gets the WebSocket closed with 1011 at setup.
    #   We request AUDIO and read the spoken text via
    #   output_audio_transcription — Gemini emits the transcription
    #   alongside the audio bytes, and that's what we keep (audio bytes
    #   are discarded; МОЗГ posts to Telegram as text).
    #
    # - thinking_level="minimal" is mandatory for 3.1 Live to avoid
    #   long server-side "thinking" that times out the WS.
    #
    # - system_instruction can be a plain string here (SDK wraps it).
    #
    # - send_client_content with turns + turn_complete=True is the
    #   correct way to push a single user turn for batch Q&A.
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),  # type: ignore[arg-type]
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=_BRAIN_SYSTEM,
    )

    # Build multi-turn payload: prior history (if any) + the new user turn.
    # history is a list of dicts {role: "user"|"model", text: "..."} —
    # passed in from telegram_client when this is a reply-continuation.
    turns: list[types.Content] = []
    if history:
        for h in history:
            role = h.get("role")
            text = (h.get("text") or "").strip()
            if not text or role not in ("user", "model"):
                continue
            turns.append(types.Content(role=role, parts=[types.Part(text=text)]))
    turns.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    chunks: list[str] = []
    async with client.aio.live.connect(model=model_name, config=config) as session:
        await session.send_client_content(
            turns=turns,  # type: ignore[arg-type]
            turn_complete=True,
        )
        async for response in session.receive():
            sc = response.server_content
            if not sc:
                continue
            # The audio-mode transcription IS our answer. The actual audio
            # bytes in sc.model_turn.parts[*].inline_data are ignored.
            ot = getattr(sc, "output_transcription", None)
            if ot and ot.text:
                chunks.append(ot.text)
            if sc.turn_complete:
                break

    return "".join(chunks).strip()


async def _ask_sync_fallback(
    prompt: str,
    model_name: str,
    api_key: str,
    history: list[dict] | None = None,
) -> str:
    """
    Fallback to non-streaming Gemini request when Live API errors.
    Same `google-genai` SDK, just `generate_content` instead of a Live session.
    Runs the blocking call in a thread to keep the bot's asyncio loop unblocked.

    (We deliberately do NOT import the legacy google-generativeai SDK —
    https://github.com/google-gemini/deprecated-generative-ai-python — Google
    has deprecated it; new SDK `google-genai` covers both paths.)
    """
    from google import genai
    from google.genai import types

    contents: list[types.Content] = []
    if history:
        for h in history:
            role = h.get("role")
            text = (h.get("text") or "").strip()
            if not text or role not in ("user", "model"):
                continue
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    def _blocking() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                system_instruction=_BRAIN_SYSTEM,
            ),
        )
        return (response.text or "").strip()

    return await asyncio.to_thread(_blocking)


async def search_and_format(
    country: str,
    query: str,
    history: list[dict] | None = None,
) -> str:
    """
    Query Firestore (wisdom_base + wikivoyage_base) and synthesize answer
    via Gemini Live API. Returns "" on failure so the caller can fall back
    to DeepSeek+SQLite.

    Args:
        country: ignored — Gemini sees broad context and picks relevance
            from query text. Kept in signature for caller compat.
        query: current user turn.
        history: optional list of {"role": "user"|"model", "text": "..."}
            entries representing prior exchanges. When the user replies
            to a previous bot answer in Telegram, telegram_client passes
            in [{role:user, text:original_q}, {role:model, text:prev_a}]
            so Gemini understands "Какое есть такси" as a follow-up.
    """
    cfg = get_config()

    docs = _fetch_wisdom_and_wiki(query=query)
    if not docs:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате — "
            "кто-нибудь точно подскажет!"
        )

    context = _format_context(docs)
    useful_count = context.count("\n\n") + 1 if context else 0
    log.info(
        "Gemini МОЗГ: %d docs fetched (%d useful), history=%d turns, query=%r",
        len(docs),
        useful_count,
        len(history or []),
        query[:60],
    )

    if not context:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате!"
        )

    prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"База знаний ({useful_count} записей):\n\n{context}"
    )

    # Primary path: Live API (Unlimited RPD on free tier).
    answer = ""
    if cfg.gemini.live_model:
        try:
            answer = await _ask_live_api(
                prompt,
                cfg.gemini.live_model,
                cfg.gemini.api_key,
                history=history,
            )
        except Exception as e:
            log.warning(
                "Gemini Live API failed (%s) — falling back to sync %s",
                e,
                cfg.gemini.model,
            )

    # Fallback: legacy synchronous Gemini (shares 500 RPD cap).
    if not answer:
        try:
            answer = await _ask_sync_fallback(
                prompt,
                cfg.gemini.model,
                cfg.gemini.api_key,
                history=history,
            )
        except Exception as e:
            log.error("Gemini sync fallback also failed: %s", e)
            return ""  # Caller now falls back to DeepSeek

    if not answer:
        return ""

    return f"🧠 {answer}\n\n<i>На основе {useful_count} записей из базы знаний</i>"


def is_enabled() -> bool:
    """True if Gemini МОЗГ is configured (GEMINI_API_KEY present)."""
    try:
        return bool(get_config().gemini.api_key)
    except Exception:
        return False
