"""
gemini_brain.py — МОЗГ via Gemini Live API + Firestore wisdom_base.

Flow:
1. Query Firestore wisdom_base: pull the top-N most recent docs across
   ALL countries.
2. Format context with source attribution (WikiVoyage vs База данных).
3. Open Gemini Live API session (or sync fallback).
4. Return answer in Russian.
"""

from __future__ import annotations

import asyncio
import os
import time as _time

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
_EMBEDDING_MODEL_V2 = "gemini-embedding-2"
_EMBEDDING_DIM_V2 = 1536
_EMBEDDING_MODEL_TAG_V2 = "gemini-embedding-2-1536"


def _get_embedding_api_key() -> str:
    """Legacy single key path."""
    cfg = get_config()
    return str(
        getattr(cfg.gemini, "api_key", None) or os.environ.get("GEMINI_API_KEY", "")
    )


def _get_embedding_api_keys() -> list[str]:
    """Return all available Gemini API keys for embedding (multi-key pool)."""
    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    if raw_multi:
        keys = [k.strip() for k in raw_multi.split(",") if k.strip()]
        if keys:
            return keys
    single = _get_embedding_api_key()
    return [single] if single else []


# Round-robin pointer for the multi-key pool.
_key_rr_idx = 0


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
    """Single :batchEmbedContents REST call carrying N texts."""
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
        resp = _req.post(url, json=body, timeout=timeout)
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
    while len(out) < len(texts):
        out.append(None)
    return 200, out[: len(texts)], None


def compute_document_embeddings_v2_batch(
    texts: list[str],
    dim: int = _EMBEDDING_DIM_V2,
    chunk_size: int = 10,
) -> list[list[float] | None]:
    """Bulk-embed documents via REST with key rotation and 429 protection."""
    if not texts:
        return []

    keys = _get_embedding_api_keys()
    if not keys:
        log.warning("v2 batch: no API keys available")
        return [None] * len(texts)

    global _key_rr_idx
    out: list[list[float] | None] = []
    
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        chunk_result: list[list[float] | None] | None = None
        
        for sweep in range(3):
            min_retry_after: float | None = None
            for offset in range(len(keys)):
                idx = (_key_rr_idx + offset) % len(keys)
                status, result, retry_after = _embed_v2_rest_batch(
                    chunk, "RETRIEVAL_DOCUMENT", dim, keys[idx]
                )
                
                # РОТАЦИЯ: всегда переключаем ключ
                _key_rr_idx = (idx + 1) % len(keys)
                
                if status == 200 and result is not None:
                    chunk_result = result
                    break
                if status == 429:
                    log.warning("v2 batch: key #%d throttled. Wait: %s", idx, retry_after)
                    if retry_after:
                        min_retry_after = retry_after if min_retry_after is None else min(min_retry_after, retry_after)
                    continue
            
            if chunk_result is not None:
                break
            
            wait = (min_retry_after + 1.0) if min_retry_after else 30.0
            log.warning("v2 batch: all keys cooling down, sleeping %.1fs", wait)
            _time.sleep(wait)
            
        if chunk_result is None:
            chunk_result = [None] * len(chunk)
            
        out.extend(chunk_result)
        
        # ПРОФИЛАКТИЧЕСКИЙ СОН
        if start + chunk_size < len(texts):
            _time.sleep(3.5 if len(keys) < 2 else 1.0)

    return out


def compute_query_embedding_v2(
    text: str, dim: int = _EMBEDDING_DIM_V2
) -> list[float] | None:
    """Embed a single user query for retrieval (task_type=RETRIEVAL_QUERY)."""
    if not text or not text.strip():
        return None
    keys = _get_embedding_api_keys()
    if not keys:
        return None
        
    from google.genai import types as genai_types
    from google import genai
    
    client = genai.Client(api_key=keys[0])
    cfg = genai_types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=dim,
    )
    try:
        result = client.models.embed_content(
            model=_EMBEDDING_MODEL_V2,
            contents=text,
            config=cfg,
        )
        embeddings = result.embeddings or []
        return list(embeddings[0].values or []) if embeddings else None
    except Exception as e:
        log.warning("v2 query embed failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Retrieval & Brain Logic
# ---------------------------------------------------------------------------

def _fetch_wisdom_and_wiki(
    query: str = "",
    wisdom_limit: int = 150,
    wiki_limit: int = 50,
) -> list[dict]:
    """Fetch top-N docs from wisdom_base and wikivoyage_base."""
    from google.cloud import firestore as fs
    cfg = get_config()
    if not cfg.google.firestore_project_id:
        return []

    try:
        db = _build_firestore_client()
    except Exception as e:
        log.error("Firestore init failed: %s", e)
        return []

    query_vector = compute_query_embedding_v2(query)
    results: list[dict] = []

    # 1. wisdom_base (chat facts)
    # 2. wikivoyage_base (wiki facts)
    collections = [
        (cfg.google.assistant_collection, wisdom_limit, "База данных", "createdAt"),
        ("wikivoyage_base", wiki_limit, "WikiVoyage", "importedAt"),
    ]

    for coll_name, limit, label, time_field in collections:
        try:
            if query_vector:
                from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
                from google.cloud.firestore_v1.vector import Vector
                docs = db.collection(coll_name).find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit,
                ).stream()
            else:
                docs = db.collection(coll_name).order_by(time_field, direction=fs.Query.DESCENDING).limit(limit).stream()
            
            for d in docs:
                data = d.to_dict()
                if data:
                    data["_source"] = label
                    results.append(data)
        except Exception as e:
            log.warning("Fetch from %s failed: %s", coll_name, e)

    return results


def _format_context(docs: list[dict]) -> str:
    """Format docs as numbered context for Gemini."""
    parts = []
    for idx, doc in enumerate(docs, 1):
        instruction = (doc.get("instruction") or "").strip()
        if not instruction:
            continue
        header = f"[{idx}. {doc.get('title')} | {doc.get('tag')} | {doc.get('country')} | {doc.get('_source')}]"
        parts.append(f"{header}\n{instruction}")
    return "\n\n".join(parts)


async def _ask_live_api(
    prompt: str,
    model_name: str,
    api_key: str,
    history: list[dict] | None = None,
) -> str:
    """Single-shot Q&A via Gemini Live API (Audio transcription mode)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=_BRAIN_SYSTEM,
    )

    turns: list[types.Content] = []
    if history:
        for h in history:
            turns.append(types.Content(role=h['role'], parts=[types.Part(text=h['text'])]))
    turns.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    chunks: list[str] = []
    async with client.aio.live.connect(model=model_name, config=config) as session:
        await session.send_client_content(turns=turns, turn_complete=True)
        async for response in session.receive():
            sc = response.server_content
            if sc and sc.output_transcription:
                chunks.append(sc.output_transcription.text)
            if sc and sc.turn_complete:
                break
    return "".join(chunks).strip()


async def _ask_sync_fallback(
    prompt: str,
    model_name: str,
    api_key: str,
    history: list[dict] | None = None,
) -> str:
    """Fallback to non-streaming Gemini request."""
    from google import genai
    from google.genai import types

    contents: list[types.Content] = []
    if history:
        for h in history:
            contents.append(types.Content(role=h['role'], parts=[types.Part(text=h['text'])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    def _blocking() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=_BRAIN_SYSTEM),
        )
        return (response.text or "").strip()

    return await asyncio.to_thread(_blocking)


async def search_and_format(
    country: str,
    query: str,
    history: list[dict] | None = None,
) -> str:
    """Entry point for МОЗГ Q&A."""
    docs = _fetch_wisdom_and_wiki(query=query)
    if not docs:
        return "🧠 Не нашёл ничего по этому запросу. Попробуй переформулировать!"

    context = _format_context(docs)
    prompt = f"Вопрос пользователя: {query}\n\nБаза знаний:\n\n{context}"
    cfg = get_config()

    answer = ""
    try:
        answer = await _ask_live_api(prompt, cfg.gemini.live_model, cfg.gemini.api_key, history)
    except Exception as e:
        log.warning("Live API failed (%s), falling back to sync", e)

    if not answer:
        try:
            answer = await _ask_sync_fallback(prompt, cfg.gemini.model, cfg.gemini.api_key, history)
        except Exception as e:
            log.error("Sync fallback failed: %s", e)
            return ""

    return f"🧠 {answer}\n\n<i>На основе {len(docs)} записей из базы знаний</i>"


def is_enabled() -> bool:
    """True if Gemini МОЗГ is configured."""
    try:
        return bool(get_config().gemini.api_key)
    except Exception:
        return False
