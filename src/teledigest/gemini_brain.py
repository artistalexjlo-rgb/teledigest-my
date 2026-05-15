"""
gemini_brain.py — МОЗГ via Gemini Live API + Firestore wisdom_base.
(Минимальные правки только для фикса 429 и ротации ключей)
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

_EMBEDDING_MODEL_V2 = "gemini-embedding-2"
_EMBEDDING_DIM_V2 = 1536
_EMBEDDING_MODEL_TAG_V2 = "gemini-embedding-2-1536"


def _get_embedding_api_key() -> str:
    """Return Gemini API key from config or env."""
    cfg = get_config()
    return str(
        getattr(cfg.gemini, "api_key", None) or os.environ.get("GEMINI_API_KEY", "")
    )


def _get_embedding_api_keys() -> list[str]:
    """Return all available Gemini API keys for embedding."""
    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    if raw_multi:
        keys = [k.strip() for k in raw_multi.split(",") if k.strip()]
        if keys:
            return keys
    single = _get_embedding_api_key()
    return [single] if single else []


# Round-robin pointer for the multi-key pool.
_key_rr_idx = 0


def _compute_embedding(text: str, model_idx: int = 0) -> list[float] | None:
    from google import genai
    api_key = _get_embedding_api_key()
    if not api_key:
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
                continue
            return list(embeddings[0].values or [])
        except Exception as e:
            if "429" in str(e):
                continue
            return None
    return None


def _parse_retry_after_seconds(body_text: str) -> float | None:
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
    import requests as _req
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"
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
    except Exception:
        return -1, None, None
    if resp.status_code != 200:
        retry_after = _parse_retry_after_seconds(resp.text)
        return resp.status_code, None, retry_after
    data = resp.json()
    embeds = data.get("embeddings", [])
    out = [list(emb.get("values") or []) for emb in embeds]
    while len(out) < len(texts):
        out.append(None)
    return 200, out[: len(texts)], None


def compute_document_embeddings_v2_batch(
    texts: list[str],
    dim: int = _EMBEDDING_DIM_V2,
    chunk_size: int = 20, # Снижено для WikiVoyage
) -> list[list[float] | None]:
    """Bulk-embed documents with surgical fixes for 429 and rotation."""
    if not texts:
        return []

    keys = _get_embedding_api_keys()
    if not keys:
        return [None] * len(texts)

    global _key_rr_idx
    out: list[list[float] | None] = []
    
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        chunk_result = None
        
        for sweep in range(3):
            min_retry_after = None
            for offset in range(len(keys)):
                idx = (_key_rr_idx + offset) % len(keys)
                status, result, retry_after = _embed_v2_rest_batch(
                    chunk, "RETRIEVAL_DOCUMENT", dim, keys[idx]
                )
                
                # ВСЕГДА двигаем индекс, чтобы следующий запрос шел с нового ключа
                _key_rr_idx = (idx + 1) % len(keys)
                
                if status == 200 and result is not None:
                    chunk_result = result
                    break
                if status == 429 and retry_after:
                    min_retry_after = retry_after if min_retry_after is None else min(min_retry_after, retry_after)
            
            if chunk_result is not None:
                break
            
            wait = (min_retry_after + 1.0) if min_retry_after else 30.0
            log.warning("TPM Limit! Sleeping %.1fs", wait)
            _time.sleep(wait)
            
        if chunk_result is None:
            chunk_result = [None] * len(chunk)
            
        out.extend(chunk_result)
        
        # Микро-пауза после успеха, чтобы не выжигать TPM
        if start + chunk_size < len(texts):
            _time.sleep(1.5)

    return out


# ---------------------------------------------------------------------------
# Остальная часть файла БЕЗ ИЗМЕНЕНИЙ (как в оригинале)
# ---------------------------------------------------------------------------

def _fetch_wisdom_and_wiki(query: str = "", wisdom_limit: int = 150, wiki_limit: int = 50) -> list[dict]:
    from google.cloud import firestore as fs
    cfg = get_config()
    db = _build_firestore_client()
    query_vector = _compute_embedding(query) if query else None
    results: list[dict] = []

    # Wisdom
    try:
        coll = db.collection(cfg.google.assistant_collection)
        docs = coll.order_by("createdAt", direction=fs.Query.DESCENDING).limit(wisdom_limit).stream()
        for d in docs:
            data = d.to_dict()
            if data:
                data["_source"] = "База данных"
                results.append(data)
    except Exception as e: log.error("Wisdom query failed: %s", e)

    # Wiki
    try:
        coll = db.collection("wikivoyage_base")
        docs = coll.order_by("importedAt", direction=fs.Query.DESCENDING).limit(wiki_limit).stream()
        for d in docs:
            data = d.to_dict()
            if data:
                data["_source"] = "WikiVoyage"
                results.append(data)
    except Exception as e: log.warning("Wiki query skipped: %s", e)
    
    return results

def _format_context(docs: list[dict]) -> str:
    parts = []
    for idx, doc in enumerate(docs, 1):
        instruction = (doc.get("instruction") or "").strip()
        if not instruction: continue
        title = (doc.get("title") or "").strip()
        tag = (doc.get("tag") or "").strip()
        country = (doc.get("country") or "").strip()
        source = doc.get("_source", "База данных")
        header = f"[{idx}. {title} | {tag} | {country} | {source}]"
        parts.append(f"{header}\n{instruction}")
    return "\n\n".join(parts)

async def _ask_live_api(prompt: str, model_name: str, api_key: str, history: list[dict] | None = None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=_BRAIN_SYSTEM,
    )
    turns = []
    if history:
        for h in history:
            turns.append(types.Content(role=h.get("role"), parts=[types.Part(text=h.get("text"))]))
    turns.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    chunks = []
    async with client.aio.live.connect(model=model_name, config=config) as session:
        await session.send_client_content(turns=turns, turn_complete=True)
        async for response in session.receive():
            ot = getattr(response.server_content, "output_transcription", None)
            if ot and ot.text: chunks.append(ot.text)
            if response.server_content and response.server_content.turn_complete: break
    return "".join(chunks).strip()

async def _ask_sync_fallback(prompt: str, model_name: str, api_key: str, history: list[dict] | None = None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    contents = []
    if history:
        for h in history:
            contents.append(types.Content(role=h.get("role"), parts=[types.Part(text=h.get("text"))]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    def _blocking():
        resp = client.models.generate_content(model=model_name, contents=contents, config=types.GenerateContentConfig(system_instruction=_BRAIN_SYSTEM))
        return (resp.text or "").strip()
    return await asyncio.to_thread(_blocking)

async def search_and_format(country: str, query: str, history: list[dict] | None = None) -> str:
    docs = _fetch_wisdom_and_wiki(query)
    if not docs: return "🧠 Не нашёл ничего в базе."
    context = _format_context(docs)
    useful_count = context.count("\n\n") + 1 if context else 0
    prompt = f"Вопрос: {query}\n\nБаза знаний ({useful_count} записей):\n\n{context}"
    cfg = get_config()
    answer = ""
    try:
        answer = await _ask_live_api(prompt, cfg.gemini.live_model, cfg.gemini.api_key, history)
    except Exception: pass
    if not answer:
        try:
            answer = await _ask_sync_fallback(prompt, cfg.gemini.model, cfg.gemini.api_key, history)
        except Exception: return ""
    return f"🧠 {answer}\n\n<i>На основе {useful_count} записей</i>"

def is_enabled() -> bool:
    try: return bool(get_config().gemini.api_key)
    except Exception: return False
