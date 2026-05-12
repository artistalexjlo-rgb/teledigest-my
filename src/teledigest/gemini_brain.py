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

The knowledge base mixes facts from many countries. Each fact has a
[country/tag] header so you can tell where it applies.

Behavior:
- Answer in Russian, conversational and informal.
- Be specific: include exact prices, addresses, timelines, service names, steps.
- KEEP IT SHORT: 2-4 sentences MAX. The user is on a phone — no walls of text.
- If the user needs depth, finish with: "хочешь подробнее — спроси конкретнее".
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
    """
    Build a Firestore client using the OAuth token shared with channel_poster.
    Mirrors channel_poster._build_firestore_client() — same token, same scope.
    """
    from google.oauth2.credentials import Credentials
    from google.cloud import firestore

    cfg = get_config()
    if not cfg.google.token_path.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {cfg.google.token_path}. "
            "Run scripts/drive_oauth_init.py with datastore scope."
        )
    if not cfg.google.firestore_project_id:
        raise RuntimeError("[google] firestore_project_id is not set in config.")

    creds = Credentials.from_authorized_user_file(
        str(cfg.google.token_path),
        scopes=["https://www.googleapis.com/auth/datastore"],
    )
    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
        cfg.google.token_path.write_text(creds.to_json(), encoding="utf-8")

    return firestore.Client(
        project=cfg.google.firestore_project_id,
        database=cfg.google.firestore_database,
        credentials=creds,
    )


def _fetch_wisdom(limit: int = 200) -> list[dict]:
    """
    Fetch the most recent `limit` wisdom_base docs across ALL countries.
    No country filter — Gemini reads the headers and picks relevant facts.

    Why 200: at ~200 chars per instruction, that's ~40K chars / ~12K tokens
    of context. Well under Live API's 65K TPM budget per request, with
    headroom for the actual answer. Bumping further hits diminishing
    returns (older facts more likely stale, and Gemini's attention spreads
    thinner over too much context).
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

    collection = cfg.google.assistant_collection
    try:
        docs = (
            db.collection(collection)
            .order_by("createdAt", direction=fs.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs if d.to_dict()]
    except Exception as e:
        log.error("Gemini МОЗГ: Firestore query failed: %s", e)
        return []


def _format_context(docs: list[dict]) -> str:
    """Format wisdom_base instruction fields as numbered context for Gemini."""
    parts = []
    idx = 1
    for doc in docs:
        instruction = (doc.get("instruction") or "").strip()
        if not instruction:
            continue
        title = (doc.get("title") or "").strip()
        tag = (doc.get("tag") or "").strip()
        country = (doc.get("country") or "").strip()
        header = f"[{idx}. {title} | {tag} | {country}]"
        parts.append(f"{header}\n{instruction}")
        idx += 1
    return "\n\n".join(parts)


async def _ask_live_api(prompt: str, model_name: str, api_key: str) -> str:
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
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=_BRAIN_SYSTEM,
    )

    chunks: list[str] = []
    async with client.aio.live.connect(model=model_name, config=config) as session:
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part(text=prompt)])],
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


async def _ask_sync_fallback(prompt: str, model_name: str, api_key: str) -> str:
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

    def _blocking() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_BRAIN_SYSTEM,
            ),
        )
        return (response.text or "").strip()

    return await asyncio.to_thread(_blocking)


async def search_and_format(country: str, query: str) -> str:
    """
    Query Firestore wisdom_base and synthesize answer via Gemini Live API.
    Returns empty string on failure (caller should fallback to DeepSeek).

    The `country` argument is currently ignored — Gemini sees broad context
    and infers the country from the question itself. Kept in the signature
    so the public API for knowledge_search.search_and_format doesn't break.
    """
    cfg = get_config()

    docs = _fetch_wisdom()
    if not docs:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате — "
            "кто-нибудь точно подскажет!"
        )

    context = _format_context(docs)
    useful_count = context.count("\n\n") + 1 if context else 0
    log.info(
        "Gemini МОЗГ: %d docs fetched (%d with instructions), query=%r",
        len(docs), useful_count, query[:60],
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
            answer = await _ask_live_api(prompt, cfg.gemini.live_model, cfg.gemini.api_key)
        except Exception as e:
            log.warning(
                "Gemini Live API failed (%s) — falling back to sync %s",
                e, cfg.gemini.model,
            )

    # Fallback: legacy synchronous Gemini (shares 500 RPD cap).
    if not answer:
        try:
            answer = await _ask_sync_fallback(prompt, cfg.gemini.model, cfg.gemini.api_key)
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
