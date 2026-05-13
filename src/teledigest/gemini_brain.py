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


def _fetch_wisdom_and_wiki(wisdom_limit: int = 150, wiki_limit: int = 50) -> list[dict]:
    """
    Fetch top-N most recent docs from BOTH wisdom_base (chat-mined facts)
    and wikivoyage_base (wiki-imported facts). Each doc is tagged with a
    `_source` field — "База данных" for wisdom, "WikiVoyage" for wiki — so
    the formatter can mark its origin in the context Gemini sees.

    No country filter on either query. Gemini picks relevance from the
    [country/tag/source] headers.

    Token math: ~200 wisdom + ~50 wiki = ~250 docs × ~200 chars = ~50K chars
    ≈ ~15K tokens of context. Fits Live API 65K TPM budget with headroom.
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

    results: list[dict] = []

    # 1. wisdom_base — chat-mined facts, sorted by createdAt
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

    # 2. wikivoyage_base — wiki-imported facts, sorted by importedAt
    # Optional: if collection name is configurable add to GoogleConfig later.
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
    return _fetch_wisdom_and_wiki(wisdom_limit=limit, wiki_limit=0)


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
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
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
            turns=turns,
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
            contents=contents,
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

    docs = _fetch_wisdom_and_wiki()
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
        len(docs), useful_count, len(history or []), query[:60],
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
                prompt, cfg.gemini.live_model, cfg.gemini.api_key, history=history,
            )
        except Exception as e:
            log.warning(
                "Gemini Live API failed (%s) — falling back to sync %s",
                e, cfg.gemini.model,
            )

    # Fallback: legacy synchronous Gemini (shares 500 RPD cap).
    if not answer:
        try:
            answer = await _ask_sync_fallback(
                prompt, cfg.gemini.model, cfg.gemini.api_key, history=history,
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
