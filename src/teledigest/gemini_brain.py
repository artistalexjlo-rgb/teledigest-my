"""
gemini_brain.py — МОЗГ via Gemini + Firestore wisdom_base.

Flow:
1. Query Firestore wisdom_base: country-specific docs + universal "any" docs
   (country-specific first, then universal "any" appended)
2. Format instruction fields as context (English facts)
3. Gemini synthesizes a concrete Russian answer
4. Return to user

Auth: same OAuth token.json as channel_poster (datastore scope).
Config: GEMINI_API_KEY env var (or [gemini] api_key in config).
"""

from __future__ import annotations

from .config import get_config, log

_BRAIN_SYSTEM = """\
You are an assistant bot for an expat community chat called "МОЗГ".
Answer the user's question using ONLY the facts provided in the knowledge base below.

Rules:
- Answer in Russian, conversational and informal
- Be specific: include exact prices, addresses, timelines, service names, steps
- 3-7 sentences maximum
- If there are conflicting opinions — mention both
- If the knowledge base has nothing useful — reply exactly: "в базе пока нет информации по этому вопросу"
- Plain text only — no Markdown, no bullet symbols, no formatting
- Do NOT invent or guess information
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


def _fetch_wisdom(country: str, limit_country: int = 30, limit_any: int = 20) -> list[dict]:
    """
    Fetch wisdom_base docs for the given country + universal "any" entries.
    Two separate Firestore queries (Firestore can't OR on different field values easily).
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
    results: list[dict] = []

    # Country-specific knowledge
    try:
        docs = (
            db.collection(collection)
            .where("country", "==", country.lower())
            .order_by("createdAt", direction=fs.Query.DESCENDING)
            .limit(limit_country)
            .stream()
        )
        results.extend(d.to_dict() for d in docs if d.to_dict())
    except Exception as e:
        log.error("Gemini МОЗГ: Firestore query failed (country=%s): %s", country, e)

    # Universal "any" knowledge (travel hacks, cross-country advice)
    try:
        docs = (
            db.collection(collection)
            .where("country", "==", "any")
            .order_by("createdAt", direction=fs.Query.DESCENDING)
            .limit(limit_any)
            .stream()
        )
        results.extend(d.to_dict() for d in docs if d.to_dict())
    except Exception as e:
        log.error("Gemini МОЗГ: Firestore query failed (any): %s", e)

    return results


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


def search_and_format(country: str, query: str) -> str:
    """
    Query Firestore wisdom_base and synthesize answer via Gemini.
    Returns empty string on failure (caller should fallback to DeepSeek).
    """
    cfg = get_config()

    docs = _fetch_wisdom(country)
    if not docs:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате — "
            "кто-нибудь точно подскажет!"
        )

    context = _format_context(docs)
    useful_count = context.count("\n\n") + 1 if context else 0
    log.info(
        "Gemini МОЗГ: %d docs fetched (%d with instructions) for country=%s, query=%r",
        len(docs), useful_count, country, query[:60],
    )

    if not context:
        return (
            "🧠 Не нашёл ничего по этому запросу. "
            "Попробуй переформулировать или спроси в чате!"
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=cfg.gemini.api_key)
        model = genai.GenerativeModel(
            model_name=cfg.gemini.model,
            system_instruction=_BRAIN_SYSTEM,
        )
        prompt = (
            f"Вопрос пользователя: {query}\n\n"
            f"База знаний ({useful_count} записей):\n\n{context}"
        )
        response = model.generate_content(prompt)
        answer = (response.text or "").strip()
        if not answer:
            raise ValueError("Empty response from Gemini")
        return f"🧠 {answer}\n\n<i>На основе {useful_count} записей из базы знаний</i>"

    except Exception as e:
        log.error("Gemini МОЗГ synthesis failed: %s", e)
        return ""  # Caller fallbacks to DeepSeek


def is_enabled() -> bool:
    """True if Gemini МОЗГ is configured (GEMINI_API_KEY present)."""
    try:
        return bool(get_config().gemini.api_key)
    except Exception:
        return False
