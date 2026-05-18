#!/usr/bin/env python3
"""migrate_vertex_simple.py — линейная миграция через Vertex AI один-клиент.

Архитектура: один service-account → один Vertex genai.Client → один поток
выполнения, текст-за-текстом. Без worker pool, без key pool, без TPM
bookkeeping. Подходит когда:
- Все запросы идут с ОДНОЙ авторизации (Vertex SA из cfg.gemini.vertex_credentials_path)
- Квота per project достаточна (Vertex Tier 1 paid: 3K RPM, 1M TPM)
- SDK `embed_content(contents=str)` корректно возвращает один вектор
  на один текст (в отличие от `contents=list` где Vertex возвращает
  один вектор на весь список как multi-part doc)

Что делает:
1. fetch_all_pending(collection, model_tag) — все doc'и без актуального тэга
2. для каждого pending — один SDK-call → один vec → один coll.document().set()
3. progress-лог раз в 10 секунд

Запуск:
    python /app/scripts/migrate_vertex_simple.py --config /config/teledigest.conf

Через curl-pipe:
    curl -sS https://raw.githubusercontent.com/.../scripts/migrate_vertex_simple.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from teledigest.country_codes import country_full_name_en as country_full_name


def build_embed_text(doc: dict) -> str:
    """Unified text: 'Country. Title. Tag. Instruction'."""
    parts = []
    cf = country_full_name(doc.get("country") or "")
    if cf:
        parts.append(cf)
    title = (doc.get("title") or doc.get("sourceTitle") or "").strip()
    if title:
        parts.append(title)
    tag = (doc.get("tag") or "").strip()
    if tag:
        parts.append(tag)
    instruction = (doc.get("instruction") or "").strip()
    if instruction:
        parts.append(instruction)
    return ". ".join(parts)


def fetch_all_pending(coll, model_tag: str) -> list[tuple[str, dict, str]]:
    """Read entire collection, return only docs that need embedding."""
    PAGE_SIZE = 500
    pending: list[tuple[str, dict, str]] = []
    skipped = 0
    scanned = 0
    last_doc_id: str | None = None
    t0 = time.time()

    while True:
        q = coll.order_by("__name__").limit(PAGE_SIZE)
        if last_doc_id is not None:
            q = q.start_after(coll.document(last_doc_id).get())
        page = list(q.stream())
        if not page:
            break
        for snap in page:
            scanned += 1
            data = snap.to_dict() or {}
            if data.get("embedding_model") == model_tag:
                skipped += 1
                continue
            pending.append((snap.id, data, build_embed_text(data)))
        last_doc_id = page[-1].id
        if scanned % 5000 == 0:
            print(
                f"  fetch_all_pending: scanned={scanned} "
                f"pending={len(pending)} skipped(already-v2)={skipped} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if len(page) < PAGE_SIZE:
            break

    print(
        f"  fetch_all_pending DONE: scanned={scanned} "
        f"pending={len(pending)} skipped(already-v2)={skipped} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return pending


def migrate_collection(
    db,
    vertex_client,
    embed_cfg_factory,
    model_name,
    collection_name,
    model_tag,
    rpm: float = 5.0,
):
    """Sequential per-text migration on Vertex. Returns (migrated, failed).

    rpm: requests-per-minute throttle. На Vertex preview-моделях типа
    gemini-embedding-2-preview по дефолту 5 RPM (12 секунд между запросами).
    Сон рассчитывается между cycle start, не после ответа — чтобы общий
    темп точно был ≤ rpm независимо от latency."""
    from google.cloud.firestore_v1.vector import Vector

    coll = db.collection(collection_name)
    print(f"\n=== Migrating {collection_name} (throttle {rpm:.1f} RPM) ===")

    pending = fetch_all_pending(coll, model_tag)
    if not pending:
        print(f"  {collection_name}: nothing to do (all docs already v2-tagged)")
        return 0, 0

    migrated = 0
    failed = 0
    t_start = time.time()
    last_print = t_start
    min_cycle_s = 60.0 / max(rpm, 0.001)

    for doc_idx, (doc_id, src, text) in enumerate(pending):
        t_cycle_start = time.time()

        # 1. Embed one text via Vertex SDK (single string → single vector).
        try:
            r = vertex_client.models.embed_content(
                model=model_name,
                contents=text,
                config=embed_cfg_factory(),
            )
            vec = list(r.embeddings[0].values) if r.embeddings else None
        except Exception as e:
            print(f"  WARN: embed failed for {doc_id}: {str(e)[:200]}", flush=True)
            failed += 1
            # Sleep остаток окна даже на failure — иначе rate скачет.
            elapsed = time.time() - t_cycle_start
            if elapsed < min_cycle_s:
                time.sleep(min_cycle_s - elapsed)
            continue

        if vec is None:
            failed += 1
            elapsed = time.time() - t_cycle_start
            if elapsed < min_cycle_s:
                time.sleep(min_cycle_s - elapsed)
            continue

        # 2. Write to Firestore.
        instr = src.get("instruction") or ""
        payload = {
            **{k: v for k, v in src.items() if k != "embedding"},
            "embedding": Vector(vec),
            "embedding_model": model_tag,
            "embedded_text": text,
            "text_length": len(instr),
            "needs_chunking": len(instr) > 500,
        }
        try:
            coll.document(doc_id).set(payload)
            migrated += 1
        except Exception as e:
            print(f"  WARN: Firestore write failed for {doc_id}: {e}", flush=True)
            failed += 1

        # 3. Progress every 60 seconds (rate is low, no point in 10s logs).
        now = time.time()
        if now - last_print >= 60.0:
            rate = migrated / (now - t_start) if now > t_start else 0.0
            remaining = len(pending) - migrated - failed
            eta_hours = (remaining / rate / 3600) if rate > 0 else 0.0
            print(
                f"  {collection_name}: migrated={migrated}/{len(pending)} "
                f"failed={failed} rate={rate * 60:.1f} docs/min "
                f"ETA={eta_hours:.1f} hours",
                flush=True,
            )
            last_print = now

        # 4. Throttle: ensure cycle is at least min_cycle_s seconds.
        elapsed = time.time() - t_cycle_start
        if elapsed < min_cycle_s:
            time.sleep(min_cycle_s - elapsed)

    total_time = time.time() - t_start
    rate = migrated / total_time if total_time > 0 else 0.0
    print(
        f"  {collection_name} DONE: migrated={migrated} failed={failed} "
        f"total_time={total_time:.1f}s ({rate:.1f} docs/sec)",
        flush=True,
    )
    return migrated, failed


# Vertex defaults — отдельная жизнь от gemini_brain.py / config.py [gemini].
# Если что-то нужно переопределить — CLI-аргументами.
DEFAULT_VERTEX_PROJECT = "project-56cb62a9-8914-4ae3-b44"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_VERTEX_MODEL = "gemini-embedding-2-preview"
DEFAULT_VERTEX_CREDENTIALS = "/home/teledigest/data/vertex.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collections",
        default="wisdom_base,wikivoyage_base",
    )
    parser.add_argument("--vertex-project", default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--vertex-location", default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--vertex-model", default=DEFAULT_VERTEX_MODEL)
    parser.add_argument("--vertex-credentials", default=DEFAULT_VERTEX_CREDENTIALS)
    parser.add_argument(
        "--rpm",
        type=float,
        default=5.0,
        help="Requests-per-minute throttle. Vertex preview-модели обычно "
        "залочены на 5 RPM. Поднять до 50-1000 если квота увеличена.",
    )
    args = parser.parse_args()

    from teledigest.config import init_config

    # init_config нужен только чтобы Firestore-клиент мог стартануть
    # (Firestore SA — в [google] service_account_path).
    init_config(Path(args.config))

    from google import genai
    from google.genai import types as genai_types
    from google.oauth2 import service_account

    from teledigest.gemini_brain import _EMBEDDING_MODEL_TAG_V2, _build_firestore_client

    print(
        f"Using Vertex model: {args.vertex_model}\n"
        f"Project: {args.vertex_project}\n"
        f"Location: {args.vertex_location}\n"
        f"Credentials: {args.vertex_credentials}",
        flush=True,
    )

    creds = service_account.Credentials.from_service_account_file(
        args.vertex_credentials,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        vertexai=True,
        project=args.vertex_project,
        location=args.vertex_location,
        credentials=creds,
    )

    def embed_cfg_factory():
        return genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=1536,
        )

    db = _build_firestore_client()
    total_migrated = total_failed = 0
    for name in [c.strip() for c in args.collections.split(",") if c.strip()]:
        m, f = migrate_collection(
            db,
            client,
            embed_cfg_factory,
            args.vertex_model,
            name,
            _EMBEDDING_MODEL_TAG_V2,
            rpm=args.rpm,
        )
        total_migrated += m
        total_failed += f

    print(f"\n=== TOTAL: migrated={total_migrated} failed={total_failed} ===")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
