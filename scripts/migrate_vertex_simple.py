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
    db, vertex_client, embed_cfg_factory, model_name, collection_name, model_tag
):
    """Sequential per-text migration on Vertex. Returns (migrated, failed)."""
    from google.cloud.firestore_v1.vector import Vector

    coll = db.collection(collection_name)
    print(f"\n=== Migrating {collection_name} ===")

    pending = fetch_all_pending(coll, model_tag)
    if not pending:
        print(f"  {collection_name}: nothing to do (all docs already v2-tagged)")
        return 0, 0

    migrated = 0
    failed = 0
    t_start = time.time()
    last_print = t_start

    for doc_idx, (doc_id, src, text) in enumerate(pending):
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
            continue

        if vec is None:
            failed += 1
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

        # 3. Progress every 10 seconds.
        now = time.time()
        if now - last_print >= 10.0:
            rate = migrated / (now - t_start) if now > t_start else 0.0
            remaining = len(pending) - migrated - failed
            eta_min = (remaining / rate / 60) if rate > 0 else 0.0
            print(
                f"  {collection_name}: migrated={migrated}/{len(pending)} "
                f"failed={failed} rate={rate:.1f} docs/sec "
                f"ETA={eta_min:.1f} min",
                flush=True,
            )
            last_print = now

    total_time = time.time() - t_start
    rate = migrated / total_time if total_time > 0 else 0.0
    print(
        f"  {collection_name} DONE: migrated={migrated} failed={failed} "
        f"total_time={total_time:.1f}s ({rate:.1f} docs/sec)",
        flush=True,
    )
    return migrated, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collections",
        default="wisdom_base,wikivoyage_base",
    )
    args = parser.parse_args()

    from teledigest.config import get_config, init_config

    init_config(Path(args.config))
    cfg = get_config()

    if not cfg.gemini.use_vertex:
        print(
            "ERROR: cfg.gemini.use_vertex is false. This script REQUIRES Vertex.\n"
            "Set [gemini] use_vertex = true in /config/teledigest.conf."
        )
        return 1

    from google import genai
    from google.genai import types as genai_types
    from google.oauth2 import service_account

    from teledigest.gemini_brain import _EMBEDDING_MODEL_TAG_V2, _build_firestore_client

    print(
        f"Using Vertex model: {cfg.gemini.vertex_model}\n"
        f"Project: {cfg.gemini.vertex_project}\n"
        f"Location: {cfg.gemini.vertex_location}\n"
        f"Credentials: {cfg.gemini.vertex_credentials_path}",
        flush=True,
    )

    creds = service_account.Credentials.from_service_account_file(
        str(cfg.gemini.vertex_credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        vertexai=True,
        project=cfg.gemini.vertex_project,
        location=cfg.gemini.vertex_location,
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
            cfg.gemini.vertex_model,
            name,
            _EMBEDDING_MODEL_TAG_V2,
        )
        total_migrated += m
        total_failed += f

    print(f"\n=== TOTAL: migrated={total_migrated} failed={total_failed} ===")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
