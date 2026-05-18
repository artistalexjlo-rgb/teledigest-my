#!/usr/bin/env python3
"""migrate_to_cipher_v2_parallel.py — v2 миграция через параллельный эмбеддер.

Архитектура (по TZ migration_pipeline_tz.md):

    1. fetch_all_pending — читает ВСЕ непомеченные документы коллекции
       в память до старта эмбеддера. Для wikivoyage_base после wipe это
       ~73K docs, ~50-100MB. Достаточно одного прохода чтобы
       embed_documents_parallel получил полный список и сразу заполнил
       очередь chunks.
    2. embed_documents_parallel с on_doc_complete callback'ом, который
       пишет каждый готовый embedding в Firestore СРАЗУ. Если миграция
       упадёт — записанные доки остаются помеченными, при перезапуске
       fetch_all_pending их пропустит.

Запуск:

    python scripts/migrate_to_cipher_v2_parallel.py --config /config/teledigest.conf

Через curl-pipe:

    curl -s .../scripts/migrate_to_cipher_v2_parallel.py | \\
        docker exec -i $CID python - --config /config/teledigest.conf
"""

from __future__ import annotations

import argparse
import sys
import threading
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
    """Read entire collection, return only docs that need embedding.

    Returns list of (doc_id, src_data, embed_text). Skips docs that
    already carry the current model_tag (already-migrated)."""
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
    collection_name: str,
    model_tag: str,
) -> tuple[int, int, int]:
    """Returns (migrated, skipped_already_v2, failed)."""
    from google.cloud.firestore_v1.vector import Vector

    from teledigest.embedder_v2_parallel import embed_documents_parallel

    coll = db.collection(collection_name)

    print(f"\n=== Migrating {collection_name} ===")
    pending = fetch_all_pending(coll, model_tag)
    if not pending:
        print(f"  {collection_name}: nothing to do (all docs already v2-tagged)")
        return 0, 0, 0

    # Track counters across worker threads.
    migrated = 0
    failed = 0
    counter_lock = threading.Lock()
    t_start = time.time()
    last_progress_print = t_start

    def write_back(doc_idx: int, vec: list[float] | None) -> None:
        """Called by embed_documents_parallel from worker threads as soon
        as each doc is fully embedded. Writes to Firestore immediately so
        a crash doesn't lose completed work."""
        nonlocal migrated, failed, last_progress_print
        doc_id, src, text = pending[doc_idx]
        if vec is None:
            with counter_lock:
                failed += 1
            return
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
        except Exception as e:
            # Don't propagate — log and count as failed so the whole
            # migration doesn't blow up over a single Firestore hiccup.
            print(f"  WARN: Firestore write for {doc_id} failed: {e}", flush=True)
            with counter_lock:
                failed += 1
            return

        with counter_lock:
            migrated += 1
            now = time.time()
            if now - last_progress_print >= 10.0:
                rate = migrated / (now - t_start) if now > t_start else 0.0
                remaining = len(pending) - migrated - failed
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"  {collection_name}: migrated={migrated}/{len(pending)} "
                    f"failed={failed} rate={rate:.1f} docs/sec "
                    f"ETA={eta / 60:.1f} min",
                    flush=True,
                )
                last_progress_print = now

    texts = [t for _, _, t in pending]

    # В Vertex-режиме все воркеры делят одну service-account auth = одну
    # квоту проекта. >1 воркера = burst → 429. Используем vertex_worker_count
    # из конфига (default 1). В free-tier режиме keys=None — берётся
    # GEMINI_API_KEYS из env, один воркер на ключ.
    from teledigest.config import get_config

    cfg = get_config()
    if cfg.gemini.use_vertex:
        keys = [f"vertex-{i}" for i in range(cfg.gemini.vertex_worker_count)]
    else:
        keys = None

    embed_documents_parallel(texts, dim=1536, keys=keys, on_doc_complete=write_back)

    total_time = time.time() - t_start
    print(
        f"  {collection_name} DONE: migrated={migrated} failed={failed} "
        f"total_time={total_time:.1f}s "
        f"({migrated / total_time:.1f} docs/sec)",
        flush=True,
    )
    return migrated, 0, failed  # skipped already counted inside fetch_all_pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="/config/teledigest.conf")
    parser.add_argument(
        "--collections",
        default="wisdom_base,wikivoyage_base",
    )
    args = parser.parse_args()

    from teledigest.config import init_config

    init_config(Path(args.config))

    from teledigest.gemini_brain import _EMBEDDING_MODEL_TAG_V2, _build_firestore_client

    db = _build_firestore_client()
    total_migrated = total_failed = 0
    for name in [c.strip() for c in args.collections.split(",") if c.strip()]:
        m, _s, f = migrate_collection(db, name, _EMBEDDING_MODEL_TAG_V2)
        total_migrated += m
        total_failed += f

    print(f"\n=== TOTAL: migrated={total_migrated} failed={total_failed} ===")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
