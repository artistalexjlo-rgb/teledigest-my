"""embedder_v2_parallel.py — параллельный эмбеддер с per-key TPM-pacing.

Заменяет `compute_document_embeddings_v2_batch` из gemini_brain.py для
крупных бэкфилл-задач (миграция wikivoyage_base ~73K docs).

Архитектура (по TZ migration_pipeline_tz.md):

    pre-fetch ALL pending → split into pieces → pack into chunks →
    fill shared queue ONCE → start N workers (one per key) →
    each worker:
        check own cooldown (sleep if needed)
        dequeue chunk
        check own TPM-budget (sleep if needed)
        send via REST :batchEmbedContents
        on 200 → record pieces, mark complete docs, fire callback
        on 429 → set own cooldown to retry_after, push chunk to END of
                 queue (max N attempts), continue
        on other error → mark chunk's pieces failed, continue

Что критически отличается от старой реализации:
- Pre-fetch ВСЕХ непомеченных доков ДО старта воркеров (раньше
  миграция читала по 200 docs за раз → 2 chunks → 5 из 7 воркеров
  простаивали).
- 429 не возвращается в начало FIFO. Воркер засыпает на cooldown,
  потом дёргает СЛЕДУЮЩИЙ chunk из очереди. Failed chunk попадает
  в хвост и достанется когда у воркера квота восстановится.
- on_doc_complete callback — даёт миграции писать в Firestore
  ИНКРЕМЕНТАЛЬНО по мере готовности доков, а не в конце. Если
  процесс упадёт — записанное останется, при рестарте перезапустим
  только непомеченные.
- Per-worker cooldown_until уважает Google-овский retry_after из тела
  429-ответа, а не только локальный TPM-бюджет.

Использование из миграции:

    from teledigest.embedder_v2_parallel import embed_documents_parallel

    def write_back(doc_idx: int, vec: list[float] | None) -> None:
        # called from worker thread as soon as one doc is fully embedded
        if vec is not None:
            firestore_doc.set({..., "embedding": Vector(vec)})

    embed_documents_parallel(texts, dim=1536, on_doc_complete=write_back)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (sane free-tier defaults for gemini-embedding-2)
# ---------------------------------------------------------------------------

# Per-key minute window (Google free-tier limits).
TPM_PER_KEY = 30000
RPM_PER_KEY = 100
RPD_PER_KEY = 1000  # daily cap; we stop at SOFT cap to leave margin
RPD_SOFT_CAP = 950

# Per HTTP request limits.
# batchEmbedContents accepts up to ~100 requests OR ~20K tokens per call,
# whichever hits first. We aim for 18K to leave safety margin.
TARGET_TOKENS_PER_REQUEST = 18000
MAX_TEXTS_PER_REQUEST = 100

# Per-text size cap. Texts larger than this get split into pieces and
# embeddings are averaged back. Avoids one huge text poisoning a chunk.
MAX_TOKENS_PER_TEXT = 1500
SPLIT_OVERLAP_CHARS = 200

# Pacing constants.
DEFAULT_429_COOLDOWN_S = 30.0  # if Google's 429 didn't include retry_after
MAX_CHUNK_ATTEMPTS = 5  # give up on a chunk after N 429s, mark its pieces None

# RPD-state persistence (Step 5 of TZ). Survives process restart so a
# crashed migration doesn't blow past 1000/day on resume.
RPD_STATE_PATH = Path(
    os.environ.get("EMBED_RPD_STATE_FILE", "/tmp/teledigest_keys_state.json")
)


# ---------------------------------------------------------------------------
# Token estimation + text splitting
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Conservative token count: 1 token ≈ 3 chars (real ratio ~4, we
    overestimate to stay under TPM rather than burst over)."""
    return max(1, len(text) // 3)


def split_long_text(text: str, max_tokens: int = MAX_TOKENS_PER_TEXT) -> list[str]:
    """Split text into pieces each ≤ max_tokens. Prefers paragraph then
    sentence boundaries. Keeps small overlap between pieces."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    max_chars = max_tokens * 3
    pieces: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            pieces.append(remaining)
            break
        cut = remaining.rfind("\n\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = remaining.rfind(". ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
        piece = remaining[:cut].rstrip()
        if piece:
            pieces.append(piece)
        next_start = max(cut - SPLIT_OVERLAP_CHARS, cut - max_chars // 4)
        next_start = max(next_start, 1)
        remaining = remaining[next_start:].lstrip()

    return pieces or [text]


def average_vectors(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of equal-dim vectors. Used to combine embeddings
    of multiple pieces of one document back into a single vector."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    summed = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            summed[i] += v[i]
    n = float(len(vectors))
    return [s / n for s in summed]


# ---------------------------------------------------------------------------
# Chunk packing — pack pieces into HTTP-request-sized batches
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One HTTP-request worth of texts. piece_indices map back to the
    flat piece list so worker can write results to the right slots."""

    texts: list[str]
    piece_indices: list[int]
    total_tokens: int
    attempts: int = 0  # incremented on 429, capped at MAX_CHUNK_ATTEMPTS


def pack_chunks(pieces: list[str]) -> list[Chunk]:
    """Pack pieces into chunks that fit one HTTP request."""
    chunks: list[Chunk] = []
    cur_texts: list[str] = []
    cur_indices: list[int] = []
    cur_tokens = 0

    for i, p in enumerate(pieces):
        tk = estimate_tokens(p)
        if tk > TARGET_TOKENS_PER_REQUEST:
            if cur_texts:
                chunks.append(Chunk(cur_texts, cur_indices, cur_tokens))
                cur_texts, cur_indices, cur_tokens = [], [], 0
            chunks.append(Chunk([p], [i], tk))
            continue
        if (
            cur_tokens + tk > TARGET_TOKENS_PER_REQUEST
            or len(cur_texts) >= MAX_TEXTS_PER_REQUEST
        ):
            chunks.append(Chunk(cur_texts, cur_indices, cur_tokens))
            cur_texts, cur_indices, cur_tokens = [], [], 0
        cur_texts.append(p)
        cur_indices.append(i)
        cur_tokens += tk

    if cur_texts:
        chunks.append(Chunk(cur_texts, cur_indices, cur_tokens))
    return chunks


# ---------------------------------------------------------------------------
# Per-key token-bucket — tracks rolling 60-second TPM usage
# ---------------------------------------------------------------------------


class KeyBudget:
    """Per-key adaptive pace — wait proportional to tokens after each send.

    Formula from TZ migration_pipeline_tz.md §2.1:
        pause = (tokens_sent / TPM) * 60

    После send'а 11K токенов ждём 11K/30K*60 = 22 секунды до следующего
    HTTP на этом ключе. После 30K — 60 секунд. Это smooth rate-limiting:
    долгосрочный темп никогда не превышает TPM_PER_KEY токенов/мин,
    короткие burst'ы тоже невозможны (один запрос блокирует ключ
    пропорционально его размеру).

    Раньше тут была logика "fill window до 30K → wait 65с" — она
    разрешала 11K+10K=21K за 10 секунд (в окне), Google ловил burst и
    клал 429 на весь project (per User/Project/Model лимит).
    """

    def __init__(self, tpm: int = TPM_PER_KEY, rpm: int = RPM_PER_KEY) -> None:
        self.tpm = tpm
        self.rpm = rpm  # kept for API compat; TPM is the binding constraint
        self._next_allowed_at = 0.0

    def can_send(self, tokens: int) -> float:
        """Return seconds to wait before sending `tokens`. 0 = send now."""
        now = time.time()
        return max(self._next_allowed_at - now, 0.0)

    def record(self, tokens: int) -> None:
        """Mark `tokens` as sent — schedule next-allowed time proportionally."""
        now = time.time()
        delay = (tokens / self.tpm) * 60.0
        self._next_allowed_at = now + delay


# ---------------------------------------------------------------------------
# RPD state persistence (Step 5 of TZ)
# ---------------------------------------------------------------------------


def _today_pt() -> str:
    """UTC date string — Google's quota reset is at midnight Pacific Time
    which is UTC 07:00/08:00 depending on DST. We grossly approximate with
    plain UTC date here; operator should manually delete the state file
    if running a fresh batch after PT reset but before our UTC midnight."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def load_rpd_state() -> dict[int, int]:
    """Load per-key RPD counter from disk. Returns {} if file missing or
    the date doesn't match (auto-reset on day rollover)."""
    if not RPD_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(RPD_STATE_PATH.read_text())
    except Exception as e:
        log.warning(
            "rpd_state: failed to load %s (%s) — starting fresh", RPD_STATE_PATH, e
        )
        return {}
    if data.get("date") != _today_pt():
        log.info(
            "rpd_state: stale (date=%s, today=%s) — resetting",
            data.get("date"),
            _today_pt(),
        )
        return {}
    raw = data.get("rpd", {})
    # JSON keys are strings; cast back to int.
    return {int(k): int(v) for k, v in raw.items()}


def save_rpd_state(rpd: dict[int, int]) -> None:
    """Persist per-key RPD counter to disk."""
    try:
        RPD_STATE_PATH.write_text(json.dumps({"date": _today_pt(), "rpd": rpd}))
    except Exception as e:
        log.warning("rpd_state: failed to save %s (%s)", RPD_STATE_PATH, e)


# ---------------------------------------------------------------------------
# Worker — one per key
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    key_idx: int
    chunks_sent: int = 0
    tokens_sent: int = 0
    chunks_429: int = 0
    chunks_failed: int = 0
    chunks_gave_up: int = 0
    rpd_count: int = 0


@dataclass
class SharedState:
    """State shared across workers — protected by results_lock."""

    results: list[Optional[list[float]]]
    remaining_pieces_per_doc: dict[int, int]
    piece_to_doc: dict[int, int]
    doc_to_piece_idx: list[list[int]]
    on_doc_complete: Optional[Callable[[int, Optional[list[float]]], None]]
    rpd_global: dict[int, int] = field(default_factory=dict)


def _mark_chunk_results(
    chunk: Chunk,
    vectors: list[Optional[list[float]]] | None,
    state: SharedState,
    results_lock: threading.Lock,
) -> list[tuple[int, Optional[list[float]]]]:
    """Write chunk's piece-vectors into shared results, decrement per-doc
    counters, return list of (doc_idx, vec_or_none) for newly-completed
    docs. Caller invokes on_doc_complete outside the lock."""
    newly_completed: list[int] = []
    with results_lock:
        if vectors is None:
            # Chunk gave up entirely — mark all its pieces as None.
            for piece_idx in chunk.piece_indices:
                state.results[piece_idx] = None
                doc_idx = state.piece_to_doc[piece_idx]
                state.remaining_pieces_per_doc[doc_idx] -= 1
                if state.remaining_pieces_per_doc[doc_idx] == 0:
                    newly_completed.append(doc_idx)
        else:
            for piece_idx, vec in zip(chunk.piece_indices, vectors):
                state.results[piece_idx] = vec
                doc_idx = state.piece_to_doc[piece_idx]
                state.remaining_pieces_per_doc[doc_idx] -= 1
                if state.remaining_pieces_per_doc[doc_idx] == 0:
                    newly_completed.append(doc_idx)

    # Compute final per-doc vectors outside the lock.
    out: list[tuple[int, Optional[list[float]]]] = []
    for doc_idx in newly_completed:
        piece_idxs = state.doc_to_piece_idx[doc_idx]
        good_vecs = [
            state.results[i] for i in piece_idxs if state.results[i] is not None
        ]
        if not good_vecs:
            out.append((doc_idx, None))
        elif len(good_vecs) == 1:
            out.append((doc_idx, good_vecs[0]))
        else:
            out.append((doc_idx, average_vectors(good_vecs)))  # type: ignore[arg-type]
    return out


def _worker(
    key_idx: int,
    api_key: str,
    queue: Queue[Chunk],
    state: SharedState,
    results_lock: threading.Lock,
    budget: KeyBudget,
    stats: WorkerStats,
    dim: int,
    stop_event: threading.Event,
    rpd_state_lock: threading.Lock,
) -> None:
    """One worker per key.

    Loop:
      1. Check own cooldown (cooldown_until). Sleep if cooling.
      2. Check RPD soft-cap. Exit thread if exhausted (for the day).
      3. Dequeue chunk (timeout 2s → exit on empty queue).
      4. Check budget.can_send(chunk.total_tokens). Sleep if needed.
      5. Send via _embed_v2_rest_batch.
      6. On 200 → mark pieces, fire callbacks, persist RPD.
      7. On 429 → set own cooldown to retry_after, increment chunk.attempts,
         re-queue at END (give up if attempts >= MAX_CHUNK_ATTEMPTS).
      8. On other status → mark chunk's pieces failed.
    """
    from teledigest.gemini_brain import _embed_v2_rest_batch  # type: ignore

    # Vertex paid tier has Unlimited RPD — soft cap не применяется.
    # Free tier (Gemini API): cap 1000 RPD per (account, project, model),
    # стопимся на 950.
    skip_rpd_cap = False
    try:
        from teledigest.config import get_config

        skip_rpd_cap = bool(get_config().gemini.use_vertex)
    except Exception:
        skip_rpd_cap = False

    cooldown_until = 0.0

    while not stop_event.is_set():
        # 1. Own cooldown.
        now = time.time()
        if cooldown_until > now:
            time.sleep(min(cooldown_until - now + 0.1, 90.0))
            continue

        # 2. RPD soft-cap (free-tier only).
        if not skip_rpd_cap and stats.rpd_count >= RPD_SOFT_CAP:
            log.warning(
                "worker key #%d: RPD soft-cap reached (%d/%d), worker exiting",
                key_idx,
                stats.rpd_count,
                RPD_SOFT_CAP,
            )
            return

        # 3. Dequeue.
        try:
            chunk = queue.get(timeout=2.0)
        except Empty:
            return  # queue drained — exit worker

        # 4. TPM/RPM budget.
        wait = budget.can_send(chunk.total_tokens)
        if wait > 0:
            log.debug(
                "worker key #%d: TPM budget wait %.1fs (%d tokens)",
                key_idx,
                wait,
                chunk.total_tokens,
            )
            time.sleep(wait + 0.1)

        # 5. Send.
        t0 = time.time()
        status, vectors, retry_after = _embed_v2_rest_batch(
            chunk.texts,
            "RETRIEVAL_DOCUMENT",
            dim,
            api_key,
        )
        dt = time.time() - t0
        # Google considers each text inside batchEmbedContents as a separate
        # RPD tick. So a 100-text chunk = +100 toward the 1000/day cap, NOT +1.
        # Both 200-OK and 429 burn quota (Google still meters the request).
        stats.rpd_count += len(chunk.texts)
        budget.record(chunk.total_tokens)

        # Update shared RPD state (used for persist).
        with rpd_state_lock:
            state.rpd_global[key_idx] = stats.rpd_count

        if status == 200 and vectors is not None:
            stats.chunks_sent += 1
            stats.tokens_sent += chunk.total_tokens
            completed = _mark_chunk_results(chunk, vectors, state, results_lock)
            log.info(
                "worker key #%d: chunk OK (%d texts, %d tokens, %.1fs) "
                "[rpd=%d/%d, %d docs finished]",
                key_idx,
                len(chunk.texts),
                chunk.total_tokens,
                dt,
                stats.rpd_count,
                RPD_SOFT_CAP,
                len(completed),
            )
            # Persist RPD on every success (cheap, ~100 bytes file).
            with rpd_state_lock:
                save_rpd_state(state.rpd_global)
            # Fire callbacks OUTSIDE locks.
            if state.on_doc_complete:
                for doc_idx, vec in completed:
                    try:
                        state.on_doc_complete(doc_idx, vec)
                    except Exception as e:
                        log.warning(
                            "on_doc_complete(doc_idx=%d) raised: %s — continuing",
                            doc_idx,
                            e,
                        )
        elif status == 429:
            stats.chunks_429 += 1
            wait_s = retry_after if retry_after else DEFAULT_429_COOLDOWN_S
            cooldown_until = time.time() + wait_s + 1.0
            chunk.attempts += 1
            if chunk.attempts >= MAX_CHUNK_ATTEMPTS:
                stats.chunks_gave_up += 1
                log.warning(
                    "worker key #%d: chunk gave up after %d attempts — "
                    "marking %d pieces failed",
                    key_idx,
                    chunk.attempts,
                    len(chunk.piece_indices),
                )
                completed = _mark_chunk_results(chunk, None, state, results_lock)
                if state.on_doc_complete:
                    for doc_idx, vec in completed:
                        try:
                            state.on_doc_complete(doc_idx, vec)
                        except Exception as e:
                            log.warning("on_doc_complete raised: %s", e)
            else:
                log.warning(
                    "worker key #%d: 429 (retry_after=%.1fs, attempt %d/%d) "
                    "— cooling, chunk back to queue [rpd=%d/%d]",
                    key_idx,
                    wait_s,
                    chunk.attempts,
                    MAX_CHUNK_ATTEMPTS,
                    stats.rpd_count,
                    RPD_SOFT_CAP,
                )
                queue.put(chunk)  # back of queue
        else:
            # Non-retriable (400/401/403/5xx/network). Mark pieces failed.
            stats.chunks_failed += 1
            log.warning(
                "worker key #%d: non-retriable status=%s — marking %d pieces failed",
                key_idx,
                status,
                len(chunk.piece_indices),
            )
            completed = _mark_chunk_results(chunk, None, state, results_lock)
            if state.on_doc_complete:
                for doc_idx, vec in completed:
                    try:
                        state.on_doc_complete(doc_idx, vec)
                    except Exception as e:
                        log.warning("on_doc_complete raised: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_api_keys() -> list[str]:
    """Read keys from GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY."""
    plural = os.environ.get("GEMINI_API_KEYS", "")
    if plural:
        keys = [k.strip() for k in plural.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GEMINI_API_KEY", "")
    return [single.strip()] if single.strip() else []


def embed_documents_parallel(
    texts: list[str],
    dim: int = 1536,
    keys: Optional[list[str]] = None,
    on_doc_complete: Optional[Callable[[int, Optional[list[float]]], None]] = None,
) -> list[Optional[list[float]]]:
    """Parallel embedding pipeline.

    Pre-fetches all texts, splits long ones into pieces, packs into
    HTTP-sized chunks, fills a shared queue ONCE, then starts one worker
    per key. Workers drain queue concurrently with per-key TPM pacing,
    per-key cooldown on 429, and back-of-queue retry (max 5 attempts).

    If `on_doc_complete(doc_idx, vec_or_none)` is provided, it's called
    from worker threads as soon as a doc is fully embedded. This lets
    callers (e.g. migration) write to Firestore incrementally so a crash
    doesn't lose all completed work.

    Returns final list of vectors (or None) in original `texts` order.
    """
    if not texts:
        return []

    if keys is None:
        keys = get_api_keys()
    if not keys:
        log.warning("embed_documents_parallel: no API keys")
        if on_doc_complete:
            for i in range(len(texts)):
                on_doc_complete(i, None)
        return [None] * len(texts)

    # Step 1: split long texts into pieces.
    pieces: list[str] = []
    doc_to_piece_idx: list[list[int]] = []
    piece_to_doc: dict[int, int] = {}
    n_split = 0
    for doc_idx, t in enumerate(texts):
        ps = split_long_text(t)
        if len(ps) > 1:
            n_split += 1
        idxs = []
        for p in ps:
            piece_to_doc[len(pieces)] = doc_idx
            idxs.append(len(pieces))
            pieces.append(p)
        doc_to_piece_idx.append(idxs)

    log.info(
        "embed_parallel: %d docs → %d pieces (%d needed splitting)",
        len(texts),
        len(pieces),
        n_split,
    )

    # Step 2: pack pieces into chunks (entire job at once).
    chunks = pack_chunks(pieces)
    total_tokens = sum(c.total_tokens for c in chunks)
    log.info(
        "embed_parallel: %d chunks, %d total tokens, %d keys, "
        "theoretical min time: %.1fs",
        len(chunks),
        total_tokens,
        len(keys),
        total_tokens / (TPM_PER_KEY * len(keys)) * 60,
    )

    # Step 3: fill queue ONCE.
    queue: Queue[Chunk] = Queue()
    for c in chunks:
        queue.put(c)

    # Step 4: shared state for workers.
    state = SharedState(
        results=[None] * len(pieces),
        remaining_pieces_per_doc={
            doc_idx: len(piece_idxs)
            for doc_idx, piece_idxs in enumerate(doc_to_piece_idx)
        },
        piece_to_doc=piece_to_doc,
        doc_to_piece_idx=doc_to_piece_idx,
        on_doc_complete=on_doc_complete,
        rpd_global=load_rpd_state(),  # restore from disk (Step 5 of TZ)
    )
    if state.rpd_global:
        log.info(
            "embed_parallel: restored RPD state from %s: %s",
            RPD_STATE_PATH,
            state.rpd_global,
        )
    results_lock = threading.Lock()
    rpd_state_lock = threading.Lock()
    stop_event = threading.Event()
    stats: list[WorkerStats] = []
    threads: list[threading.Thread] = []

    # Step 5: spin up workers.
    for i, key in enumerate(keys):
        budget = KeyBudget()
        s = WorkerStats(key_idx=i, rpd_count=state.rpd_global.get(i, 0))
        stats.append(s)
        th = threading.Thread(
            target=_worker,
            args=(
                i,
                key,
                queue,
                state,
                results_lock,
                budget,
                s,
                dim,
                stop_event,
                rpd_state_lock,
            ),
            daemon=True,
            name=f"embed-worker-{i}",
        )
        th.start()
        threads.append(th)

    # Step 6: wait for queue to drain (workers exit on empty-queue timeout).
    for th in threads:
        th.join()

    # Step 7: build final per-doc vector list (mostly already written via
    # callbacks, but we still need to return the in-memory list for callers
    # that don't use the callback).
    out: list[Optional[list[float]]] = []
    for idxs in doc_to_piece_idx:
        vecs: list[list[float]] = [
            state.results[i] for i in idxs if state.results[i] is not None  # type: ignore[misc]
        ]
        if not vecs:
            out.append(None)
        elif len(vecs) == 1:
            out.append(vecs[0])
        else:
            out.append(average_vectors(vecs))

    # Step 8: summary.
    total_sent = sum(s.chunks_sent for s in stats)
    total_429 = sum(s.chunks_429 for s in stats)
    total_failed = sum(s.chunks_failed for s in stats)
    total_gave_up = sum(s.chunks_gave_up for s in stats)
    log.info(
        "embed_parallel: done. sent=%d 429s=%d failed=%d gave_up=%d",
        total_sent,
        total_429,
        total_failed,
        total_gave_up,
    )
    for s in stats:
        log.info(
            "  key #%d: chunks=%d tokens=%d 429s=%d gave_up=%d rpd=%d",
            s.key_idx,
            s.chunks_sent,
            s.tokens_sent,
            s.chunks_429,
            s.chunks_gave_up,
            s.rpd_count,
        )

    return out
