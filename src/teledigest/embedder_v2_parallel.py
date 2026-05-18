"""embedder_v2_parallel.py — параллельный эмбеддер с per-key TPM-pacing.

Замена `compute_document_embeddings_v2_batch` из gemini_brain.py.
Архитектура: каждый ключ — независимый worker-поток с собственным
token-bucket. Общая очередь chunks разбирается воркерами параллельно.

Чем отличается от старой реализации:

- Старая: последовательно идёт по ключам как resilience (если первый
  задушен — берём следующий). 8 ключей дают max ~30K TPM суммарно
  потому что в каждый момент работает один.

- Новая: 8 ключей × 30K TPM = 240K TPM суммарно. Каждый worker сам
  следит за своим минутным бюджетом и засыпает когда он близок к 30K.

Pacing — не "константа inter_chunk_sleep", а вычисляется из реального
размера chunk: pause = (tokens_sent / TPM) * 60. Шлёшь 18K — спишь 36с
до следующего HTTP на этом ключе. Шлёшь 1K — спишь 2с.

Использование из миграции:

    from teledigest.embedder_v2_parallel import embed_documents_parallel
    vectors = embed_documents_parallel(texts, dim=1536)

Возвращает list[list[float] | None] в том же порядке что входной список.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

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
MIN_PAUSE_AFTER_CHUNK = 0.1  # always yield CPU
MAX_PAUSE_AFTER_CHUNK = 60.0  # cap so retry_after surprises don't lock us
SWEEP_KEY_GAP = 5.0  # intra-sweep gap if ever needed (legacy safety)


# ---------------------------------------------------------------------------
# Token estimation + text splitting (reused from gemini_brain.py logic)
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


def pack_chunks(pieces: list[str]) -> list[Chunk]:
    """Pack pieces into chunks that fit one HTTP request.

    Greedy: accumulate pieces until either TARGET_TOKENS_PER_REQUEST or
    MAX_TEXTS_PER_REQUEST is hit, then start a new chunk."""
    chunks: list[Chunk] = []
    cur_texts: list[str] = []
    cur_indices: list[int] = []
    cur_tokens = 0

    for i, p in enumerate(pieces):
        tk = estimate_tokens(p)
        # If single piece exceeds request budget (shouldn't happen post-split,
        # but defend), send it alone.
        if tk > TARGET_TOKENS_PER_REQUEST:
            if cur_texts:
                chunks.append(Chunk(cur_texts, cur_indices, cur_tokens))
                cur_texts, cur_indices, cur_tokens = [], [], 0
            chunks.append(Chunk([p], [i], tk))
            continue
        # If adding this piece would overflow — close current chunk.
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
    """Tracks tokens sent per key in a rolling 60-second window.

    can_send(N) returns wait_seconds — 0 if you can send N tokens right
    now, positive value if you need to wait first."""

    def __init__(self, tpm: int = TPM_PER_KEY, rpm: int = RPM_PER_KEY) -> None:
        self.tpm = tpm
        self.rpm = rpm
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Drop events older than 60s. Caller must hold lock."""
        cutoff = now - 60.0
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def can_send(self, tokens: int) -> float:
        """Return seconds-to-wait before sending `tokens`. 0 = send now."""
        with self._lock:
            now = time.time()
            self._prune(now)
            used_tokens = sum(t for _, t in self._events)
            used_requests = len(self._events)
            # Both limits matter; take the longer wait.
            wait = 0.0
            if used_tokens + tokens > self.tpm and self._events:
                # Wait until oldest event ages out of the window.
                oldest_ts = self._events[0][0]
                wait = max(wait, (oldest_ts + 60.0) - now)
            if used_requests + 1 > self.rpm and self._events:
                oldest_ts = self._events[0][0]
                wait = max(wait, (oldest_ts + 60.0) - now)
            return max(wait, 0.0)

    def record(self, tokens: int) -> None:
        """Mark `tokens` as sent right now."""
        with self._lock:
            now = time.time()
            self._prune(now)
            self._events.append((now, tokens))


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
    rpd_count: int = 0  # success + 429 (Google counts both)


def _worker(
    key_idx: int,
    api_key: str,
    queue: Queue[Chunk],
    results: list[Optional[list[float]]],
    results_lock: threading.Lock,
    budget: KeyBudget,
    stats: WorkerStats,
    dim: int,
    stop_event: threading.Event,
) -> None:
    """One worker per key. Pulls chunks from queue, embeds with own key,
    writes results into shared list at piece_indices.

    Self-paced via KeyBudget: before each send, asks budget how long to
    sleep. After send, records token count. No global coordination needed
    beyond the queue."""
    from teledigest.gemini_brain import _embed_v2_rest_batch  # type: ignore

    while not stop_event.is_set():
        try:
            chunk = queue.get(timeout=1.0)
        except Empty:
            return  # queue drained

        # RPD soft-cap: stop using this key for the rest of the day.
        if stats.rpd_count >= RPD_SOFT_CAP:
            log.warning(
                "worker key #%d: RPD soft-cap reached (%d/%d), stopping",
                key_idx,
                stats.rpd_count,
                RPD_SOFT_CAP,
            )
            # Put the chunk back for another worker.
            queue.put(chunk)
            return

        # Wait until our budget allows this chunk.
        wait = budget.can_send(chunk.total_tokens)
        if wait > 0:
            log.debug(
                "worker key #%d: waiting %.1fs before %d-token chunk",
                key_idx,
                wait,
                chunk.total_tokens,
            )
            time.sleep(wait + 0.1)

        # Send.
        t0 = time.time()
        status, vectors, retry_after = _embed_v2_rest_batch(
            chunk.texts,
            "RETRIEVAL_DOCUMENT",
            dim,
            api_key,
        )
        dt = time.time() - t0
        stats.rpd_count += 1
        budget.record(chunk.total_tokens)

        if status == 200 and vectors is not None:
            stats.chunks_sent += 1
            stats.tokens_sent += chunk.total_tokens
            with results_lock:
                for piece_idx, vec in zip(chunk.piece_indices, vectors):
                    results[piece_idx] = vec
            log.info(
                "worker key #%d: chunk OK (%d texts, %d tokens, %.1fs) " "[rpd=%d/%d]",
                key_idx,
                len(chunk.texts),
                chunk.total_tokens,
                dt,
                stats.rpd_count,
                RPD_SOFT_CAP,
            )
        elif status == 429:
            stats.chunks_429 += 1
            wait = retry_after if retry_after else 30.0
            log.warning(
                "worker key #%d: 429 (retry_after=%.1fs), waiting [rpd=%d/%d]",
                key_idx,
                wait,
                stats.rpd_count,
                RPD_SOFT_CAP,
            )
            time.sleep(wait + 1.0)
            # Re-queue chunk for another attempt (any worker).
            queue.put(chunk)
        else:
            stats.chunks_failed += 1
            log.warning(
                "worker key #%d: non-retriable status=%s, marking pieces failed",
                key_idx,
                status,
            )
            with results_lock:
                for piece_idx in chunk.piece_indices:
                    results[piece_idx] = None


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for compute_document_embeddings_v2_batch
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
) -> list[Optional[list[float]]]:
    """Parallel embedding pipeline.

    Each text → optional split into pieces → packed into chunks → workers
    (one per key) drain queue concurrently with per-key TPM pacing →
    pieces of one doc are averaged back into one vector.

    Returns list of vectors (or None on failure) in original text order.
    """
    if not texts:
        return []

    if keys is None:
        keys = get_api_keys()
    if not keys:
        log.warning("embed_documents_parallel: no API keys")
        return [None] * len(texts)

    # Step 1: split long texts into pieces.
    pieces: list[str] = []
    doc_to_piece_idx: list[list[int]] = []
    n_split = 0
    for t in texts:
        ps = split_long_text(t)
        if len(ps) > 1:
            n_split += 1
        idxs = []
        for p in ps:
            idxs.append(len(pieces))
            pieces.append(p)
        doc_to_piece_idx.append(idxs)

    log.info(
        "embed_parallel: %d docs → %d pieces (%d needed splitting)",
        len(texts),
        len(pieces),
        n_split,
    )

    # Step 2: pack pieces into HTTP-sized chunks.
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

    # Step 3: spin up workers, one per key.
    queue: Queue[Chunk] = Queue()
    for c in chunks:
        queue.put(c)

    results: list[Optional[list[float]]] = [None] * len(pieces)
    results_lock = threading.Lock()
    stop_event = threading.Event()
    stats: list[WorkerStats] = []
    threads: list[threading.Thread] = []

    for i, key in enumerate(keys):
        budget = KeyBudget()
        s = WorkerStats(key_idx=i)
        stats.append(s)
        th = threading.Thread(
            target=_worker,
            args=(i, key, queue, results, results_lock, budget, s, dim, stop_event),
            daemon=True,
            name=f"embed-worker-{i}",
        )
        th.start()
        threads.append(th)

    # Step 4: wait for queue to drain. Workers exit when queue is empty
    # for >1s (timeout in worker .get()).
    for th in threads:
        th.join()

    # Step 5: merge pieces back into per-doc vectors.
    out: list[Optional[list[float]]] = []
    for idxs in doc_to_piece_idx:
        # Filter Nones explicitly so the type narrows to list[list[float]].
        vecs: list[list[float]] = [
            results[i] for i in idxs if results[i] is not None  # type: ignore[misc]
        ]
        if not vecs:
            out.append(None)
        elif len(vecs) == 1:
            out.append(vecs[0])
        else:
            out.append(average_vectors(vecs))

    # Step 6: summary.
    total_sent = sum(s.chunks_sent for s in stats)
    total_429 = sum(s.chunks_429 for s in stats)
    total_failed = sum(s.chunks_failed for s in stats)
    log.info(
        "embed_parallel: done. sent=%d 429s=%d failed=%d",
        total_sent,
        total_429,
        total_failed,
    )
    for s in stats:
        log.info(
            "  key #%d: chunks=%d tokens=%d 429s=%d rpd=%d",
            s.key_idx,
            s.chunks_sent,
            s.tokens_sent,
            s.chunks_429,
            s.rpd_count,
        )

    return out
