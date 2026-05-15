from __future__ import annotations

import asyncio
import os
import time as _time
from .config import get_config, log

_BRAIN_SYSTEM = """\
You are an assistant bot for an expat community chat called "МОЗГ".
Answer the user's question using ONLY the facts in the knowledge base below.
... (rest of your system prompt) ...
"""

# ... (константы эмбеддингов остаются прежними) ...
_EMBEDDING_MODEL_V2 = "gemini-embedding-2"
_EMBEDDING_DIM_V2 = 1536

def _get_embedding_api_keys() -> list[str]:
    """Возвращает список всех доступных ключей из переменных окружения."""
    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    if raw_multi:
        keys = [k.strip() for k in raw_multi.split(",") if k.strip()]
        if keys:
            return keys
    
    # Фолбэк на одиночный ключ
    cfg = get_config()
    single = getattr(cfg.gemini, "api_key", None) or os.environ.get("GEMINI_API_KEY", "")
    return [str(single)] if single else []

_key_rr_idx = 0

def _parse_retry_after_seconds(body_text: str) -> float | None:
    import re
    m = re.search(r"retry in ([\d.]+)s", body_text or "")
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

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
    except Exception as exc:
        log.warning("v2 batch REST: сетевая ошибка: %s", exc)
        return -1, None, None
    
    if resp.status_code != 200:
        retry_after = _parse_retry_after_seconds(resp.text)
        return resp.status_code, None, retry_after
        
    data = resp.json()
    embeds = data.get("embeddings", [])
    out = [list(emb.get("values") or []) for emb in embeds]
    while len(out) < len(texts): out.append(None)
    return 200, out[: len(texts)], None

def compute_document_embeddings_v2_batch(
    texts: list[str],
    dim: int = _EMBEDDING_DIM_V2,
    chunk_size: int = 10, # Для тяжелого WikiVoyage лучше держать 10
) -> list[list[float] | None]:
    """Массовое получение эмбеддингов с ротацией ключей и защитой от TPM 429."""
    if not texts: return []

    keys = _get_embedding_api_keys()
    if not keys:
        log.warning("v2 batch: ключи API не найдены!")
        return [None] * len(texts)

    log.info("v2 batch: запуск миграции (%d текстов) используя %d ключей", len(texts), len(keys))

    global _key_rr_idx
    out: list[list[float] | None] = []
    
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        chunk_result: list[list[float] | None] | None = None
        
        # Попытки для одного и того же чанка
        for sweep in range(3):
            min_retry_after: float | None = None
            
            # Перебираем ключи по кругу
            for offset in range(len(keys)):
                idx = (_key_rr_idx + offset) % len(keys)
                current_key = keys[idx]
                
                status, result, retry_after = _embed_v2_rest_batch(
                    chunk, "RETRIEVAL_DOCUMENT", dim, current_key
                )
                
                # ВАЖНО: Всегда сдвигаем указатель на следующий ключ, 
                # чтобы следующий запрос (или повтор) шел с другого ключа.
                _key_rr_idx = (idx + 1) % len(keys)
                
                if status == 200 and result is not None:
                    chunk_result = result
                    break
                    
                if status == 429:
                    log.warning("v2 batch: ключ #%d словил 429 (TPM/RPM). Ждать: %s", idx, retry_after)
                    if retry_after:
                        min_retry_after = retry_after if min_retry_after is None else min(min_retry_after, retry_after)
                    continue
                
                log.warning("v2 batch: ключ #%d вернул ошибку %d", idx, status)
            
            if chunk_result is not None:
                break
            
            # Если все ключи в ауте, спим столько, сколько просил самый «быстрый» из них
            wait = (min_retry_after + 1.0) if min_retry_after else 30.0
            log.warning("v2 batch: все ключи перегружены (свип %d), спим %.1fs", sweep + 1, wait)
            _time.sleep(wait)
            
        if chunk_result is None:
            log.error("v2 batch: не удалось обработать чанк %d-%d после всех попыток", start, start + len(chunk))
            chunk_result = [None] * len(chunk)
            
        out.extend(chunk_result)
        
        # ПРОФИЛАКТИЧЕСКИЙ СОН (Самое важное!)
        # Даже если запрос успешен, делаем паузу, чтобы не выжечь TPM всех ключей разом.
        if start + chunk_size < len(texts):
            # Если ключей много, можно спать меньше (1с), если мало — больше (3с)
            sleep_time = 3.0 if len(keys) < 3 else 1.0
            _time.sleep(sleep_time)

    return out

# ... (остальные функции: search_and_format и т.д.) ...
