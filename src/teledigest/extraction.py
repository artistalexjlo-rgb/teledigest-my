"""extraction.py — Python-порт apps_script/Code.gs::runMining_.

Читает sample-файлы из samples_dir (созданы daily_samples.dump_all_targets),
прогоняет каждый через Gemini API (gemini-3.1-flash-lite-preview, free-tier
ключ), извлекает JSON patterns и складывает в SQLite таблицу
extracted_patterns. Маркирует обработанные файлы сайдкаром .processed
чтобы не перечитывать.

Apps Script больше НЕ используется — Cloud-suspend сделал его inferно
ненадёжным. Эта функция всё делает в Python в нашем боте-контейнере.

Поток:
1. Walk samples/{country}/*.txt
2. Skip если есть {file}.processed sidecar
3. Read content → systemPrompt → Gemini generate_content (JSON mode)
4. Parse {"patterns": [...]} response
5. Для каждого pattern:
   - routing in (both, assistant_only) → write to extracted_patterns
     с collection_target=wisdom_base
   - routing in (both, channel_only) AND human_story → write
     с collection_target=telegram_queue
6. Touch sidecar {file}.processed
7. embed_pump.py отдельным проходом подбирает pending и заливает в Qdrant
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .config import get_config, log
from .extraction_db import (
    COLLECTION_STORIES,
    COLLECTION_WISDOM,
    init_extraction_tables,
    insert_extracted_pattern,
)

# System-prompt — дословно из Apps Script Code.gs:280-316.
_SYSTEM_PROMPT = (
    "Ты — главный архитектор данных MultySpeak. "
    "Фильтрация и маршрутизация опыта из чатов.\n"
    'Преврати лог в JSON: {"patterns": [...]}.\n\n'
    "ОБЯЗАТЕЛЬНЫЕ ПОЛЯ КАЖДОГО ЭЛЕМЕНТА:\n"
    "- title: на английском (универсальный ключ).\n"
    "- country: ISO 3166-1 alpha-2 в нижнем регистре (br, id, lk, vn, tr, "
    "и т.д. по стандарту). Если pattern универсальный и не привязан к одной "
    'стране — укажи "any".\n'
    "- routing: одна из строк. ВЫБИРАЙ ПО ПРАВИЛАМ:\n"
    '    * "both" — есть И живая история/байка/контекст, И полезный сухой '
    "факт (цифры, цены, инструкция, ссылка). ЭТО ДЕФОЛТ — большинство "
    'интересных кейсов сюда. Если сомневаешься — ставь "both".\n'
    '    * "assistant_only" — голый сухой факт без живой истории. Например: '
    "контакт чиновника, точная цена, шаг бюрократической процедуры, "
    "название документа. ИИ-помощнику пригодится, но публиковать в канал "
    "скучно.\n"
    '    * "channel_only" — живая байка/мем/локальный колорит без '
    "извлекаемого факта.\n"
    "- tag: на английском (Finance, Safety, Bureaucracy, Travel и т.п.).\n"
    "- target_languages: массив ISO 639-1 кодов языков на которые история "
    'имеет смысл переводиться. По умолчанию ["ru"]. Универсальные '
    'истории — перечисли все уместные: например ["ru","en","es","pt"]. '
    'Только если routing == "both" или "channel_only".\n'
    "- human_story: ИСТОРИИ И ХАКИ ДЛЯ КАНАЛА. СТРОГО НА РУССКОМ ЯЗЫКЕ. "
    "Пиши сочно, живо, с лёгкой иронией. Сделай это интересной историей "
    'для канала. Только если routing == "both" или "channel_only".\n'
    "- ai_lesson: ИНСТРУКЦИЯ ДЛЯ ИИ-ПОМОЩНИКА. СТРОГО НА АНГЛИЙСКОМ ЯЗЫКЕ. "
    "Сухие, точные факты и цифры без эмоций. Только если routing == "
    '"both" или "assistant_only".\n\n'
    "ФИЛЬТРАЦИЯ:\n"
    "- Игнорируй слухи, пустой трёп, спам и сообщения про спамеров.\n"
    "- Игнорируй pattern если в логе нет конкретики — не выдумывай.\n"
)

# Соответствует MODELS из Apps Script: ротация моделей под free-tier RPM.
# Каждая запись: (имя, RPM cap).
_MINING_MODELS = [
    ("gemini-3.1-flash-lite-preview", 15),
    ("gemini-2.5-flash-lite", 10),
    ("gemini-2.5-flash", 5),
]

# Apps Script INTER_FILE_PAUSE_MS = 4500
_INTER_FILE_PAUSE_S = 4.5

# Retry schedule (Apps Script GEMINI_RETRY_DELAYS_MS).
_RETRY_DELAYS_S = [5.0, 20.0, 60.0]

# Sidecar suffix для пометки обработанных файлов.
_PROCESSED_MARKER = ".processed"


def _doc_id(source_file_name: str, idx: int, collection: str) -> str:
    """Deterministic ID = sha1(file_name : idx : collection)[:24] —
    как в Apps Script saveToFirestore_."""
    seed = f"{source_file_name}:{idx}:{collection}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def _pick_model(call_times: dict[str, list[float]]) -> str:
    """Round-robin модель с наименьшей текущей load (как pickModelWithCapacity_
    в Apps Script). Если все на капе — ждём."""
    while True:
        now = time.time()
        best: tuple[str, int] | None = None
        best_load = float("inf")
        soonest_free_at = float("inf")
        for name, rpm in _MINING_MODELS:
            ts = [t for t in call_times.get(name, []) if now - t < 60.0]
            call_times[name] = ts
            if len(ts) < rpm:
                load = len(ts) / rpm
                if load < best_load:
                    best_load = load
                    best = (name, rpm)
            elif ts:
                soonest_free_at = min(soonest_free_at, ts[0] + 60.0)
        if best:
            call_times[best[0]].append(now)
            return best[0]
        wait = max(0.0, soonest_free_at - now) + 0.1
        log.info("extraction: all mining models at RPM cap, sleeping %.1fs", wait)
        time.sleep(wait)


def _gemini_generate_json(
    content: str,
    model: str,
    api_key: str,
    timeout: int = 60,
) -> dict | None:
    """One generate_content call в JSON-режиме. None если HTTP не 200 или
    тело пустое."""
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": f"Текст лога:\n{content}"}]}],
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "generationConfig": {"responseMimeType": "application/json"},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except Exception as e:
        log.warning("extraction Gemini call exception: %s", e)
        return None
    if resp.status_code != 200:
        log.warning(
            "extraction Gemini HTTP %d on %s: %s",
            resp.status_code,
            model,
            resp.text[:300],
        )
        return None
    return resp.json()


def _extract_patterns_from_response(api_resp: dict) -> list[dict]:
    """Из generate_content response достать JSON массив patterns."""
    cands = api_resp.get("candidates") or []
    if not cands or not cands[0].get("content"):
        return []
    parts = cands[0]["content"].get("parts") or []
    if not parts:
        return []
    raw = parts[0].get("text") or ""
    # Strip ```json fences если есть.
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        log.warning("extraction: JSON parse failed: %s\nText: %s", e, clean[:300])
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        patterns = data.get("patterns")
        if isinstance(patterns, list):
            return patterns
    return []


def process_file(
    file_path: Path,
    api_key: str,
    call_times: dict[str, list[float]],
) -> tuple[int, int]:
    """Process one sample file. Returns (saved, attempted).

    attempted = сколько patterns Gemini вернул. saved = сколько реально
    записали в pending (с дедупом по deterministic id)."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error("extraction: read failed for %s: %s", file_path, e)
        return 0, 0
    if not content.strip():
        return 0, 0

    # Retry loop через несколько моделей с backoff
    api_resp = None
    for attempt, wait_after in enumerate([0] + _RETRY_DELAYS_S):
        if wait_after:
            log.warning(
                "extraction: retry %d/%d after %.0fs",
                attempt,
                len(_RETRY_DELAYS_S),
                wait_after,
            )
            time.sleep(wait_after)
        model = _pick_model(call_times)
        log.info("extraction: %s → %s (attempt %d)", file_path.name, model, attempt + 1)
        api_resp = _gemini_generate_json(content, model, api_key)
        if api_resp:
            break

    if api_resp is None:
        log.warning("extraction: %s — Gemini failed after retries", file_path.name)
        return 0, 0

    patterns = _extract_patterns_from_response(api_resp)
    if not patterns:
        log.info("extraction: %s — 0 patterns extracted", file_path.name)
        return 0, 0

    saved = 0
    attempted = 0
    for idx, p in enumerate(patterns):
        if not isinstance(p, dict):
            continue
        routing = (p.get("routing") or "both").strip().lower()
        country = (p.get("country") or "unknown").strip().lower()
        title = (p.get("title") or "Untitled").strip()
        tag = (p.get("tag") or "General").strip()
        target_langs = (
            p.get("target_languages")
            if isinstance(p.get("target_languages"), list)
            else None
        )
        ai_lesson = (p.get("ai_lesson") or "").strip() or None
        human_story = (p.get("human_story") or "").strip() or None

        # wisdom (мухи) — assistant_only или both
        if routing in ("both", "assistant_only") and ai_lesson:
            attempted += 1
            did = _doc_id(file_path.name, idx, COLLECTION_WISDOM)
            try:
                insert_extracted_pattern(
                    id_=did,
                    collection_target=COLLECTION_WISDOM,
                    country=country,
                    title=title,
                    tag=tag,
                    routing=routing,
                    ai_lesson=ai_lesson,
                    human_story=None,  # wisdom — без человеческой истории
                    target_languages=None,
                    source_country_file=file_path.name,
                    source_country_file_idx=idx,
                )
                saved += 1
            except Exception as e:
                log.warning("extraction insert wisdom %s failed: %s", did, e)

        # stories (котлеты) — channel_only или both, и есть human_story
        if routing in ("both", "channel_only") and human_story:
            attempted += 1
            did = _doc_id(file_path.name, idx, COLLECTION_STORIES)
            try:
                insert_extracted_pattern(
                    id_=did,
                    collection_target=COLLECTION_STORIES,
                    country=country,
                    title=title,
                    tag=tag,
                    routing=routing,
                    ai_lesson=None,  # story — без сухого факта
                    human_story=human_story,
                    target_languages=target_langs or ["ru"],
                    source_country_file=file_path.name,
                    source_country_file_idx=idx,
                )
                saved += 1
            except Exception as e:
                log.warning("extraction insert story %s failed: %s", did, e)

    log.info(
        "extraction: %s — patterns=%d saved=%d (of %d attempted)",
        file_path.name,
        len(patterns),
        saved,
        attempted,
    )
    return saved, attempted


def run_extraction_pass(
    samples_dir: Path | None = None,
    max_files: int | None = None,
    force_reprocess: bool = False,
) -> tuple[int, int, int]:
    """Walk samples_dir, process unprocessed files. Returns (files_processed,
    total_saved, total_attempted).

    Markers: для каждого обработанного файла создаётся sidecar
    `{name}.processed` чтобы не перечитывать.

    max_files = cap на сколько файлов обработать в один проход (для
    cron-job предсказуемости). None = все.
    """
    init_extraction_tables()
    cfg = get_config()
    api_key = cfg.gemini.api_key or ""
    if not api_key:
        log.error("extraction: cfg.gemini.api_key пустой — Gemini API key required")
        return 0, 0, 0

    if samples_dir is None:
        # Default из daily_samples (тот же samples_dir, обычно
        # /home/teledigest/data/samples)
        from .daily_samples import get_samples_dir

        samples_dir = get_samples_dir()

    if not samples_dir.exists():
        log.warning("extraction: samples_dir %s does not exist", samples_dir)
        return 0, 0, 0

    files: list[Path] = []
    for country_dir in samples_dir.iterdir():
        if not country_dir.is_dir():
            continue
        for f in country_dir.iterdir():
            if not f.is_file() or f.suffix != ".txt":
                continue
            marker = f.with_suffix(f.suffix + _PROCESSED_MARKER)
            if marker.exists() and not force_reprocess:
                continue
            files.append(f)

    files.sort(key=lambda p: p.stat().st_mtime)
    if max_files:
        files = files[:max_files]

    log.info("extraction: %d sample files queued", len(files))

    call_times: dict[str, list[float]] = {}
    files_processed = 0
    total_saved = 0
    total_attempted = 0

    for i, f in enumerate(files):
        if i > 0:
            time.sleep(_INTER_FILE_PAUSE_S)
        saved, attempted = process_file(f, api_key, call_times)
        total_saved += saved
        total_attempted += attempted
        if attempted > 0:
            # Marker даже если saved < attempted — те что не сохранились
            # либо дубли (idempotency), либо bad-pattern. Файл считаем
            # «processed», чтобы не зацикливаться.
            marker = f.with_suffix(f.suffix + _PROCESSED_MARKER)
            try:
                marker.write_text(
                    dt.datetime.now(dt.timezone.utc).isoformat(), encoding="utf-8"
                )
            except Exception as e:
                log.warning("extraction: marker write failed for %s: %s", f.name, e)
        files_processed += 1

    log.info(
        "extraction DONE: files_processed=%d total_saved=%d total_attempted=%d",
        files_processed,
        total_saved,
        total_attempted,
    )
    return files_processed, total_saved, total_attempted
