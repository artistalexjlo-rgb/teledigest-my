"""extraction.py — Python-порт apps_script/Code.gs::runMining_.

Читает sample-файлы из samples_dir (созданы daily_samples.dump_all_targets),
прогоняет каждый через Gemini API (free-tier ключи pool), извлекает JSON
patterns и складывает в SQLite таблицу extracted_patterns. Маркирует
обработанные файлы сайдкаром .processed чтобы не перечитывать.

Apps Script больше НЕ используется — Cloud-suspend сделал его inferно
ненадёжным. Эта функция всё делает в Python в нашем боте-контейнере.

ROTATION (consonant-slow по требованию юзера):
  Одна модель — запрос по всем ключам в круг.
  Между моделями — sleep 60s.
  Если у пары (key, model) RPD исчерпан или она была забанена 429 —
  скипаем эту пару до конца UTC-суток.
  Счётчики персистентны в SQLite (gemini_quota), переживают рестарт.

Поток:
1. Walk samples/{country}/*.txt → skip если .processed sidecar.
2. iterate model-key pairs (rotator), per file одно успешное обращение.
3. Parse {"patterns": [...]} → write to extracted_patterns (pending).
4. Touch sidecar {file}.processed.
5. embed_pump.py отдельным проходом подбирает pending и заливает в Qdrant.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from .config import get_config, log
from .extraction_db import (
    COLLECTION_STORIES,
    COLLECTION_WISDOM,
    _key_hash,
    init_extraction_tables,
    insert_extracted_pattern,
    quota_ban_today,
    quota_increment,
    quota_state,
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
    "- Прямая реклама чужого бизнеса (объявление салона, кафе, агентства, "
    'курсов и т.п. с телефоном для брони или призывом "приходите" / '
    '"звоните" / "запись") — это НЕ история и НЕ полезный факт. '
    "Пропускай такие посты, не превращай их в pattern.\n"
)

# (name, rpd_cap). RPM cap уже задан Google'ом на стороне сервера; мы не
# идём близко к нему благодаря inter-model sleep. Имена моделей —
# best-effort GA: проверяем через ListModels при первом запуске; если
# 404 — откатываем на -preview вариант.
# gemini-3.5-flash убрана 2026-05-24 — на ней extraction давал spam-ish
# результаты: пропускала прямую рекламу как "истории" + переписывала
# простые факты в блогерскую воду с морализаторством. Это новая модель,
# не calibrated под наш мухи/котлеты-фильтр. Возвращать только если
# Google сделает её более строгой на инструкции стиля.
_MINING_MODELS: list[tuple[str, int]] = [
    ("gemini-3.1-flash-lite", 500),
    ("gemini-2.5-flash", 20),
    ("gemini-2.5-flash-lite", 20),
]

# Sleep между моделями (после прохода всех ключей в текущей модели).
_INTER_MODEL_SLEEP_S = 60.0

# Sleep между файлами в рамках одной модели (consonant slow).
_INTER_FILE_PAUSE_S = 4.5

# Retry schedule per file (если все попытки на разных pairs провалились).
_RETRY_DELAYS_S = [5.0, 20.0, 60.0]

# Sidecar suffix для пометки обработанных файлов.
_PROCESSED_MARKER = ".processed"


def _doc_id(source_file_name: str, idx: int, collection: str) -> str:
    """Deterministic ID = sha1(file_name : idx : collection)[:24] —
    как в Apps Script saveToFirestore_."""
    seed = f"{source_file_name}:{idx}:{collection}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def iter_model_key_pairs(
    keys: list[str],
    sleep_fn=time.sleep,
    models: list[tuple[str, int]] | None = None,
) -> Iterator[tuple[str, str]]:
    """Бесконечный генератор пар (model_name, api_key) по правилу:

    "Одна модель — по всем ключам — потом sleep 60s — следующая модель".

    Пропускает пары которые упёрлись в RPD-кап или забанены 429.
    Выходит (StopIteration) если за полный оборот ни одной живой пары не
    нашлось — значит все RPD-баки на сегодня пусты, есть смысл подождать
    до завтра.

    Аргументы:
      keys: список api-ключей (порядок сохраняется).
      sleep_fn: для тестов — подменяемая sleep-функция.
      models: для тестов — переопределить _MINING_MODELS.
    """
    if not keys:
        log.error("extraction rotator: пустой список ключей — нечего ротировать")
        return
    mdls = models if models is not None else _MINING_MODELS
    hashed = [(k, _key_hash(k)) for k in keys]

    while True:
        any_used_this_round = False
        for model_name, rpd_cap in mdls:
            used_in_model = 0
            for api_key, kh in hashed:
                count, banned = quota_state(kh, model_name)
                if banned or count >= rpd_cap:
                    continue
                used_in_model += 1
                any_used_this_round = True
                yield model_name, api_key
            if used_in_model > 0:
                log.info(
                    "extraction rotator: модель %s отработала %d ключей, "
                    "sleep %.0fs до следующей",
                    model_name,
                    used_in_model,
                    _INTER_MODEL_SLEEP_S,
                )
                sleep_fn(_INTER_MODEL_SLEEP_S)
        if not any_used_this_round:
            log.warning(
                "extraction rotator: все (ключ, модель) пары на капе RPD — "
                "оборот пуст, выходим до сброса квоты (UTC-полночь)"
            )
            return


def _gemini_generate_json(
    content: str,
    model: str,
    api_key: str,
    timeout: int = 60,
) -> tuple[dict | None, int]:
    """One generate_content call в JSON-режиме.

    Returns (response_json, http_status). response_json = None если HTTP не
    200 или тело пустое; http_status позволяет вызывающему отличать 429
    (бан пары) от прочих ошибок (general retry).
    """
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
        return None, 0
    if resp.status_code != 200:
        log.warning(
            "extraction Gemini HTTP %d on %s: %s",
            resp.status_code,
            model,
            resp.text[:300],
        )
        return None, resp.status_code
    return resp.json(), 200


def _extract_patterns_from_response(api_resp: dict) -> list[dict]:
    """Из generate_content response достать JSON массив patterns."""
    cands = api_resp.get("candidates") or []
    if not cands or not cands[0].get("content"):
        return []
    parts = cands[0]["content"].get("parts") or []
    if not parts:
        return []
    raw = parts[0].get("text") or ""
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


def _persist_patterns(file_name: str, patterns: list[dict]) -> tuple[int, int]:
    """Save patterns to SQLite extracted_patterns. Returns (saved, attempted)."""
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
            did = _doc_id(file_name, idx, COLLECTION_WISDOM)
            try:
                insert_extracted_pattern(
                    id_=did,
                    collection_target=COLLECTION_WISDOM,
                    country=country,
                    title=title,
                    tag=tag,
                    routing=routing,
                    ai_lesson=ai_lesson,
                    human_story=None,
                    target_languages=None,
                    source_country_file=file_name,
                    source_country_file_idx=idx,
                )
                saved += 1
            except Exception as e:
                log.warning("extraction insert wisdom %s failed: %s", did, e)

        # stories (котлеты) — channel_only или both, и есть human_story
        if routing in ("both", "channel_only") and human_story:
            attempted += 1
            did = _doc_id(file_name, idx, COLLECTION_STORIES)
            try:
                insert_extracted_pattern(
                    id_=did,
                    collection_target=COLLECTION_STORIES,
                    country=country,
                    title=title,
                    tag=tag,
                    routing=routing,
                    ai_lesson=None,
                    human_story=human_story,
                    target_languages=target_langs or ["ru"],
                    source_country_file=file_name,
                    source_country_file_idx=idx,
                )
                saved += 1
            except Exception as e:
                log.warning("extraction insert story %s failed: %s", did, e)

    return saved, attempted


def process_file(
    file_path: Path,
    rotator: Iterator[tuple[str, str]],
) -> tuple[int, int, bool]:
    """Process one sample file. Returns (saved, attempted, exhausted).

    exhausted=True означает что ротатор кончился (StopIteration) пока этот
    файл пытался обработаться — значит весь pass надо остановить.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error("extraction: read failed for %s: %s", file_path, e)
        return 0, 0, False
    if not content.strip():
        return 0, 0, False

    api_resp: dict | None = None
    for attempt, wait_after in enumerate([0] + _RETRY_DELAYS_S):
        if wait_after:
            log.warning(
                "extraction: retry %d/%d after %.0fs for %s",
                attempt,
                len(_RETRY_DELAYS_S),
                wait_after,
                file_path.name,
            )
            time.sleep(wait_after)
        try:
            model, api_key = next(rotator)
        except StopIteration:
            log.warning(
                "extraction: ротатор пуст до обработки %s — pass завершён",
                file_path.name,
            )
            return 0, 0, True

        log.info("extraction: %s → %s (attempt %d)", file_path.name, model, attempt + 1)
        api_resp, status = _gemini_generate_json(content, model, api_key)

        # Счётчик инкрементим только при реальном использовании квоты.
        # 200, 4xx/5xx — Google считает попытки тоже, но 429 = превышение,
        # помечаем пару как exhausted.
        kh = _key_hash(api_key)
        quota_increment(kh, model)
        if status == 429:
            log.warning(
                "extraction: 429 на (%s, %s…) — пара забанена до UTC-полуночи",
                model,
                api_key[:6],
            )
            quota_ban_today(kh, model)

        if api_resp:
            break

    if api_resp is None:
        log.warning("extraction: %s — Gemini failed after retries", file_path.name)
        return 0, 0, False

    patterns = _extract_patterns_from_response(api_resp)
    if not patterns:
        log.info("extraction: %s — 0 patterns extracted", file_path.name)
        return 0, 0, False

    saved, attempted = _persist_patterns(file_path.name, patterns)
    log.info(
        "extraction: %s — patterns=%d saved=%d (of %d attempted)",
        file_path.name,
        len(patterns),
        saved,
        attempted,
    )
    return saved, attempted, False


def run_extraction_pass(
    samples_dir: Path | None = None,
    max_files: int | None = None,
    force_reprocess: bool = False,
) -> tuple[int, int, int]:
    """Walk samples_dir, process unprocessed files. Returns (files_processed,
    total_saved, total_attempted).

    Per-file:
      - sidecar `{name}.processed` создаётся если хотя бы один pattern
        извлечён (иначе оставляем непомеченным для ретрая).
    max_files = cap для предсказуемости (None = все).
    """
    init_extraction_tables()
    cfg = get_config()
    keys = list(cfg.gemini.api_keys) if cfg.gemini.api_keys else []
    if not keys and cfg.gemini.api_key:
        keys = [cfg.gemini.api_key]
    if not keys:
        log.error("extraction: ни один GEMINI ключ не настроен")
        return 0, 0, 0

    if samples_dir is None:
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

    log.info(
        "extraction: %d sample files queued, %d ключей в pool", len(files), len(keys)
    )

    rotator = iter_model_key_pairs(keys)
    files_processed = 0
    total_saved = 0
    total_attempted = 0

    for i, f in enumerate(files):
        if i > 0:
            time.sleep(_INTER_FILE_PAUSE_S)
        saved, attempted, exhausted = process_file(f, rotator)
        total_saved += saved
        total_attempted += attempted
        if attempted > 0:
            marker = f.with_suffix(f.suffix + _PROCESSED_MARKER)
            try:
                marker.write_text(
                    dt.datetime.now(dt.timezone.utc).isoformat(), encoding="utf-8"
                )
            except Exception as e:
                log.warning("extraction: marker write failed for %s: %s", f.name, e)
        files_processed += 1
        if exhausted:
            log.warning(
                "extraction: pass прерван — все квоты исчерпаны. Обработано %d/%d",
                files_processed,
                len(files),
            )
            break

    log.info(
        "extraction DONE: files_processed=%d total_saved=%d total_attempted=%d",
        files_processed,
        total_saved,
        total_attempted,
    )
    return files_processed, total_saved, total_attempted
