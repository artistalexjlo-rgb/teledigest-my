"""extraction.py — сообщения чата → мухи (`extracted_patterns`), через МОЗГ.

Читает sample-файлы из samples_dir (созданы daily_samples.dump_all_targets), прогоняет
каждый через Gemini, извлекает JSON patterns и складывает в SQLite extracted_patterns.
Обработанные файлы помечает сайдкаром.

⭐ 15.09: ключи, темп, 429/503, парс-фейлы и повторы — ЦЕЛИКОМ в `keybroker.py` (один мозг
на бота и тракт, общая база квоты). Свой ротатор (`iter_model_key_pairs`), свой учёт
`gemini_quota` и своя лестница ретраев отсюда СНЕСЕНЫ: два учёта давали два разных ответа
на «ключи кончились», а свой 429 банил ключ на сутки после 2–3 вызовов (7 из 12 ключей
за первые полтора часа прохода, замер 15.09). Экстрактор зовётся у мозга именем `extract`
с ролью `primary` (полный RPD).

Поток:
1. Walk samples/{country}/*.txt → skip, если есть сайдкар `.processed` или `.empty`.
2. Файл режется на куски по `CHUNK_CHARS` (по границам строк): ответ на 900 строк чата
   не влезает ни в 60 с, ни в потолок выходных токенов — 15.09 файлы cn/id от 140 КБ
   падали по таймауту четырежды подряд и не обрабатывались никогда.
3. Каждый кусок → `keybroker.call` по списку MODELS: следующая модель, только когда у
   предыдущей нет живых ключей (`any_alive`) или вызов вернул None.
4. Parse {"patterns": [...]} → extracted_patterns (pending), индекс сквозной по файлу.
5. Сайдкар: `.processed` — извлечено хоть что-то; `.empty` — все куски ответили, паттернов
   ноль (пустой день чата, 241 файл < 300 байт крутились в очереди вечно до 15.09);
   ничего — хоть один кусок провалился, файл вернётся в следующий проход.
6. embed_pump.py отдельным проходом подбирает pending и заливает в Qdrant.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any

from . import keybroker
from .config import log
from .extraction_db import (
    COLLECTION_STORIES,
    COLLECTION_WISDOM,
    init_extraction_tables,
    insert_extracted_pattern,
)
from .ipv4_only import force_ipv4

# ⛔ IPv6 к Google с этого VPS — чёрная дыра: замер 19.08 дал IPv4 0.18 с против 8 с молчания
# по IPv6, а в логе 15 повисаний по 60 с. Ставим фильтр ДО первого запроса. Правило одно, живёт
# в `ipv4_only`; мозг ставит такой же фильтр у себя — двойной безвреден.
force_ipv4()

# System-prompt — дословно из Apps Script Code.gs:280-316.
_SYSTEM_PROMPT = (
    "Ты — главный архитектор данных MultySpeak. "
    "Фильтрация и маршрутизация опыта из чатов.\n"
    'Преврати лог в JSON: {"patterns": [...]}.\n\n'
    "ОБЯЗАТЕЛЬНЫЕ ПОЛЯ КАЖДОГО ЭЛЕМЕНТА:\n"
    "- title: на английском (универсальный ключ).\n"
    "- country: ISO 3166-1 alpha-2 в нижнем регистре (br, id, lk, vn, tr, "
    "и т.д. по стандарту). Если pattern касается НЕСКОЛЬКИХ стран — перечисли "
    'их через запятую («de, ru»), а не выбирай одну и не ставь "any". '
    '"any" — только для действительно универсального совета, верного в любой '
    "стране (например: как вести себя в аэропорту вообще).\n"
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
    "- Частные объявления о купле-продаже (продаёт/покупает вещь, технику, "
    "мебель, транспорт, самокат, телефон и т.п. — с ценой, состоянием, "
    '"контакты по запросу", "торг") — это НЕ история и НЕ факт. Пропускай.\n'
    "- Сбор денег и просьбы о финансовой помощи: переводы, реквизиты, "
    'донаты, "помогите такому-то, он пострадал/заболел", "любая сумма '
    'важна", банковские карты для перевода — ПРОПУСКАЙ ВСЕГДА. Это спам '
    "или мошенничество (эмоция + срочность + перевод незнакомцу), "
    "ни в коем случае не история и не факт.\n"
    "- Личные транзакционные объявления: человек сдаёт/ищет КОНКРЕТНУЮ "
    "квартиру/комнату для себя, предлагает/ищет работу или попутчиков, с "
    "контактами — это объявление, пропускай. ВАЖНО: советы о ПРОЦЕССЕ "
    "(какими площадками/агрегаторами искать жильё, как устроена аренда, "
    "типичные цены и подводные камни) — это НОРМАЛЬНАЯ история и факт, "
    "СОХРАНЯЙ их. Режь только конкретное объявление с контактами, не "
    "общий совет.\n"
    "- Чистый ВОПРОС или просьба о совете БЕЗ собственного опыта/факта "
    '("планирую поездку, есть ли у кого опыт?", "подскажите", "реально ли", '
    '"кто сталкивался?", "интересует, можно ли") — это НЕ история и НЕ факт, '
    "это человек спрашивает. ПРОПУСКАЙ. Извлекай pattern только если в "
    "сообщении есть ОТВЕТ/опыт/конкретика (что сделал, что получилось, цифры, "
    "шаги) — тогда бери факт, но сам вопрос в канал не публикуй.\n"
    "- Само-реклама и предложение услуг от ЧАСТНОГО лица: «я [имя], "
    "эксперт/представитель/менеджер [сервиса, напр. TravelAsk], помогаю с "
    "переездом/визой/лечением/жильём», «обращайтесь», «пишите мне», «помогу "
    "под ключ», представление себя как специалиста/посредника с призывом "
    "написать — это РЕКЛАМА, даже без телефона и без названия фирмы. "
    "ПРОПУСКАЙ. Это не история и не факт.\n"
    "- Промо чужого ПЛАТНОГО сервиса-посредника: называет конкретный сервис/"
    "фирму (напр. «Консул Прайм», «сервис X») и подаёт его как платный способ "
    "ускорить/обойти очередь/оформить («ускоряет до месяца за ~130 у.е.», "
    "«платный слот», «за деньги сделают быстрее», «купи слот») — нам это выгоды "
    "не приносит и не должно лететь в канал/Дзен. Если совет В ОСНОВНОМ про этот "
    "платный сервис — ПРОПУСКАЙ pattern. Если это полезный совет, где платный "
    "сервис лишь мельком — сохрани совет, но в human_story и ai_lesson НЕ "
    "упоминай название сервиса и его цену (можно нейтрально: «есть платный "
    "способ ускорить», без имён и сумм). ВАЖНО: обычные цены товаров, госпошлин, "
    "аренды, билетов — это НОРМАЛЬНЫЙ факт, режь ТОЛЬКО промо платного "
    "посредника, а не любую цифру с ценой.\n"
    "- Болтовня без переиспользуемого факта: благодарности чату/группе "
    '("спасибо, узнал много полезного"), личные эмоции и мнения, споры ни о '
    "чём, вайб-рассуждения («тут так свободно дышится»), хвастовство, оффтоп, "
    "анонсы своих треков/постов, субъективные впечатления БЕЗ конкретного "
    "совета/цифр/инструкции — это НЕ история для канала. ПРОПУСКАЙ. Оставляй "
    "только то, из чего читатель вынесет переиспользуемую пользу.\n"
    "\n"
    "СТИЛЬ human_story:\n"
    "- НЕ используй шаблонные клише и не повторяй одни и те же обороты из "
    'поста в пост ("тот ещё квест", "не для слабонервных", "та ещё '
    'история"). Каждую историю пиши свежо, своими словами.\n'
)

# Порядок моделей. Лимиты (RPM/RPD) и учёт — в keybroker.LIMITS, здесь только очередь:
# `3.1-flash-lite` закрывает всё (юзер 15.09), остальные — хвост на случай выбранного бюджета.
# gemini-3.5-flash убрана 2026-05-24 — на ней extraction давал spam-ish результаты:
# пропускала прямую рекламу как "истории" + переписывала простые факты в блогерскую воду.
MODELS: list[str] = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
CONSUMER = "extract"  # имя в закрытом реестре мозга (keybroker.CAPS)
ROLE = "primary"  # полный RPD: экстрактор — основной потребитель ключей ночью
# Таймаут одного HTTP-вызова. 60 с не хватало на ответ по большому куску; при CHUNK_CHARS
# ответ короткий, 180 — запас на медленную модель, не на толстый вход.
CALL_TIMEOUT_S = 180
# Порог куска в СИМВОЛАХ (кириллица в UTF-8 — 2 байта на символ: 40 000 симв. ≈ 60–80 КБ
# файла, ≈ 15–20 тыс. токенов входа). Замер 15.09: 140 КБ (≈70 тыс. симв.) — уже таймаут.
CHUNK_CHARS = 40_000

# Сайдкары. `.processed` — извлечено; `.empty` — ответ был, паттернов ноль. Оба = «не брать».
_PROCESSED_MARKER = ".processed"
_EMPTY_MARKER = ".empty"


def _doc_id(source_file_name: str, idx: int, collection: str) -> str:
    """Deterministic ID = sha1(file_name : idx : collection)[:24] —
    как в Apps Script saveToFirestore_."""
    seed = f"{source_file_name}:{idx}:{collection}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def split_content(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Порезать лог на куски не длиннее `limit` символов по границам строк.

    Одна строка длиннее лимита идёт своим куском целиком — резать сообщение пополам
    хуже, чем дать модели один длинный кусок. Пустой текст → [].
    """
    if not text.strip():
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if cur and size + len(line) > limit:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _patterns_from(resp: Any) -> list[dict]:
    """Из ответа мозга (уже распарсенный JSON) достать список patterns."""
    if isinstance(resp, list):
        return [p for p in resp if isinstance(p, dict)]
    if isinstance(resp, dict):
        patterns = resp.get("patterns")
        if isinstance(patterns, list):
            return [p for p in patterns if isinstance(p, dict)]
    return []


def ask(chunk: str, models: list[str] | None = None) -> list[dict] | None:
    """Один кусок лога → список patterns, или None, если ни одна модель не ответила.

    Модели по порядку; к следующей — только если у текущей нет живых ключей или мозг
    вернул None (бюджет выбран / сдался). Пустой список — это ОТВЕТ (паттернов нет), а не
    провал: дальше по списку не идём.
    """
    for model in models or MODELS:
        if not keybroker.any_alive(model, ROLE):
            continue
        resp = keybroker.call(
            f"Текст лога:\n{chunk}",
            _SYSTEM_PROMPT,
            CONSUMER,
            model=model,
            role=ROLE,
            timeout=CALL_TIMEOUT_S,
            salvage=("patterns", "title"),
        )
        if resp is not None:
            return _patterns_from(resp)
    return None


# Guard: ai_lesson, в котором дешёвая модель пересказала ВОПРОС/ПУСТОТУ/ЛИСТИНГ
# вместо извлечения факта. flash-lite на чистом вопросе не молчит (abstention —
# слабое место мелкой модели), а выдаёт "Inquiry about X" / "User is asking…" —
# это НЕ факт. Эмбедить нельзя: жжёт дефицитную квоту + засоряет retrieval.
# Проверено на проде: ~12% wisdom такие. Промптом не лечится → ловим детерминированно.
_JUNK_AI_LESSON_RE = re.compile(
    r"\b(?:"
    r"User (?:is asking|is looking (?:for|to)|wants to know|needs to know|is seeking|"
    r"is inquiring|is requesting|asks|inquired|wants information)|"
    r"A user (?:is asking|asks|wants|is looking|inquired)|"
    r"Inquir(?:y|ies) (?:about|is|are)|"  # только сущ.-наррация (не «inquire/for inquiries regarding» = совет)
    r"Clarification (?:needed|is needed)|"
    r"not (?:explicitly )?provided in the log|is not provided in the log|"
    r"not specified in the log|"
    r"A request for assistance|request for assistance in|"
    r"is available for rent|(?:room|apartment|flat) is available|"
    r"consult with .{0,40}Telegram|via their Telegram channel|"
    r"should be researched"
    r")\b",
    re.IGNORECASE,
)

# Второй слой — «опенер-наррации»: ai_lesson НАЧИНАЕТСЯ с описания запроса/темы, а не
# факта («Information on purchasing…», «Provide instructions…», «Seeking…»). Якорь на
# начало строки, чтобы НЕ задеть факты, где эти слова стоят в середине.
_JUNK_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"Information (?:on|about|regarding)|"
    r"Provide (?:information|instructions|details|guidance|an overview)|"
    r"Details (?:on|about|regarding)|"
    r"Inquir(?:y|ies|ing)\b|"
    r"Seeking\b|Looking for\b|Request(?:ing)? (?:for|information)|"
    r"Question(?:s)? (?:about|regarding|on)|"
    r"(?:The |A )?[Uu]ser (?:is|wants|needs|seeks|asks)|"
    r"Guidance (?:on|is)|Advice (?:is )?(?:sought|requested)"
    r")",
)


def is_junk_ai_lesson(text: str | None) -> bool:
    """True, если ai_lesson — пересказанный моделью вопрос/пустота/листинг, а НЕ
    извлечённый факт. Детерминированный guard у источника (без LLM).
    Два слоя: фразы-маркеры где угодно + опенер-наррации в начале строки."""
    if not text:
        return False
    return bool(_JUNK_AI_LESSON_RE.search(text) or _JUNK_OPENER_RE.match(text))


def _persist_patterns(
    file_name: str, patterns: list[dict], idx_offset: int = 0
) -> tuple[int, int]:
    """Save patterns to SQLite extracted_patterns. Returns (saved, attempted).

    `idx_offset` — смещение индекса для кусков одного файла: id паттерна = sha1(файл:idx),
    и второй кусок без смещения перетёр бы (INSERT OR IGNORE — молча потерял) первый.
    """
    saved = 0
    attempted = 0
    for i, p in enumerate(patterns):
        idx = idx_offset + i
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

        # wisdom (мухи) — assistant_only или both; junk-guard режет пересказы
        # вопросов/пустоты ("Inquiry about…", "User is asking…") у источника.
        if (
            routing in ("both", "assistant_only")
            and ai_lesson
            and not is_junk_ai_lesson(ai_lesson)
        ):
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


def process_file(file_path: Path) -> tuple[int, int, bool]:
    """Один файл: куски → мозг → база. Returns (saved, attempted, ok).

    ok=False — хоть один кусок остался без ответа: сайдкар не ставится, файл вернётся в
    следующий проход. Извлечённое из удачных кусков при этом уже в базе (id детерминирован,
    повтор их не задвоит).
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error("extraction: read failed for %s: %s", file_path, e)
        return 0, 0, False
    chunks = split_content(content, CHUNK_CHARS)
    if not chunks:
        return 0, 0, True

    saved = attempted = 0
    idx_offset = 0
    ok = True
    for n, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            log.info("extraction: %s — кусок %d/%d", file_path.name, n, len(chunks))
        patterns = ask(chunk)
        if patterns is None:
            log.warning(
                "extraction: %s — кусок %d/%d без ответа",
                file_path.name,
                n,
                len(chunks),
            )
            ok = False
            # смещение держим стабильным: у куска — своё окно индексов, независимо от исхода
            idx_offset += _IDX_STRIDE
            continue
        if patterns:
            s, a = _persist_patterns(file_path.name, patterns, idx_offset)
            saved += s
            attempted += a
        idx_offset += _IDX_STRIDE
    if not ok:
        return saved, attempted, False
    if attempted == 0:
        log.info("extraction: %s — 0 patterns extracted", file_path.name)
    else:
        log.info(
            "extraction: %s — saved=%d (of %d attempted)",
            file_path.name,
            saved,
            attempted,
        )
    return saved, attempted, True


# Окно индексов на кусок: id = sha1(файл:idx). Смещение фиксированное, а не «сколько
# вернул прошлый кусок»: иначе повторный прогон после провала одного куска сдвинул бы
# индексы следующих и задвоил их мухи под новыми id.
_IDX_STRIDE = 1000


def _mark(file_path: Path, marker: str) -> None:
    try:
        file_path.with_suffix(file_path.suffix + marker).write_text(
            dt.datetime.now(dt.timezone.utc).isoformat(), encoding="utf-8"
        )
    except Exception as e:
        log.warning("extraction: marker write failed for %s: %s", file_path.name, e)


def _is_done(f: Path) -> bool:
    return any(
        f.with_suffix(f.suffix + m).exists() for m in (_PROCESSED_MARKER, _EMPTY_MARKER)
    )


def _any_model_alive() -> bool:
    return any(keybroker.any_alive(m, ROLE) for m in MODELS)


def run_extraction_pass(
    samples_dir: Path | None = None,
    max_files: int | None = None,
    force_reprocess: bool = False,
) -> tuple[int, int, int]:
    """Walk samples_dir, process unprocessed files. Returns (files_processed,
    total_saved, total_attempted).

    Per-file сайдкар: `.processed` — извлечено хоть что-то; `.empty` — паттернов ноль при
    полном ответе; ничего — провал, файл вернётся. Проход останавливается, когда у мозга
    не осталось живых ключей ни на одной модели (до PT-полуночи ждать нечего).
    max_files = cap для предсказуемости (None = все).
    """
    init_extraction_tables()
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
            if _is_done(f) and not force_reprocess:
                continue
            files.append(f)

    files.sort(key=lambda p: p.stat().st_mtime)
    if max_files:
        files = files[:max_files]

    log.info("extraction: %d sample files queued (мозг: %s)", len(files), CONSUMER)

    files_processed = 0
    total_saved = 0
    total_attempted = 0

    for f in files:
        if not _any_model_alive():
            log.warning(
                "extraction: pass прерван — у мозга нет живых ключей. Обработано %d/%d",
                files_processed,
                len(files),
            )
            break
        saved, attempted, ok = process_file(f)
        total_saved += saved
        total_attempted += attempted
        if ok:
            _mark(f, _PROCESSED_MARKER if attempted > 0 else _EMPTY_MARKER)
        files_processed += 1

    log.info(
        "extraction DONE: files_processed=%d total_saved=%d total_attempted=%d",
        files_processed,
        total_saved,
        total_attempted,
    )
    return files_processed, total_saved, total_attempted
