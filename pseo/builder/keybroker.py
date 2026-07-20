"""keybroker.py — ЕДИНЫЙ МОЗГ ключей Gemini (центр).

Все потребители (копии билдера сейчас, экстрактор — потом) берут ключ ТОЛЬКО через
acquire() и репортят через report(). Никакого пейсинга в памяти потребителя — ОДИН clock
на ключ, общий и персистентный (SQLite, атомарно BEGIN IMMEDIATE). Три щупальца физически
не могут ударить ключ чаще его шага — бронь общая. Учёт каждого запроса пишется в
request_log для статистики.

Роли:
  primary    — экстрактор (нужда ~22/ключ/ночь, мерено): cap = полный RPD.
  background — билдер/раннеры (съедят сколько дадим): cap = RPD - RESERVE (резерв primary).

Изоляция: своя broker.db, чтобы не драться за лок прод-БД (messages_fts.db пишет бот).
TODO(wire-real): подмешивать в usage расход экстрактора из gemini_quota, чтобы RPD был
общим И с ботом (сейчас broker самодостаточен — для короткой ноги builder-only этого хватает).
"""

import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

# ТОЛЬКО IPv4 — дословно из build.py. У VPS IPv6 к generativelanguage = чёрная дыра (коннект
# виснет). Без этого urllib идёт по дефолту ОС (IPv6 первым) → HTTP зависает, call не доходит
# до report. Фильтруем getaddrinfo до AF_INET, чтоб urllib физически не лез в IPv6.
_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [
    r for r in _orig_gai(*a, **k) if r[0] == socket.AF_INET
]

DB = os.environ.get("KB_DB", "/root/pseo_builder/keybroker.db")
PT = ZoneInfo("America/Los_Angeles")

# ── ВСЕ константы — из ведущего дока canon_gemini_key_algorithm.md или проверенного факта.
#    НЕ из build.py (недоверенный код). Неизвестное — НЕ выдумываем (rpm:None → консерватив).
#
# Лимиты free-tier ПО МОДЕЛИ (per-project = per-key, [[fact_gemini_projects_separate]]).
# ВСЕ проверены по скрину AI Studio 2026-07-14 (колонки RPM / TPM / RPD). Только text-out;
# 0/0/0 (Pro, image/Nano Banana) недоступны на тарифе, TTS (3/10K/10) — аудио, не тащим.
LIMITS = {
    "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500},  # основная рабочая
    "gemini-2.5-flash-lite": {"rpm": 10, "tpm": 250000, "rpd": 20},
    "gemini-2.5-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3.5-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
    },  # ⚠ extraction её выкинул (spam-ish), но лимит есть
}
DEFAULT_LIMIT = {
    "rpm": 5,
    "tpm": 250000,
    "rpd": 20,
}  # неизвестная модель — по самому жёсткому

# 429-эскалация — ДОСЛОВНО из эталона gemini_brain.py (post-7f74f7c):
COOLDOWN_FIRST = (
    300.0  # _EMBED_COOLDOWN_S: 1-й 429 = транзиент → 5мин (НЕ убивать ключ)
)
COOLDOWN_REPEAT = 1800.0  # _EMBED_COOLDOWN_REPEAT_S: повторный 429 (после cooldown) → 30мин, НЕ дневной бан
# Вина vs среда: 429 на ДРУГОМ ключе за последние N сек = волна общего источника
# (факт 2026-07-20: коррелированные волны 12×429 при пустых квотах, request_log) —
# текущий ключ НЕ наказываем, иначе внешний чих прожигает весь пул в cooldown
# (самострел). Окно 30с: волна в логах — кластер секунд, между волнами — минуты.
ENV429_WINDOW = float(os.environ.get("KB_ENV429_WINDOW", "30"))
# ОЧЕРЕДЬ ПУЛА (канон юзера 2026-07-20: раннеры снесены — очередь держит МОЗГ).
# Такт НА ВЫДАЧУ, не на полёт: мозг отдаёт ключи по одному и не чаще раза в GRANT_STEP
# на весь пул; выдал — следующий подходит через такт, а полёт (HTTP) идёт сам и никого
# не держит (юзер: «рот получил ключ и пошёл — сосок ему не нужен»; длинный перевод
# больше не блокирует пул). Частота запусков с IP ограничена тактом железно:
# 3с = ≤20 запусков/мин. Число — ручка юзера (точного по-IP порога НЕ существует,
# см. факт про общее окно 429 ниже в report()).
# В отличие от GLOBAL_FLOOR из build.py (пер-процессный «пол», осьминог), такт живёт
# в общей SQLite — один на все процессы.
GRANT_STEP = float(os.environ.get("KB_GRANT_STEP", "3"))
# ⛔ Глобальная abuse-пауза ВЫЧИЩЕНА (канон §2.5, 2026-07-18): была слепо скопирована из embed.
# Защита от пулемётинга = per-key cooldown (COOLDOWN_FIRST/REPEAT выше): задолбанный ключ сам остывает,
# залп 429 гасится ПОШТУЧНО. Останавливать весь пул из-за нескольких ключей — лишнее.
RESERVE = int(os.environ.get("KB_RESERVE", "60"))
# ↑ канон п.2 (резерв primary) + ФАКТ A2 (экстрактор мерено ~22 запр/ключ/ночь, логи 2026-07-14).
#   60 = ~22 + запас. НЕ 120 из build.py.

# Пейсинг — КОНСЕРВАТИВНО, держим ~1/RPM_DIVISOR от лимита RPM (канон наблюдал 2-4 из 15 = далеко).
RPM_DIVISOR = float(
    os.environ.get("KB_RPM_DIV", "4")
)  # 15/4≈3.75 RPM/ключ → шаг ~16с. Далеко под потолком.
_FORCE_STEP = os.environ.get("KB_FORCE_STEP")  # только для тестов (фиксированный шаг)

# ⛔ ГЛОБАЛЬНЫЙ пол (GLOBAL_FLOOR/_GLOBAL) ВЫЧИЩЕН (канон §5, подвал; 2026-07-18): был мой выдуманный
# глобальный аггрегат-дроссель (1 выдача/4.5с на весь пул = ~13/мин) — приблуда из
# extraction._INTER_FILE_PAUSE_S (пауза ВНУТРИ процесса, не глоб-пол). Канон ЗАПРЕЩАЕТ аггрегат поверх
# независимых проектов («душил бы 12 как 1»). Темп держит per-key шаг (step_for). RPM_DIVISOR НЕ трогаю —
# это отдельная (впритык vs запас), гейтчена на 429-body.

# ── ПРЕДОХРАНИТЕЛЬ per-РОТ: рот не съест больше своей доли (защита от runaway → осушения пула).
# Не оптимизация — колпак. Тротл (per-key шаг + cooldown) держит катастрофу (429-шторм); этот кап
# держит «один сломанный рот медленно выел весь пул и заморил остальных».
# Реальные капы ртов задаёт ЮЗЕР через set_cap() (числа не выдумка кода). Незаписанный рот →
# этот дефолт, НИКОГДА не uncapped. Консервативно (дневной приток ~300 мух); юзер уточняет.
DEFAULT_CONSUMER_CAP = int(os.environ.get("KB_DEFAULT_CONSUMER_CAP", "300"))


def _lim(model):
    return LIMITS.get(model, DEFAULT_LIMIT)


def cap_for(model, role):
    """RPD-кап на ключ ПО МОДЕЛИ. primary — полный RPD; background — минус резерв (не ниже 0)."""
    rpd = _lim(model)["rpd"]
    return rpd if role == "primary" else max(0, rpd - RESERVE)


def step_for(model):
    """Шаг per-key ПО МОДЕЛИ от её RPM (факт из LIMITS). Неизвестный/0 → DEFAULT_LIMIT rpm (=5)."""
    if _FORCE_STEP:
        return float(_FORCE_STEP)
    rpm = _lim(model)["rpm"] or DEFAULT_LIMIT["rpm"]
    return 60.0 * RPM_DIVISOR / rpm


def _consumer_cap(c, consumer):
    """Дневной кап рта из consumer_cap; нет записи → DEFAULT_CONSUMER_CAP (никогда не uncapped)."""
    r = c.execute(
        "SELECT rpd_cap FROM consumer_cap WHERE consumer=?", (consumer,)
    ).fetchone()
    return r[0] if r else DEFAULT_CONSUMER_CAP


def set_cap(consumer, rpd_cap, rpm_cap=None):
    """Задать/обновить кап рта. Числа задаёт ЮЗЕР, не код. Вызывать разово при засеве."""
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        c.execute(
            "INSERT INTO consumer_cap(consumer, rpd_cap, rpm_cap) VALUES(?,?,?) "
            "ON CONFLICT(consumer) DO UPDATE SET "
            "rpd_cap=excluded.rpd_cap, rpm_cap=excluded.rpm_cap",
            (consumer, rpd_cap, rpm_cap),
        )
        c.execute("COMMIT")
    finally:
        c.close()


def _log_event(consumer, model, event, status=0):
    """Записать аномалию в request_log отдельным коннектом (чтоб говно было ВИДНО в stats):
    cap_block (рот упёрся в кап) / parse_fail (200, но мусор).
    """
    try:
        c = _conn()
        c.execute(
            "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) "
            "VALUES(?,?,?,?,?,?)",
            (time.time(), consumer, "", model, event, status),
        )
        c.commit()
        c.close()
    except Exception:
        pass  # логирование не должно ронять выдачу


_BODY_LOG = os.path.join(os.path.dirname(DB) or ".", "error_bodies.log")


def _log_body(consumer, model, status, body):
    """Тело ЛЮБОГО не-200 ответа (400/429/5xx/сеть) в файл — диагностика ПРИЧИНЫ.
    В request_log тела нет; ловим здесь, чтобы боевой 400/429 показал, что реально ломает
    (RESOURCE_EXHAUSTED? INVALID_ARGUMENT? SAFETY?). НЕ роняет выдачу (всё в try)."""
    try:
        line = "%.0f\t%s\t%s\t%s\t%s\n" % (
            time.time(),
            status,
            consumer,
            model,
            " ".join((body or "").split())[:800],
        )
        with open(_BODY_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


CAPS = {  # капы ртов: метод «дневной пик × запас≈3», значения из ARCHITECTURE.md (юзер принял).
    "facet": 1500,  # замер: пик 482 ×3
    "translate": 400,  # 482×13яз/50 ×3
    "questions": 300,  # оценка, замерить
    # «consolidate» переименован 07-19: рот должен зваться своим делом (юзер).
    # Старые строки consumer_usage остаются под старым именем — история, не мигрируем.
    "carve": 300,  # экс-consolidate (кап юзер принял); распил плотных семей
    "assign": 100,  # хвост-раскладка по полкам; ~ceil(хвост/90)×≤3 на гео, br≈63
    "faq": 300,  # дефолт-класс
    "synth": 200,  # дефолт-класс
    "labels": 200,  # копейки
    "kratko": 300,  # короткий ответ страницы (dedup.py --kratko): 1 вызов/вид-страница, br=134
}


def seed_caps():
    """Разово залить CAPS в consumer_cap. Идемпотентно (set_cap = upsert)."""
    for consumer, rpd in CAPS.items():
        set_cap(consumer, rpd)


def _kh(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _pt_day():
    return datetime.now(PT).strftime("%Y-%m-%d")


def _conn():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("PRAGMA busy_timeout=8000")
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS key_clock(
            key_hash TEXT PRIMARY KEY,
            next_free REAL DEFAULT 0,
            cooldown_until REAL DEFAULT 0,
            was_cd INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS usage(
            key_hash TEXT, model TEXT, pt_day TEXT,
            count INTEGER DEFAULT 0, banned INTEGER DEFAULT 0,
            PRIMARY KEY(key_hash, model, pt_day)
        );
        CREATE TABLE IF NOT EXISTS request_log(
            ts REAL, consumer TEXT, key_hash TEXT, model TEXT, event TEXT, status INTEGER
        );
        CREATE INDEX IF NOT EXISTS ix_log_ts ON request_log(ts);
        CREATE INDEX IF NOT EXISTS ix_log_429 ON request_log(status, ts);
        CREATE TABLE IF NOT EXISTS consumer_cap(
            consumer TEXT PRIMARY KEY,
            rpd_cap  INTEGER NOT NULL,
            rpm_cap  REAL          -- задел, пока НЕ enforce (темп держит per-ключ шаг)
        );
        CREATE TABLE IF NOT EXISTS consumer_usage(
            consumer TEXT, pt_day TEXT, count INTEGER DEFAULT 0,
            PRIMARY KEY(consumer, pt_day)
        );
        """
    )
    # defensive-миграция для старых broker.db без was_cd
    try:
        c.execute("ALTER TABLE key_clock ADD COLUMN was_cd INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    # очередь пула: busy_ts в broker_global = время последней выдачи (такт GRANT_STEP);
    # busy_consumer — кто взял последним (диагностика)
    c.execute(
        "CREATE TABLE IF NOT EXISTS broker_global("
        "id INTEGER PRIMARY KEY CHECK(id=1), abuse_pause_until REAL DEFAULT 0)"
    )
    for col, typ in (("busy_consumer", "TEXT"), ("busy_ts", "REAL DEFAULT 0")):
        try:
            c.execute(f"ALTER TABLE broker_global ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    c.execute("INSERT OR IGNORE INTO broker_global(id) VALUES(1)")
    c.commit()
    c.close()


def acquire(consumer, role, model, keys):
    """Взять ключ через центр. role: 'primary'|'background'. keys: список api-ключей.
    Возврат:
      (key, None)        — выдан ключ;
      (None, 0.0)        — ОЧЕРЕДЬ: такт выдачи ещё не прошёл; ждать НЕ тратит бюджет
                           вызова (канон юзера: пауз нет, только очередь);
      (None, wait_s>0)   — сейчас нет, но освободится через wait_s (RPM/clock);
      (None, -1.0)       — все ключи на капе/бане (бюджет модели выбран).
    """
    cap = cap_for(model, role)
    step = step_for(model)
    now = time.time()
    day = _pt_day()
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        # ОЧЕРЕДЬ ПУЛА: такт на выдачу — с последней выдачи прошло < GRANT_STEP → ждать.
        brow = c.execute("SELECT busy_ts FROM broker_global WHERE id=1").fetchone()
        if brow and now - (brow[0] or 0) < GRANT_STEP:
            c.execute("ROLLBACK")
            return (None, 0.0)  # очередь: стоим у кассы, бюджет вызова не тратим
        # Капы ртов СНЯТЫ (юзер 2026-07-21: «убрать капы и измерить реальные запросы»).
        # Учёт consumer_usage ЖИВ — по нему меряем реальный расход каждого рта; единственный
        # enforce-забор = per-ключ RPD + такт очереди. Таблица consumer_cap осталась заделом.
        clocks = {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT key_hash, next_free, cooldown_until FROM key_clock"
            )
        }
        used = {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT key_hash, count, banned FROM usage WHERE model=? AND pt_day=?",
                (model, day),
            )
        }
        elig, soonest = [], None
        for k in keys:
            kh = _kh(k)
            nf, cd = clocks.get(kh, (0.0, 0.0))
            cnt, ban = used.get(kh, (0, 0))
            if ban or cnt >= cap:
                continue  # RPD/бан — этот ключ не годен вовсе
            if cd > now:
                continue  # в 429-cooldown
            if nf > now:  # годен, но ещё не остыл по RPM — кандидат на wait
                soonest = nf if soonest is None else min(soonest, nf)
                continue
            elig.append((k, kh, nf))
        if not elig:
            c.execute("ROLLBACK")
            if soonest is not None:
                return (None, max(0.05, soonest - now))
            return (None, -1.0)  # все на капе/бане
        elig.sort(key=lambda x: x[2])  # самый остывший
        key, kh, _ = elig[0]
        c.execute(
            "INSERT INTO key_clock(key_hash, next_free, cooldown_until) VALUES(?,?,0) "
            "ON CONFLICT(key_hash) DO UPDATE SET next_free=excluded.next_free",
            (kh, now + step),
        )
        c.execute(
            "INSERT INTO usage(key_hash, model, pt_day, count) VALUES(?,?,?,1) "
            "ON CONFLICT(key_hash, model, pt_day) DO UPDATE SET count=count+1",
            (kh, model, day),
        )
        c.execute(  # per-РОТ счёт дня — рядом с per-ключ usage, в той же транзакции
            "INSERT INTO consumer_usage(consumer, pt_day, count) VALUES(?,?,1) "
            "ON CONFLICT(consumer, pt_day) DO UPDATE SET count=count+1",
            (consumer, day),
        )
        c.execute(
            "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) VALUES(?,?,?,?,'grant',0)",
            (now, consumer, kh, model),
        )
        c.execute(  # метка выдачи: следующий рот подойдёт через GRANT_STEP
            "UPDATE broker_global SET busy_consumer=?, busy_ts=? WHERE id=1",
            (consumer, now),
        )
        c.execute("COMMIT")
        return (key, None)
    finally:
        c.close()


def report(consumer, key, model, status):
    """Отчёт об исходе — per-key эскалация 429 (из эталона gemini_brain, БЕЗ глоб-abuse §2.5):
    1-й 429 → cooldown 300с + пометка was_cd;  повторный 429 → 1800с (НЕ дневной бан);
    успех(200) → прощение (разбан): сброс cooldown И истории was_cd.
    """
    kh = _kh(key)
    now = time.time()
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        c.execute(
            "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) VALUES(?,?,?,?,'report',?)",
            (now, consumer, kh, model, status),
        )
        if status == 429:
            # МЕХАНИЗМ вина-vs-среда: 429 на ДРУГОМ ключе в окне ENV429_WINDOW = волна
            # среды (общий источник, не грех ключа) → cooldown НЕ ставим, пул не
            # прожигается (макс. один ключ на волну — первый). Пересиживание окна —
            # backoff в call(). Изолированный 429 = вина ключа → эскалация как была.
            env = c.execute(
                "SELECT 1 FROM request_log WHERE status=429 AND ts>? AND key_hash<>? "
                "AND event='report' LIMIT 1",
                (now - ENV429_WINDOW, kh),
            ).fetchone()
            if env:
                c.execute(
                    "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) "
                    "VALUES(?,?,?,?,'429_env',429)",
                    (now, consumer, kh, model),
                )
            else:
                row = c.execute(
                    "SELECT was_cd FROM key_clock WHERE key_hash=?", (kh,)
                ).fetchone()
                was = row[0] if row else 0
                cd = (
                    COOLDOWN_REPEAT if was else COOLDOWN_FIRST
                )  # повтор → 30мин, первый → 5мин
                c.execute(
                    "INSERT INTO key_clock(key_hash, next_free, cooldown_until, was_cd) VALUES(?,0,?,1) "
                    "ON CONFLICT(key_hash) DO UPDATE SET cooldown_until=excluded.cooldown_until, was_cd=1",
                    (kh, now + cd),
                )
            # abuse-пауза вычищена (§2.5): волна среды глушится НЕ-наказанием + backoff,
            # виновный ключ остужается поштучно.
        elif status == 200:
            c.execute(
                "UPDATE key_clock SET cooldown_until=0, was_cd=0 WHERE key_hash=?",
                (kh,),
            )
        c.execute("COMMIT")
    finally:
        c.close()


# ─────────────────────────────────────────────── СОСОК: единственная дверь к Gemini
# Ключи живут ТОЛЬКО здесь (ниппель: рот молока не касается — получает dict, не ключ).
_KEYS = None


def get_keys():
    """Ключи из env бот-контейнера. Кэш на процесс — это чтение env, НЕ состояние пейсинга
    (то в SQLite, переживает спавн).

    Порядок как у `config.gemini_api_keys_from_env` (CLAUDE.md): numbered `GEMINI_API_KEY_N`
    → legacy `GEMINI_API_KEYS` (comma) → single `GEMINI_API_KEY`. ⚠️ РАНЬШЕ грепал
    `startswith("GEMINI_API_KEY")` и хватал ЛЕГАСИ `GEMINI_API_KEYS` как ЛИШНИЙ богус-ключ →
    400 API_KEY_INVALID на ~1/N запросов (диагностика 2026-07-18). Теперь через precedence.
    """
    global _KEYS
    if _KEYS is None:
        cid = subprocess.check_output(
            "docker ps --format '{{.Names}}' | grep bots-grab | head -1",
            shell=True,
            text=True,
        ).strip()
        env = subprocess.check_output(["docker", "exec", cid, "printenv"], text=True)
        vals = {}
        for ln in env.splitlines():
            if "=" in ln:
                name, val = ln.split("=", 1)
                vals[name] = val
        numbered = [
            vals[k]
            for k in sorted(
                (k for k in vals if re.fullmatch(r"GEMINI_API_KEY_\d+", k)),
                key=lambda k: int(k.rsplit("_", 1)[1]),
            )
            if vals[k].strip()
        ]
        if numbered:  # numbered есть → легаси/single ИГНОРИРУЕМ (как хелпер)
            _KEYS = numbered
        elif vals.get("GEMINI_API_KEYS", "").strip():
            _KEYS = [k.strip() for k in vals["GEMINI_API_KEYS"].split(",") if k.strip()]
        elif vals.get("GEMINI_API_KEY", "").strip():
            _KEYS = [vals["GEMINI_API_KEY"].strip()]
        else:
            _KEYS = []
        if not _KEYS:
            sys.exit("no GEMINI keys in container env")
    return _KEYS


MAX_FAILS = 5  # РЕАЛЬНЫХ провалов (429/4xx/5xx/сеть) на 1 вызов → сдаёмся
MAX_WAIT_TOTAL = 1800.0  # суммарный бюджет ОЖИДАНИЯ слота на 1 вызов (30 мин)
MAX_SLEEP = 30.0  # максимум спим за один «нет слота», потом снова спрашиваем мозг


def call(
    user,
    sysprompt,
    consumer,
    model="gemini-3.1-flash-lite",
    role="background",
    timeout=60,
):
    """СОСОК. Рот шлёт (user, sysprompt) + называет себя (consumer) и молоко (model);
    получает dict или None. Ключа рот НЕ видит — он живёт и умирает здесь.
    acquire → HTTP → report (в finally: выйти без учёта негде) → разбор JSON.

    WORST-CASE НА ВЫЗОВ: ≤MAX_FAILS(5) реальных запросов к Google + ≤MAX_WAIT_TOTAL(30мин)
    сна в ожидании СЛОТА КЛЮЧА + ≤75с backoff-сна при серии 429 (окно троттла).
    Очередь пула (такт выдачи GRANT_STEP) ждётся ОТДЕЛЬНО и бюджет не тратит — из
    очереди не уходят; полёт (HTTP) пул не держит — блокировки как класса нет.
    None: бюджет выбран / 5 провалов / 30мин без слота ключа.
    """
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": sysprompt}]},
        "generationConfig": {"responseMimeType": "application/json"},
    }
    data = json.dumps(payload).encode()
    keys = get_keys()
    fails, waited = 0, 0.0
    while fails < MAX_FAILS and waited < MAX_WAIT_TOTAL:
        key, wait = acquire(consumer, role, model, keys)
        if key is None:
            if wait is None or wait < 0:
                print(f"  бюджет модели {model} выбран — стоп ({consumer})")
                return None
            if wait == 0.0:  # ОЧЕРЕДЬ (такт выдачи): стоим сколько нужно,
                time.sleep(0.7)  # бюджет вызова НЕ тратим — из очереди не уходят
                continue
            nap = min(wait, MAX_SLEEP)
            time.sleep(nap)
            waited += nap
            continue
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={key}"
        )
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        body, status, err = None, 0, ""
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            body = r.read()
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                err = e.read().decode("utf-8", "replace")[:800]
            except Exception:
                err = str(e)[:120]
        except Exception as e:
            status = -1  # сеть/таймаут — ключ не виноват
            err = str(e)[:120]
        finally:
            report(consumer, key, model, status)  # ← выйти без учёта НЕГДЕ

        if status != 200:
            _log_body(
                consumer, model, status, err
            )  # тело в error_bodies.log — диагностика причины

        if status == 200:
            try:
                api = json.loads(body)
                raw = api["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(re.sub(r"```json|```", "", raw).strip())
            except Exception as e:
                print("  parse err", str(e)[:100])
                _log_event(
                    consumer, model, "parse_fail"
                )  # 200-мусор виден, не как успех
                return None
        fails += 1
        if status == 429:
            # Ключу — cooldown (мозг уже поставил). Но СЛЕДУЮЩИЙ ключ брать НЕ сразу:
            # 429 при пустых квотах = общее окно RESOURCE_EXHAUSTED (по-IP/server-side;
            # факт 2026-07-20: волны ровно по 12×429 на весь пул при 22-45 grants/мин —
            # continue без паузы за секунды прожигал ВСЕ ключи в cooldown). Пересиживаем
            # окно backoff'ом ВНУТРИ вызова (5→10→20→40с) — это не глобальный аггрегат
            # (запрещён каноном §5), пул другим ртам не держим.
            nap = min(5.0 * (2 ** (fails - 1)), 60.0)
            time.sleep(nap)
            waited += nap
            continue
        if status in (500, 502, 503, -1):
            continue  # транзиент сервера/сети → другой ключ (пейсинг per-key, без выдуманной паузы)
        if status in (400, 403):
            return None  # проблема запроса/ключа (INVALID_ARGUMENT) — ретрай не лечит
        print("  HTTP", status, err[:200])
        return None
    print(f"  сдались ({consumer}): fails={fails}/{MAX_FAILS}, ждали {waited:.0f}с")
    return None


def stats(hours=24):
    """Короткая сводка для мониторинга/статы."""
    c = _conn()
    since = time.time() - hours * 3600
    rows = c.execute(
        "SELECT consumer, event, status, COUNT(*) FROM request_log WHERE ts>=? "
        "GROUP BY consumer, event, status ORDER BY consumer",
        (since,),
    ).fetchall()
    c.close()
    return rows


if __name__ == "__main__":
    init()
    seed_caps()
    print(
        "keybroker init OK:", DB, "| caps seeded:", len(CAPS)
    )  # ASCII: не падать под C-локалью
