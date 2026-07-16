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
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

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
ABUSE_THRESHOLD = 3  # _EMBED_ABUSE_THRESHOLD: ≥3 ответа 429 за окно
ABUSE_WINDOW = 60.0  # _EMBED_ABUSE_WINDOW_S
ABUSE_PAUSE = 1800.0  # _EMBED_ABUSE_PAUSE_S: тогда ГЛОБАЛЬНАЯ пауза всему пулу 30мин (не жечь каждый ключ)
RESERVE = int(os.environ.get("KB_RESERVE", "60"))
# ↑ канон п.2 (резерв primary) + ФАКТ A2 (экстрактор мерено ~22 запр/ключ/ночь, логи 2026-07-14).
#   60 = ~22 + запас. НЕ 120 из build.py.

# Пейсинг — КОНСЕРВАТИВНО, держим ~1/RPM_DIVISOR от лимита RPM (канон наблюдал 2-4 из 15 = далеко).
RPM_DIVISOR = float(
    os.environ.get("KB_RPM_DIV", "4")
)  # 15/4≈3.75 RPM/ключ → шаг ~16с. Далеко под потолком.
STEP_UNVERIFIED = 30.0  # модель с неизвестным RPM → консервативно 30с (2 RPM), пока не подтвердишь скрином
_FORCE_STEP = os.environ.get("KB_FORCE_STEP")  # только для тестов (фиксированный шаг)

# ГЛОБАЛЬНЫЙ пол между ЛЮБЫМИ двумя выдачами (независимо от ключа) — как extraction._INTER_FILE_PAUSE_S=4.5.
# Канон НИКОГДА не шлёт back-to-back: даже при мгновенных 429 нельзя прожечь пул машинганом за секунду.
GLOBAL_FLOOR = float(os.environ.get("KB_GLOBAL_FLOOR", "4.5"))
_GLOBAL = "__global__"  # спец-ключ в key_clock: next_free = когда можно ЛЮБУЮ следующую выдачу

# ── ПРЕДОХРАНИТЕЛЬ per-РОТ: рот не съест больше своей доли (защита от runaway → осушения пула).
# Не оптимизация — колпак. Тротл (GLOBAL_FLOOR+шаг+abuse) держит катастрофу (429-шторм); этот кап
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
    """Шаг per-key ПО МОДЕЛИ, консервативно от её RPM. Неизвестный RPM → фикс-консерватив, не выдумка."""
    if _FORCE_STEP:
        return float(_FORCE_STEP)
    rpm = _lim(model)["rpm"]
    if not rpm:
        return STEP_UNVERIFIED
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
    cap_block (рот упёрся в кап) / parse_fail (200, но мусор) / abuse_pause (пул встал).
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


CAPS = {  # капы ртов: метод «дневной пик × запас≈3», значения из ARCHITECTURE.md (юзер принял).
    "facet": 1500,  # замер: пик 482 ×3
    "translate": 400,  # 482×13яз/50 ×3
    "questions": 300,  # оценка, замерить
    "consolidate": 300,  # оценка, замерить
    "faq": 300,  # дефолт-класс
    "synth": 200,  # дефолт-класс
    "labels": 200,  # копейки
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
        CREATE TABLE IF NOT EXISTS broker_global(
            id INTEGER PRIMARY KEY CHECK(id=1),
            abuse_pause_until REAL DEFAULT 0
        );
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
    c.commit()
    c.close()


def acquire(consumer, role, model, keys):
    """Взять ключ через центр. role: 'primary'|'background'. keys: список api-ключей.
    Возврат:
      (key, None)        — выдан ключ;
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
        # ГЛОБАЛЬНАЯ abuse-пауза (эталон): если недавно ≥3 ответа 429 в окне — весь пул отдыхает,
        # НЕ выпиливаем каждый ключ по отдельности. Снимает смерть-спираль из short-leg 2026-07-14.
        gp = c.execute(
            "SELECT abuse_pause_until FROM broker_global WHERE id=1"
        ).fetchone()
        if gp and gp[0] > now:
            c.execute("ROLLBACK")
            return (None, gp[0] - now)
        # ПРЕДОХРАНИТЕЛЬ per-РОТ: рот выбрал дневную долю → отказ (как per-ключ кап). Не даём
        # одному сломанному рту осушить весь пул и заморить остальных.
        crow = c.execute(
            "SELECT count FROM consumer_usage WHERE consumer=? AND pt_day=?",
            (consumer, day),
        ).fetchone()
        if crow and crow[0] >= _consumer_cap(c, consumer):
            c.execute("ROLLBACK")
            _log_event(consumer, model, "cap_block", crow[0])  # рот виден в stats
            return (None, -1.0)
        clocks = {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT key_hash, next_free, cooldown_until FROM key_clock"
            )
        }
        # ГЛОБАЛЬНЫЙ пол: между любыми двумя выдачами ≥ GLOBAL_FLOOR — не даём машинган по пулу.
        gnext = clocks.get(_GLOBAL, (0.0, 0.0))[0]
        if gnext > now:
            c.execute("ROLLBACK")
            return (None, gnext - now)
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
        c.execute(  # глобальный пол — следующая ЛЮБАЯ выдача не раньше now+GLOBAL_FLOOR
            "INSERT INTO key_clock(key_hash, next_free, cooldown_until) VALUES(?,?,0) "
            "ON CONFLICT(key_hash) DO UPDATE SET next_free=excluded.next_free",
            (_GLOBAL, now + GLOBAL_FLOOR),
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
        c.execute("COMMIT")
        return (key, None)
    finally:
        c.close()


def report(consumer, key, model, status):
    """Отчёт об исходе — эскалация 429 ДОСЛОВНО из эталона gemini_brain:
    1-й 429 → cooldown 300с + пометка was_cd;  повторный 429 → 1800с (НЕ дневной бан);
    успех(200) → прощение: сброс cooldown И истории was_cd;
    abuse: ≥3 ответа 429 за 60с → глобальная пауза 1800с всему пулу.
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
            # abuse-окно: текущий 429 уже в логе, считаем все 429-репорты за ABUSE_WINDOW
            n = c.execute(
                "SELECT COUNT(*) FROM request_log WHERE event='report' AND status=429 AND ts>=?",
                (now - ABUSE_WINDOW,),
            ).fetchone()[0]
            if n >= ABUSE_THRESHOLD:
                c.execute(
                    "INSERT INTO broker_global(id, abuse_pause_until) VALUES(1,?) "
                    "ON CONFLICT(id) DO UPDATE SET abuse_pause_until=excluded.abuse_pause_until",
                    (now + ABUSE_PAUSE,),
                )
                c.execute(  # пул встал — видно в stats
                    "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) "
                    "VALUES(?,?,?,?,'abuse_pause',?)",
                    (now, consumer, "", model, n),
                )
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
    (то в SQLite, переживает спавн)."""
    global _KEYS
    if _KEYS is None:
        cid = subprocess.check_output(
            "docker ps --format '{{.Names}}' | grep bots-grab | head -1",
            shell=True,
            text=True,
        ).strip()
        env = subprocess.check_output(["docker", "exec", cid, "printenv"], text=True)
        _KEYS = [
            ln.split("=", 1)[1]
            for ln in env.splitlines()
            if ln.startswith("GEMINI_API_KEY") and "=" in ln
        ]
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
    сна в ожидании слота. Пулемёта нет: сон только по указанию мозга.
    None: бюджет модели выбран / не смогли за 5 попыток / вышли за 30мин ожидания.
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
                err = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                err = str(e)[:120]
        except Exception as e:
            status = -1  # сеть/таймаут — ключ не виноват
            err = str(e)[:120]
        finally:
            report(consumer, key, model, status)  # ← выйти без учёта НЕГДЕ

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
            continue  # мозг уже поставил cooldown → берём другой ключ
        if status in (400, 500, 502, 503, -1):
            time.sleep(3)  # транзиент → повторить
            continue
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
