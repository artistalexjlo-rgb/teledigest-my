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

# ЛЕСТНИЦА ОТДЫХА КЛЮЧА (числа юзера, 2026-07-21). Первый 429 — только МЕТКА, наказания
# нет: следующий ход ключа всё равно наступит через оборот круга. Отказал СНОВА — ступени:
#   60с   — пересечь минутную границу (оборота в ~12с не хватит, если 429 из минутного ведра);
#   300с  — минута не помогла, дело не в минутном ведре;
#   1800с — не помогли и пять минут;
#   6000с — не остыл за полчаса.
# Прошёл ВСЮ лестницу и снова 429 → ДНЕВНОЙ БАН (канон §6: подтверждённое дневное
# исчерпание — тут подтверждение делом, ~2.3 часа эскалации; тело 429 у Google немое).
# ⛔ Бан НИКОГДА не ставится раньше полной лестницы (катастрофа экстрактора — бан с первого
# 429) и истекает по PACIFIC: usage ведётся по pt_day, завтрашняя строка чистая.
# Отсидка САМА ПО СЕБЕ ступень не обнуляет («он же не остыл» — юзер): обнуляет УСПЕХ либо
# ДАВНОСТЬ СЕРИИ (см. ниже).
COOLDOWN_LADDER = (60.0, 300.0, 1800.0, 6000.0)
# ДАВНОСТЬ СЕРИИ (2026-07-25). Ступень принадлежит СЕРИИ подряд идущих отказов, а НЕ ключу:
# «первый 429» — первый В СЕРИИ, а где серия кончается, до сих пор не было сказано нигде.
# Срок забвения = только что отсиженная отсидка (у голой метки — первая ступень): отказал
# снова в пределах этого срока → та же серия, лестница вверх; прожил тихо дольше → серия
# кончилась, счёт с нуля.
# ⛔ Отдельного числа НЕ вводим: масштаб уже задан самой лестницей. Фиксированный порог
# обрезал бы её верх — отсидевший 1800с всегда возвращался бы «нескоро», и до 6000 ключ не
# дошёл бы никогда. Это reset timeout из circuit breaker, стандарт, а не наша выдумка.
# Зачем понадобилось: 07-24 все 12 ключей остались с cd_level≥1 при кулдаунах, истёкших
# ЧЕТЫРЕ ЧАСА назад — по часам здоровы, по лестнице больны, и первый же 429 наказал бы их
# тремястами секунд вместо бесплатной метки. Поэтому cooldown_until пишется И на метке
# (отсидка нулевой длины): поле всегда значит «когда закончилось последнее наказание».
# ОЧЕРЕДЬ ПУЛА (канон юзера 2026-07-20: раннеры снесены — очередь держит МОЗГ).
# Такт НА ВЫДАЧУ, не на полёт: мозг отдаёт ключи по одному и не чаще раза в такт
# на весь пул; выдал — следующий подходит через такт, а полёт (HTTP) идёт сам и никого
# не держит (юзер: «рот получил ключ и пошёл — сосок ему не нужен»; длинный перевод
# больше не блокирует пул). В отличие от GLOBAL_FLOOR из build.py (пер-процессный «пол»,
# осьминог), такт живёт в общей SQLite — один на все процессы.
# ДИНАМИЧЕСКИЙ ТАКТ (юзер 07-24): меньше ЖИВЫХ ключей → больше такт, чтобы темп НА КЛЮЧ
# держался безопасным. clamp((MAX+1)−живых/2, MIN, MAX): 1→5, 2→5, 3→4.5, 8→2, 12→1.
# Худший случай на ключ — 12 RPM при ОДНОМ живом (5с × 1 ключ), под лимитом 15. Поэтому
# отдельный пол на ключ не нужен: такт покрывает его на всём диапазоне.
# ⛔ Низ 2 (07-24, «пул ≤30/мин») был ПРОВЕРКОЙ ТОРМОЖЕНИЕМ — не помогло, дело было не в
# темпе НА КЛЮЧ (юзер снял 07-25), и низ вернули на 1.
#
# ⭐ НИЗ 3 (юзер, 2026-07-26) — это ПОТОЛОК ПУЛА, а не темп на ключ. Разница принципиальна:
# замерами 25-26.07 установлено, что стена 429 сидит на ИСТОЧНИКЕ, а не на ключе — свежий
# нетронутый ключ падает на первом же запросе, при том что RPD 2/440, RPM 3.9/15,
# TPM 8К/250К (ни один персональный лимит не тронут). Граница: чисто ≤16-23 запроса/мин
# НА ВЕСЬ ПУЛ, число ключей на неё не влияет. Разбор — memory/fact_429_source_limit.md.
# При 12 живых clamp даёт ровно низ: 3с = 20 запросов/мин, внутри чистой зоны. Это возвращает
# то, чем был GRANT_STEP=3 до перехода на динамику, и лежит рядом с темпом экстрактора
# (4.5с между файлами + 60с после круга = 6.3/мин), который живёт 3 месяца без шторма.
# На ключ приходится 1.7 обращения в минуту — вдесятеро под лимитом 15.
# ⛔ Батч рта потолок НЕ заменяет: 26.07 батч facet свёл разметку к 4 запросам и нулю
# отказов, но следом carve разогнался до 26/мин и лёг на 45-м. Регулятора темпа нет ни у
# одного рта — только этот такт.
GRANT_MAX = float(os.environ.get("KB_GRANT_MAX", "5"))  # верх такта
GRANT_MIN = float(os.environ.get("KB_GRANT_MIN", "3"))  # низ = потолок пула 20/мин

# ⭐⭐ ЗВЕНЬЯ (юзер, 2026-07-27): работаем не всем пулом, а звеном по 4 ключа — звено
# работает какое-то время, потом пауза, потом следующее; за три звена обходим все 12.
#
# ЗАЧЕМ. Замер 27.07 закрыл вопрос, от чего стена 429, и ответ не тот, что мы крутили
# сутки. Два прогона с ОДИНАКОВЫМ темпом пула 13 запросов/мин:
#   4 ключа  (такт 4.0с) → каждый ключ по 7-8 запросов, отказов НОЛЬ
#   12 ключей (такт 4.6с) → ТЕ ЖЕ САМЫЕ ключи сыплются со 2-3-го обращения
# На четырёх ключ дёргали раз в 16с, на двенадцати — раз в 55с: давили ВТРОЕ РЕЖЕ, и легло.
# Значит решает не темп и не нагрузка на ключ, а СКОЛЬКО РАЗНЫХ КЛЮЧЕЙ засветилось с адреса.
# Это же объясняет то, чего не объясняла ни одна прошлая версия: почему свежий, ни разу не
# использованный ключ падает на первом запросе — он не исчерпан, он просто седьмой подряд.
# И тот же механизм в кроссовере 26.07: первые 5-6 ключей проходят, дальше отсекает всех
# независимо от того, какие это ключи. Разбор — memory/fact_429_source_limit.md.
#
# ⛔ Такт (GRANT_*) это НЕ лечит и лечить не может: он про скорость, а упираемся в состав.
# Проверено боем — 1.0с, 3.0с и 4.6с дали стену одинаково.
#
# ЧИСЛА. Размер звена 4 — юзер. Работа 180с взята с ЧИСТОГО прогона (06:55-06:58: четыре
# ключа, 3.5 минуты, 31 запрос, ноль отказов) — измерена, но только СНИЗУ: сколько звено
# выдержит сверх этого, неизвестно. Пауза 60с — ГАДАНИЕ, ничем не подтверждена; её и надо
# крутить первой по факту прогона. KB_GROUP_SIZE=0 выключает звенья целиком.
GROUP_SIZE = int(os.environ.get("KB_GROUP_SIZE", "4"))  # ключей в звене; 0 = выключено
GROUP_WORK = float(os.environ.get("KB_GROUP_WORK", "180"))  # звено работает, сек
GROUP_PAUSE = float(os.environ.get("KB_GROUP_PAUSE", "60"))  # пауза между звеньями, сек


def _dyn_grant_step(alive):
    alive = max(1, alive)
    return max(GRANT_MIN, min(GRANT_MAX, (GRANT_MAX + 1.0) - alive / 2.0))


# ⛔ ПАУЗА НА ЗАКРЫТИИ КРУГА (ROUND_PAUSE) УДАЛЕНА 2026-07-25 вместе со сквозной нумерацией
# оборотов: наследие эмбеддинга/экстрактора (пауза между МОДЕЛЯМИ с RPD 20) — у нас одна
# модель, а очередь теперь без «оборотов», приткнуть паузу некуда. Темп держит такт.
# ⛔ Глобальная abuse-пауза ВЫЧИЩЕНА (канон §2.5, 2026-07-18): была слепо скопирована из embed.
# Защита от пулемётинга = лестница отдыха (COOLDOWN_LADDER выше): задолбанный ключ сам остывает,
# залп 429 гасится ПОШТУЧНО. Останавливать весь пул из-за нескольких ключей — лишнее.
RESERVE = int(os.environ.get("KB_RESERVE", "60"))
# ↑ канон п.2 (резерв primary) + ФАКТ A2 (экстрактор мерено ~22 запр/ключ/ночь, логи 2026-07-14).
#   60 = ~22 + запас. НЕ 120 из build.py.

# ⛔ PER-KEY ШАГ (RPM_DIVISOR=4 → 16с на ключ) УДАЛЁН 2026-07-21: моё число, и оно лишнее —
# КРУГ уже гарантирует, что ключ не получит второй запрос, пока не отработают все
# остальные. Темп держат круг + такт между выдачами.

# ⛔ ГЛОБАЛЬНЫЙ пол (GLOBAL_FLOOR/_GLOBAL) ВЫЧИЩЕН (канон §5, подвал; 2026-07-18): был мой выдуманный
# глобальный аггрегат-дроссель (1 выдача/4.5с на весь пул = ~13/мин) — приблуда из
# extraction._INTER_FILE_PAUSE_S (пауза ВНУТРИ процесса, не глоб-пол). Канон ЗАПРЕЩАЕТ аггрегат поверх
# независимых проектов («душил бы 12 как 1»). Темп держат КРУГ + такт между выдачами.

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
_HDR_LOG = os.path.join(os.path.dirname(DB) or ".", "ratelimit_headers.log")
_HDR_SEEN = [0]  # первые N ответов логируем ВСЕ имена заголовков (разведка)
_TRACE = os.path.join(os.path.dirname(DB) or ".", "grant_trace.tsv")
_TRACE_CTX = (
    {}
)  # контекст последнего гранта (acquire кладёт, call пишет строку с исходом)
_CALL_SEQ = [0]  # сквозной номер логического вызова в этом процессе


def _trace_row(status, call_no=0, attempt=0, chars=0, toks=0, ms=0):
    """ТРАССА (строка на запрос). Колонки 2026-07-25:
      сек | №ключа | статус | alive | step | rpd_ключа | вызов# | попытка | симв | токены | мс | грантов_за_60с

    Зачем расширена. Все три лимита КЛЮЧА закрыты замером и ни один не выбран (RPD 2/440,
    RPM 3.9/15, TPM ~8К/250К при мухах ≤898 символов), а 429 есть. Значит различать надо то,
    что этими колонками не видно:
      вызов#/попытка — ОДНА муха валится на четырёх разных ключах или четыре разные?
                       Первое = виноват ЗАПРОС, второе = виноват пул. Сейчас `сдались 4/4`
                       эти два случая не различает вовсе.
      симв/токены    — размер запроса; токены берём из usageMetadata самого Google (на 429
                       он их не отдаёт, поэтому симв нужны отдельно — иначе отказавшие
                       запросы останутся без размера).
      мс             — отказ за 20мс (отбит на входе) ≠ отказ за 800мс (дошёл до модели).
      грантов_за_60с — суммарный темп пула читается в строке, а не восстанавливается потом.
    ⚠️ Старые строки — 6 колонок, новые 12; различать по их числу.
    """
    try:
        d = _TRACE_CTX
        if not d:
            return
        row = [
            round(d.get("t", 0.0), 1),
            d.get("keyno", "?"),
            status,
            d.get("alive", "?"),
            d.get("step", 0.0),
            d.get("rpd", "?"),
            call_no,
            attempt,
            chars,
            toks,
            ms,
            d.get("rate60", "?"),
        ]
        with open(_TRACE, "a", encoding="utf-8") as f:
            f.write("\t".join(str(x) for x in row) + "\n")
    except Exception:
        pass


def _log_hdrs(consumer, model, status, hdrs):
    """РАЗВЕДКА: шлёт ли Google остаток квоты в заголовках ответа (доку это НЕ подтверждает,
    но многие API шлют недокументированно). Пишем квота-подобные заголовки; первые 3 ответа —
    ВСЕ имена, чтобы увидеть, что вообще есть. Пассивно, лишней квоты не жжёт."""
    try:
        items = list(hdrs.items())
        quota = {
            k: v
            for k, v in items
            if any(
                t in k.lower()
                for t in ("ratelimit", "quota", "remaining", "reset", "x-goog")
            )
        }
        extra = ""
        if _HDR_SEEN[0] < 3:
            _HDR_SEEN[0] += 1
            extra = " | ВСЕ=" + ",".join(k for k, _ in items)
        line = "%.0f\t%s\t%s\t%s\tquota=%s%s\n" % (
            time.time(),
            status,
            consumer,
            model,
            json.dumps(quota, ensure_ascii=False),
            extra,
        )
        with open(_HDR_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


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


# ⛔ Капы НЕ enforce (сняты юзером 2026-07-21). Числа ниже — МОИ оценки на глаз («×3»),
# юзер их НЕ принимал (оспаривал translate=400). Держим как исторический ориентир объёмов;
# реальный расход мерит consumer_usage.
CAPS = {
    "facet": 1500,  # замер: пик 482 ×3
    "translate": 400,  # 482×13яз/50 ×3
    "questions": 300,  # оценка, замерить
    # «consolidate» переименован 07-19: рот должен зваться своим делом (юзер).
    # Старые строки consumer_usage остаются под старым именем — история, не мигрируем.
    "carve": 300,  # экс-consolidate; распил плотных семей
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


_SCHEMA_OK = False


def _conn():
    global _SCHEMA_OK
    c = sqlite3.connect(DB, timeout=10)
    c.execute("PRAGMA busy_timeout=8000")
    c.execute("PRAGMA journal_mode=WAL")
    if not _SCHEMA_OK:
        # ⚠️ БОЕВОЙ ПРОЦЕСС init() НЕ ЗОВЁТ: база уже есть, рот идёт сразу в acquire.
        # Из-за этого миграция новых колонок не применялась и мозг падал на живом
        # прогоне (07-21: «no such column: served_round»). Схема досоздаётся ЛЕНИВО,
        # при первом подключении процесса: CREATE IF NOT EXISTS + ALTER идемпотентны.
        _SCHEMA_OK = True  # ставим ДО init(): он сам зовёт _conn(), иначе рекурсия
        try:
            init()
        except Exception as e:
            print("keybroker: миграция схемы не прошла:", e)
    return c


def init():
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS key_clock(
            key_hash TEXT PRIMARY KEY,
            cooldown_until REAL DEFAULT 0
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
    # очередь пула: busy_ts в broker_global = время последней выдачи (такт);
    # busy_consumer — кто взял последним (диагностика)
    c.execute(
        "CREATE TABLE IF NOT EXISTS broker_global("
        "id INTEGER PRIMARY KEY CHECK(id=1), abuse_pause_until REAL DEFAULT 0)"
    )
    # ⛔ served_round/next_free/was_cd БОЛЬШЕ НЕ ЗАВОДЯТСЯ (2026-07-25): очередь идёт по
    # last_grant_ts, сквозной нумерации оборотов нет. В старых базах эти колонки остаются
    # лежать — код их просто не читает, дропать (и рисковать живой базой) незачем.
    for col, typ in (
        ("last_grant_ts", "REAL DEFAULT 0"),  # когда ключом пользовались последний раз
        ("struck", "INTEGER DEFAULT 0"),  # метка первого 429 (без наказания)
        ("cd_level", "INTEGER DEFAULT 0"),  # ступень лестницы отдыха (0 = здоров)
    ):
        try:
            c.execute(f"ALTER TABLE key_clock ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in (
        ("busy_consumer", "TEXT"),
        ("busy_ts", "REAL DEFAULT 0"),
        # ⭐ КУРСОР КРУГА: индекс последнего выданного ключа в списке. ОДНО число, ни с чем
        # не сопряжённое — рассинхрон, из-за которого снесли пару round_no+served_round,
        # тут невозможен по построению.
        ("cursor", "INTEGER DEFAULT -1"),
        # ⭐ ЗВЕНО: какой срез списка дежурит и когда заступил. Тоже без пары-двойника —
        # фаза (работа/пауза/смена) вычисляется из одного grp_since, хранить её негде.
        ("grp", "INTEGER DEFAULT 0"),
        ("grp_since", "REAL DEFAULT 0"),
    ):
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
      (None, wait_s>0)   — годных сейчас нет, но через wait_s кто-то выйдет из кулдауна;
      (None, -1.0)       — все ключи на капе/бане: до конца PT-суток ждать НЕЧЕГО.
    """
    cap = cap_for(model, role)
    now = time.time()
    day = _pt_day()
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        # Капы ртов СНЯТЫ (юзер 2026-07-21). Состояние ключей грузим ДО такт-проверки:
        # такт ДИНАМИЧЕСКИЙ от числа ЖИВЫХ ключей (не бан/кап/кулдаун).
        clocks = {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT key_hash, last_grant_ts, cooldown_until FROM key_clock"
            )
        }
        used = {
            r[0]: (r[1], r[2])
            for r in c.execute(
                "SELECT key_hash, count, banned FROM usage WHERE model=? AND pt_day=?",
                (model, day),
            )
        }
        # ⭐ КРУГ ПОЗИЦИОННЫЙ — база, требование юзера 2026-07-26 дословно: «правка должна
        # делать только одно — правильно отсчитывать ключ в круге, но НЕ менять его позицию
        # в стартовом порядке. 11 ключ всегда 11. И всегда идёт после 1-10 ключей, даже если
        # там кто-то выпадает». Выпавший теряет свой ход и возвращается НА СВОЁ МЕСТО, а не
        # в голову очереди. Это генератор экстрактора (`for api_key, kh in hashed`), развёрнутый
        # в цикл: порядок списка и есть порядок круга, пересортировки нет никакой.
        # ⛔ Чинилось ТОЛЬКО счетоводство: пара round_no (broker_global) + served_round
        # (key_clock) — два числа в разных таблицах, обязанные совпадать. Обрыв прогона
        # посреди оборота замораживал рассинхрон, и следующий прогон шёл огрызком круга
        # (факт 07-24: round_no=749 при ключах с 749/748/745). Заменено ОДНИМ курсором.
        # ⛔ Промежуточный вариант с сортировкой по last_grant_ts (25.07) СНЯТ: у ключа,
        # вышедшего из кулдауна, штамп самый старый, и он влезал в голову очереди — то есть
        # менял позицию, чего делать нельзя. Ловится `pseo/builder/test_order.py`.
        elig, min_cd, cds = [], None, {}
        for idx, k in enumerate(keys):
            kh = _kh(k)
            _last, cd = clocks.get(kh, (0.0, 0.0))
            cnt, ban = used.get(kh, (0, 0))
            if ban or cnt >= cap:
                continue  # RPD/бан — до конца PT-суток мёртв, ждать его бессмысленно
            if cd > now:  # в 429-кулдауне: это ВРЕМЕННО, помним когда вернётся
                min_cd = cd if min_cd is None else min(min_cd, cd)
                cds[idx] = cd  # поимённо: звену нужен СВОЙ минимум, не общий
                continue
            elig.append((idx, k, kh))  # ПОРЯДОК СПИСКА, без пересортировки
        if not elig:
            c.execute("ROLLBACK")
            # ЧЕСТНАЯ РАЗВИЛКА (2026-07-25): «все отдыхают» ≠ «бюджет выбран». Раньше оба
            # случая отдавали -1.0, и call() сдавался НАВСЕГДА при живой квоте — 07-24 так
            # вылетела 31 муха при расходе 6-8 из 440 (combine_logs/1784908198_facet.log).
            if min_cd is not None:
                return (None, max(0.1, min_cd - now))  # подождать и спросить снова
            return (None, -1.0)  # бан/кап у ВСЕХ — сегодня работать правда нечем
        brow = c.execute(
            "SELECT busy_ts, cursor, grp, grp_since FROM broker_global WHERE id=1"
        ).fetchone()
        grp = brow[2] if brow and brow[2] is not None else 0
        grp_since = (brow[3] if brow and brow[3] is not None else 0.0) or 0.0
        # ⭐ ЗВЕНО (замер и обоснование — у GROUP_SIZE): дежурит СРЕЗ списка, не весь пул.
        # Срезы последовательные, поэтому позиционная база цела: 11-й ключ остаётся 11-м,
        # он просто в третьем звене.
        if GROUP_SIZE > 0 and len(keys) > GROUP_SIZE:
            ngroups = (len(keys) + GROUP_SIZE - 1) // GROUP_SIZE
            grp %= ngroups  # пул ужали руками — не вылетать за край списка
            if grp_since <= 0:
                grp_since = now  # первый запуск: звено заступает прямо сейчас
            cycle = GROUP_WORK + GROUP_PAUSE
            phase = now - grp_since
            if phase >= cycle:  # отработало и отдохнуло → смена звена
                grp, grp_since = (grp + 1) % ngroups, now
            elif phase >= GROUP_WORK:  # ПАУЗА: ключей не трогаем вообще, ждём
                c.execute(
                    "UPDATE broker_global SET grp=?, grp_since=? WHERE id=1",
                    (grp, grp_since),
                )
                c.execute("COMMIT")
                return (None, max(0.1, grp_since + cycle - now))
            lo, hi = grp * GROUP_SIZE, min(len(keys), (grp + 1) * GROUP_SIZE)
            g_elig = [e for e in elig if lo <= e[0] < hi]
            if not g_elig:
                g_cd = [cd for i, cd in cds.items() if lo <= i < hi]
                if g_cd:
                    # ЖДЁМ СВОЁ ЗВЕНО, а не убегаем в следующее: убежать = засветить лишние
                    # ключи, то есть сделать ровно то, от чего звенья и заведены.
                    c.execute(
                        "UPDATE broker_global SET grp=?, grp_since=? WHERE id=1",
                        (grp, grp_since),
                    )
                    c.execute("COMMIT")
                    return (None, max(0.1, min(g_cd) - now))
                # Звено мертво НАСОВСЕМ (бан/дневной кап) — ждать его бессмысленно, ход
                # уходит следующему сразу. ⛔ Не возвращать -1.0: у него живые соседи, а
                # -1.0 для рта значит «сдавайся до завтра» (так 24.07 вылетела 31 муха).
                c.execute(
                    "UPDATE broker_global SET grp=?, grp_since=? WHERE id=1",
                    ((grp + 1) % ngroups, now),
                )
                c.execute("COMMIT")
                return (None, 0.0)  # спросить снова сразу, бюджет вызова не тратим
            elig = g_elig
        alive = len(elig)  # годные И ЕСТЬ живые: отсеивать «по обороту» больше нечего
        step = _dyn_grant_step(alive)
        # ОЧЕРЕДЬ ПУЛА: такт на выдачу (ДИНАМИЧЕСКИЙ) — с последней выдачи < step → ждать.
        if brow and now - (brow[0] or 0) < step:
            c.execute("ROLLBACK")
            return (None, 0.0)  # очередь: стоим у кассы, бюджет вызова не тратим
        # Первый годный ПОСЛЕ курсора; никого дальше — замыкаем круг на первого годного
        # вообще. Позиции при этом не двигаются: пропущенный просто не встретился.
        cur = brow[1] if brow and brow[1] is not None else -1
        keyno, key, kh = next((e for e in elig if e[0] > cur), elig[0])
        _TRACE_CTX.clear()  # ТРАССА: контекст гранта (429 допишет call после HTTP)
        _TRACE_CTX.update(
            {
                "t": now,
                "keyno": keyno,
                "alive": alive,
                "step": step,
                "rpd": used.get(kh, (0, 0))[0],
                # суммарный темп ПУЛА за минуту — прямо в строке трассы, чтобы не
                # восстанавливать его потом вручную (индекс ix_log_ts, дёшево)
                "rate60": c.execute(
                    "SELECT COUNT(*) FROM request_log WHERE event='grant' AND ts>?",
                    (now - 60,),
                ).fetchone()[0],
            }
        )
        c.execute(
            "INSERT INTO key_clock(key_hash, last_grant_ts) VALUES(?,?) "
            "ON CONFLICT(key_hash) DO UPDATE SET last_grant_ts=excluded.last_grant_ts",
            (kh, now),
        )
        # usage.count НЕ инкрементим тут: RPD считается по УСПЕХУ (status 200) в report(),
        # а не по гранту. Грант с 429 в дневную квоту Google НЕ идёт (не обслужили) —
        # считать его в кап значило завышать RPD (факт 07-24: наш 440 vs Google 249).
        c.execute(  # per-РОТ счёт дня — по грантам (это НЕ RPD-кап, а расход рта)
            "INSERT INTO consumer_usage(consumer, pt_day, count) VALUES(?,?,1) "
            "ON CONFLICT(consumer, pt_day) DO UPDATE SET count=count+1",
            (consumer, day),
        )
        c.execute(
            "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) VALUES(?,?,?,?,'grant',0)",
            (now, consumer, kh, model),
        )
        c.execute(  # метка выдачи (такт) + курсор круга + чьё звено и когда заступило
            "UPDATE broker_global SET busy_consumer=?, busy_ts=?, cursor=?, "
            "grp=?, grp_since=? WHERE id=1",
            (consumer, now, keyno, grp, grp_since),
        )
        c.execute("COMMIT")
        return (key, None)
    finally:
        c.close()


def report(consumer, key, model, status):
    """Отчёт об исходе — эскалация 429 ПО ОБОРОТАМ ОЧЕРЕДИ (правило юзера 2026-07-21):

    1-й 429  → только МЕТКА (strike), ключ остаётся в очереди. Наказывать сразу не за
               что: следующий его ход всё равно наступит не раньше, чем очередь обойдёт
               все остальные ключи — этого может хватить.
    2-й 429  → тот же ключ отказал СНОВА, дойдя до своего хода, то есть обход не помог
               → cooldown 300с (и 1800с, если он уже сидел в кулдауне раньше).
    успех    → прощение: снимаем и метку, и cooldown, и ступень лестницы.
    давность → серия «рассасывается» сама: тихо прожил дольше только что отсиженного —
               следующий отказ считается ПЕРВЫМ (см. COOLDOWN_LADDER, «давность серии»).
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
                "SELECT cd_level, struck, cooldown_until FROM key_clock WHERE key_hash=?",
                (kh,),
            ).fetchone()
            lvl, struck, cd_end = (
                (row[0] or 0, row[1] or 0, row[2] or 0.0) if row else (0, 0, 0.0)
            )
            # ДАВНОСТЬ СЕРИИ: срок забвения = только что отсиженное (у метки — первая
            # ступень). Прожил тихо дольше — прошлая серия к делу не относится, счёт с нуля.
            grace = COOLDOWN_LADDER[lvl - 1] if lvl else COOLDOWN_LADDER[0]
            if cd_end and now - cd_end > grace:
                lvl, struck = 0, 0
            if (
                lvl == 0 and not struck
            ):  # ПЕРВЫЙ в серии — только метка, ключ остаётся в очереди.
                # cooldown_until=now: наказания нет (в очередь пускают сразу, `cd > now`
                # ложно), но отметка «когда закончилось последнее» есть — по ней считается
                # давность следующего отказа.
                c.execute(
                    "INSERT INTO key_clock(key_hash, cooldown_until, struck) "
                    "VALUES(?,?,1) ON CONFLICT(key_hash) DO UPDATE SET "
                    "cooldown_until=excluded.cooldown_until, struck=1, cd_level=0",
                    (kh, now),
                )
            elif lvl >= len(
                COOLDOWN_LADDER
            ):  # лестница пройдена вся → дневной бан (до PT-полуночи)
                c.execute(
                    "INSERT INTO usage(key_hash, model, pt_day, count, banned) VALUES(?,?,?,0,1) "
                    "ON CONFLICT(key_hash, model, pt_day) DO UPDATE SET banned=1",
                    (kh, model, _pt_day()),
                )
                # событие с КЛЮЧОМ (не через _log_event — там key_hash пустой):
                # по нему пульт шлёт сигнал юзеру, а мы копим статистику причин
                c.execute(
                    "INSERT INTO request_log(ts,consumer,key_hash,model,event,status) "
                    "VALUES(?,?,?,?,'day_ban',429)",
                    (now, consumer, kh, model),
                )
            else:  # отказал снова → следующая ступень (отсидка не прощает)
                lvl += 1
                c.execute(
                    "INSERT INTO key_clock(key_hash, cooldown_until, cd_level, struck) "
                    "VALUES(?,?,?,0) ON CONFLICT(key_hash) DO UPDATE SET "
                    "cooldown_until=excluded.cooldown_until, cd_level=excluded.cd_level, struck=0",
                    (kh, now + COOLDOWN_LADDER[lvl - 1], lvl),
                )
        elif status == 200:  # прощение: метка, кулдаун и история — всё снимается
            c.execute(
                "UPDATE key_clock SET cooldown_until=0, cd_level=0, struck=0 WHERE key_hash=?",
                (kh,),
            )
            # RPD-СЧЁТ: инкремент ТОЛЬКО на успехе (не на гранте) — совпадает с Google-RPD,
            # который считает обслуженные запросы, а не отклонённые 429.
            c.execute(
                "INSERT INTO usage(key_hash, model, pt_day, count) VALUES(?,?,?,1) "
                "ON CONFLICT(key_hash, model, pt_day) DO UPDATE SET count=count+1",
                (kh, model, _pt_day()),
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
        # ДУБЛЬ ДЛЯ КОМБАЙНА: ключи в СВОЁМ env контейнера (Dokploy их туда кладёт).
        # Исходник лез в чужой контейнер `docker exec bots-grab printenv` — в нашем
        # контейнере docker-бинаря нет и не должно быть. Docker-путь оставлен запасным
        # на случай запуска дубля прямо на хосте.
        vals = {k: v for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY")}
        if not vals:
            cid = subprocess.check_output(
                "docker ps --format '{{.Names}}' | grep bots-grab | head -1",
                shell=True,
                text=True,
            ).strip()
            env = subprocess.check_output(
                ["docker", "exec", cid, "printenv"], text=True
            )
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


# ЭТАЛОН extraction.py:164 `_RETRY_DELAYS_S = [5, 20, 60]` — первая попытка без паузы,
# затем ТРИ повтора с этими паузами. Числа не наши, не крутить.
RETRY_DELAYS = (5.0, 20.0, 60.0)
MAX_FAILS = 1 + len(
    RETRY_DELAYS
)  # 4 обращения на вызов (1 + 3 повтора), дальше сдаёмся
MAX_WAIT_TOTAL = 1800.0  # суммарный бюджет ОЖИДАНИЯ слота на 1 вызов (30 мин)
MAX_SLEEP = 30.0  # максимум спим за один «нет слота», потом снова спрашиваем мозг


class _BadShape(ValueError):
    """Ответ распарсился, но пришёл не JSON-ОБЪЕКТ (обычно массив). Отдельный тип нужен,
    чтобы в логе отличать «не та форма» (`parse_shape`) от «мусор» (`parse_junk`): это
    разные болезни, а писались они одним именем `parse_fail` и в базе не различались."""


# ⭐ СПАСЕНИЕ ЗАПИСЕЙ (заказ юзера 2026-08-20). Ответ рта на пачку — один большой JSON, и
# одна незакрытая кавычка внутри перевода уносила ВСЮ пачку: 25 мух за один символ. Замер
# по логу: 42 таких случая, из них 31 на переводе.
#
# Кавычки не наша забота — их ломает рот, запретить ему нельзя. Наша — чтобы порча одной
# записи стоила одну муху. Поэтому на битом ответе режем текст по БАЛАНСУ ФИГУРНЫХ СКОБОК и
# разбираем каждый кусок сам по себе. Скобки для этого годятся: замер по 26 241 тексту
# корпуса — `{` и `}` не встречаются НИ РАЗУ, придумывать свои метки не нужно.
def salvage_objects(raw, need):
    """Куски `{...}` из битого текста → список записей, у которых есть ключ `need`."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                chunk = raw[start : i + 1]
                try:
                    rec = json.loads(chunk)
                except Exception:
                    rec = None
                if isinstance(rec, dict) and need in rec:
                    out.append(rec)
                elif "{" in chunk[1:-1]:
                    # ⛔ Внутрь лезем И КОГДА КУСОК НЕ РАЗОБРАЛСЯ: записи лежат в обёртке
                    # `{"rows": [...]}`, а битая кавычка ломает именно обёртку. Первая версия
                    # рекурсию делала только на РАЗОБРАННОЙ обёртке — то есть ровно в том
                    # случае, когда спасать нечего. Сторож это и поймал.
                    out.extend(salvage_objects(chunk[1:-1], need))
                start = None
    return out


def call(
    user,
    sysprompt,
    consumer,
    model="gemini-3.1-flash-lite",
    role="background",
    timeout=60,
    salvage=None,
):
    """СОСОК. Рот шлёт (user, sysprompt) + называет себя (consumer) и молоко (model);
    получает dict или None. Ключа рот НЕ видит — он живёт и умирает здесь.
    acquire → HTTP → report (в finally: выйти без учёта негде) → разбор JSON.

    WORST-CASE НА ВЫЗОВ: 4 реальных запроса к Google (1 + 3 повтора по эталону) +
    ≤MAX_WAIT_TOTAL(30мин) сна в ожидании СЛОТА КЛЮЧА + ≤85с пауз между повторами (5+20+60).
    Очередь пула (такт выдачи) ждётся ОТДЕЛЬНО и бюджет не тратит — из
    очереди не уходят; полёт (HTTP) пул не держит — блокировки как класса нет.
    None: бюджет выбран / MAX_FAILS неудач / 30мин без слота ключа.

    ⭐ НЕУДАЧА — это и 200 с непарсящимся телом (2026-07-27). Раньше такой ответ давал
    мгновенный None без единого повтора, и один испорченный ответ модели стоил дороже
    целой серии отказов пула: 27.07 он отнял нарезку у 37 мух и поставил гео на перепрогон.
    Потолок запросов на вызов от этого НЕ вырос — парс-фейл тратит ту же квоту MAX_FAILS.
    """
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": sysprompt}]},
        "generationConfig": {"responseMimeType": "application/json"},
    }
    data = json.dumps(payload).encode()
    keys = get_keys()
    # СКВОЗНОЙ НОМЕР ЛОГИЧЕСКОГО ВЫЗОВА + номер попытки внутри него. Без них `сдались 4/4`
    # не отличает «одна муха провалилась на 4 РАЗНЫХ ключах» (виноват запрос) от «4 разные
    # мухи по разу» (виноват пул) — а это два совершенно разных диагноза.
    _CALL_SEQ[0] += 1
    call_no, attempt = _CALL_SEQ[0], 0
    chars = len(user) + len(sysprompt)
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
        body, status, err, hdrs = None, 0, "", None
        attempt += 1
        t_send = time.time()
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            body = r.read()
            status = 200
            hdrs = r.headers  # РАЗВЕДКА квота-заголовков (см. _log_hdrs)
        except urllib.error.HTTPError as e:
            status = e.code
            hdrs = e.headers
            try:
                err = e.read().decode("utf-8", "replace")[:800]
            except Exception:
                err = str(e)[:120]
            try:  # СЕТЬ: заголовок называет окно квоты (60=поминутка, больше=жёсткая)
                err = f"retry-after={e.headers.get('Retry-After', '-')} {err}"
            except Exception:
                pass
        except Exception as e:
            status = -1  # сеть/таймаут — ключ не виноват
            err = str(e)[:120]
        finally:
            ms = int((time.time() - t_send) * 1000)
            report(consumer, key, model, status)  # ← выйти без учёта НЕГДЕ
        toks = 0
        if body:  # токены СЧИТАЕТ САМ GOOGLE — берём его число, не свою оценку
            try:
                toks = (json.loads(body).get("usageMetadata") or {}).get(
                    "totalTokenCount", 0
                )
            except Exception:
                pass
        _trace_row(status, call_no, attempt, chars, toks, ms)
        if hdrs is not None:  # шлёт ли Google остаток RPD в заголовках — узнаём фактом
            _log_hdrs(consumer, model, status, hdrs)

        if status != 200:
            # СЕТЬ: pid+ключ на каждом отказе — два разных pid = нахлёст процессов
            err = f"[pid={os.getpid()} kh={_kh(key)[:8]}] {err}"
            _log_body(
                consumer, model, status, err
            )  # тело в error_bodies.log — диагностика причины

        if status == 200:
            raw = ""  # до try: в except он нужен для лога, а присваивается ВНУТРИ
            try:
                api = json.loads(body)
                raw = api["candidates"][0]["content"]["parts"][0]["text"]
                # LLM РЕГУЛЯРНО лепит хвост после объекта (второй JSON, пояснение) →
                # строгий json.loads падает 'Extra data'. Берём ПЕРВЫЙ валидный объект
                # через raw_decode, хвост игнорируем (факт 07-24: Extra data ронял carve).
                parsed, _ = json.JSONDecoder().raw_decode(
                    re.sub(r"```json|```", "", raw).strip()
                )
                # КОНТРАКТ: все рты просят JSON-ОБЪЕКТ. Модель иногда отдаёт массив
                # `[...]` — валидный JSON, но не dict → у потребителя `.items()` = краш
                # (факт 07-22: 'list' object has no attribute 'items' рушил перевод гео).
                # Не-dict = такой же брак ответа, как мусор → в общую ветку повтора ниже.
                if not isinstance(parsed, dict):
                    raise _BadShape("не dict, а " + type(parsed).__name__)
                return parsed
            except Exception as e:
                # ⭐ СПАСЕНИЕ вместо потери пачки: рот просил записи объектами, значит из
                # битого тела достаём уцелевшие. Хоть одна нашлась — отдаём их и НЕ повторяем:
                # повтор стоил бы вызова, а потеря — одной записи.
                if salvage:
                    recs = salvage_objects(raw, salvage[1])
                    if recs:
                        print(
                            f"  спасено записей: {len(recs)} (тело битое: {str(e)[:60]})",
                            flush=True,
                        )
                        return {salvage[0]: recs}
                # ⭐ ПАРС-ФЕЙЛ = ОБЫЧНАЯ НЕУДАЧА, А НЕ КОНЕЦ ВЫЗОВА (юзер 2026-07-27).
                # Было: любой 200-с-мусором → мгновенный None БЕЗ единого повтора, хотя на
                # инфраструктурные сбои у вызова бюджет в 4 попытки. Цена реальная: 27.07
                # один парс-фейл в 10:03 отнял нарезку у 37 мух и поставил гео vn на
                # перепрогон — при НУЛЕ отказов пула в том окне. Модель на повторе (другой
                # ключ, другой сэмпл) обычно отдаёт валидный ответ, и лишний запрос дешевле
                # потерянной работы.
                # ⛔ Ключ при этом НЕ наказывается: 200 пришёл, ключ отработал — виноват
                # ответ модели. Наказывать значило бы гасить здоровые ключи за чужую вину.
                kind = "parse_shape" if isinstance(e, _BadShape) else "parse_junk"
                print(f"  parse err ({kind}):", str(e)[:100])
                # ДВА РАЗНЫХ ИМЕНИ: «не та форма» и «мусор» — разные болезни, а писались
                # одним `parse_fail`, и в базе их было не отличить.
                _log_event(consumer, model, kind)
                # ТЕЛО В ЛОГ: _log_body зовётся только на не-200, поэтому что именно
                # присылает модель, мы не видели НИ РАЗУ — разбирали вслепую.
                _log_body(consumer, model, 200, f"[{kind}] {(raw or '')[:400]}")
                fails += 1
                continue
        fails += 1
        if status == 429:
            # Ключ ушёл в cooldown (report) — берём СЛЕДУЮЩЕГО ПО КРУГУ СРАЗУ, без паузы:
            # следующий ключ к отказу текущего отношения не имеет. Темп держит не пауза
            # здесь, а ПАУЗА НА ЗАКРЫТИИ КРУГА в acquire (эталон extraction: «одна модель —
            # по всем ключам — потом sleep 60s»). Прежний нарастающий backoff 5→10→20→40
            # был остатком теории «общей волны» — удалён 2026-07-21.
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
