# -*- coding: utf-8 -*-
"""Комбайн-пульт: отдельный ТГ-бот + исполнитель прогонов В ОДНОМ контейнере.

Канон (принят юзером 2026-07-21, memory roadmap_combine_tg_pult):
  - запуск/стоп/статус — ТОЛЬКО юзер из ТГ (whitelist chat_id); одна задача за раз;
  - отчёт на каждые 50 попыток мозга (RPD жгут попытки) + прогресс + кнопка ⛔ СТОП;
  - финальный отчёт при ЛЮБОМ исходе — молчаливых смертей нет;
  - исходный код НЕ трогается: рты бегут ДУБЛЯМИ из /app/builder, данные — через
    маунт BRAIN_DIR (/brain = /root/pseo_builder хоста): keybroker.db, out_facet и пр.

ENV: COMBINE_BOT_TOKEN, ADMIN_ID, BRAIN_DIR, GEMINI_API_KEY_N (ртам).
Запуск процесса = запись в jobs (audit) → subprocess дубля. Нет команды — нет процесса.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import requests

TOKEN = os.environ["COMBINE_BOT_TOKEN"]
# ADMIN_ID — соглашение проекта (личный telegram-id юзера). COMBINE_CHAT_ID — старое
# имя, которое я завёл зря; читаем оба, чтобы уже настроенный сервис не сломался.
CHAT = int(os.environ.get("ADMIN_ID") or os.environ["COMBINE_CHAT_ID"])
# Данные монтируются в контейнер ПО ТЕМ ЖЕ путям, что на хосте (/root/pseo_builder,
# /home/teledigest/data, /root/embed_ab) — дубли ртов несут абсолютные пути, не правим их.
BRAIN = os.environ.get("BRAIN_DIR", "/root/pseo_builder")
BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builder")
API = f"https://api.telegram.org/bot{TOKEN}"
REPORT_EVERY = int(os.environ.get("COMBINE_REPORT_EVERY", "50"))  # попыток мозга
# сколько ждём вежливого выхода рта после стоп-флага (один вызов к Gemini + запись;
# у lang_runner шаг крупнее — гео×язык, потому не секунды)
GRACE_S = int(os.environ.get("COMBINE_GRACE_S", "180"))
# сколько раз перепробовать ФАЗУ при неудаче внутри шага, прежде чем звать юзера
MAX_PHASE_TRIES = int(os.environ.get("COMBINE_PHASE_TRIES", "3"))
LIVE_EVERY = int(
    os.environ.get("COMBINE_LIVE_EVERY", "25")
)  # сек между правками живой строки
JOBS_DB = os.path.join(BRAIN, "combine_jobs.db")
KB_DB = os.path.join(BRAIN, "keybroker.db")
# два флага: facet-рты чтут RUNNER_STOP, lang_runner — LANG_RUNNER_STOP
STOP_FLAGS = [
    os.path.join(BRAIN, "RUNNER_STOP"),
    os.path.join(BRAIN, "LANG_RUNNER_STOP"),
]

# Меню: kind → (кнопка, argv дубля; {geo} подставляется). cwd=BRAIN — данные хоста.
# Версия таксономии — из ТОГО ЖЕ модуля, по которому раскладывают рты. Литералом нельзя:
# на копиях чисел этот проект уже горел (DEAD_AT разъехался по трём файлам).
sys.path.insert(0, BUILDER)
import tail_taxonomy as _tax  # noqa: E402

_TAX_VERSION = _tax.VERSION
_TAX_NAMES = set(_tax.SHELF_NAMES)


MENU = {
    # ШАГ 3. Разметка: муха → перевод + тема из 13 + подтема. Пачка 25, добирает неразмеченных.
    # «гео» или «гео:сколько» — второе для пробного куска.
    "mark": (
        "Разметка <гео[:сколько]>",
        ["python", "-u", f"{BUILDER}/tract.py", "{geo}", "--mark", "{shelf}"],
    ),
    # ШАГ 4. Списки: мухи темы → списки с именами → страницы и остаток. Пачка 90.
    "svod": (
        "Списки <гео>",
        ["python", "-u", f"{BUILDER}/tract.py", "{geo}", "--svod"],
    ),
    # ШАГ 5. Сборка и рендер всего дерева — код, ключей не тратит.
    "build": ("Сборка сайта", ["python", "-u", f"{BUILDER}/../render.py", "--all"]),
    # ШАГ 6. Переводы на 13 языков.
    "translate": ("Переводы (очередь)", ["python", "-u", f"{BUILDER}/lang_runner.py"]),
    # АДРЕСА страниц: ДО переводов (перевод несёт `key` из русского файла).
    "stamp": (
        "Адреса страниц <гео>",
        ["python", "-u", f"{BUILDER}/facet_lang.py", "--stamp-keys", "{geo}"],
    ),
}


def log(*a):
    """Всё, что делает пульт — в stdout контейнера (вкладка Logs в Dokploy)."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


_T0 = time.time()  # старт процесса — для окна пересменки реплик
_conflict = {"since": 0.0, "said": 0.0}


def _conflict_log(desc):
    """Conflict при редеплое = штатная пересменка реплик (swarm держит старую, пока
    поднимает новую) — молчим. Орём, только если он НЕ проходит: тогда это реальный
    второй едок токена. Шум, который приучаешься игнорировать, прячет настоящие сбои.
    """
    now = time.time()
    if not _conflict["since"]:
        _conflict["since"] = now
    tail = now - _conflict["since"]
    if now - _T0 < 90 and tail < 90:
        return  # окно пересменки после старта — норма, не шумим
    if now - _conflict["said"] > 300:  # затянулся → сигнал, но не поток
        _conflict["said"] = now
        log(
            f"⚠️ Conflict уже {tail / 60:.0f} мин — похоже, токен опрашивает кто-то ещё: {desc}"
        )


def tg(method, **kw):
    try:
        r = requests.post(f"{API}/{method}", json=kw, timeout=35).json()
        if not r.get("ok", True):
            desc = r.get("description") or ""
            if "Conflict" in desc:
                _conflict_log(desc)
            else:
                log("TG-ОШИБКА", method, desc)
        elif _conflict["since"]:
            _conflict["since"] = _conflict["said"] = 0.0  # разошлись — забываем
        return r
    except Exception as e:
        log("TG-СБОЙ", method, type(e).__name__, e)
        return {}


_STOP_KB = {"inline_keyboard": [[{"text": "⛔ СТОП", "callback_data": "stop"}]]}


def say(text, stop_btn=False):
    """Отправить сообщение. Возвращает message_id — чтобы потом ПРАВИТЬ его же."""
    kw = {"chat_id": CHAT, "text": text}
    if stop_btn:
        kw["reply_markup"] = _STOP_KB
    log("→ЮЗЕРУ:", text.replace("\n", " | ")[:200])
    r = tg("sendMessage", **kw)
    return ((r or {}).get("result") or {}).get("message_id")


_last_edit = {}  # msg_id → последний отправленный текст (не долбить одинаковым)


def edit(msg_id, text, stop_btn=True):
    """Живой прогресс = ПРАВКА одного сообщения, не поток новых (юзер не должен сидеть
    в докплой-логах, но и спамить чат раз в 3 секунды нельзя).
    ⚠️ Telegram отклоняет правку ТЕМ ЖЕ текстом ('message is not modified') — если рот
    час стоит на одном гео, tail не меняется, и мы флудили сотнями ошибок (факт 07-22).
    Пропускаем правку, когда текст не изменился."""
    if not msg_id or _last_edit.get(msg_id) == text:
        return
    _last_edit[msg_id] = text
    kw = {"chat_id": CHAT, "message_id": msg_id, "text": text}
    if stop_btn:
        kw["reply_markup"] = _STOP_KB
    tg("editMessageText", **kw)


def jobs_conn():
    c = sqlite3.connect(JOBS_DB, timeout=30)
    c.execute(
        "CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL, kind TEXT, args TEXT, status TEXT, note TEXT)"
    )
    return c


def close_interrupted_jobs():
    """Рестарт контейнера убивает рот вместе с PID-namespace (пульт = PID 1), но строка
    в jobs остаётся 'running' навсегда, и юзеру никто не говорит — а канон обещает
    «финальный отчёт при ЛЮБОМ исходе, молчаливых смертей нет». Закрываем на старте и
    докладываем. Возвращает список оборванных (kind, ts).
    """
    try:
        c = jobs_conn()
        rows = c.execute(
            "SELECT id, kind, ts FROM jobs WHERE status='running' ORDER BY id"
        ).fetchall()
        if rows:
            c.execute(
                "UPDATE jobs SET status='interrupted', note=? WHERE status='running'",
                ("оборван рестартом пульта",),
            )
            c.commit()
        c.close()
        return [(k, ts) for _, k, ts in rows]
    except Exception as e:
        log("close_interrupted_jobs сбой:", type(e).__name__, e)
        return []


def kb():
    return sqlite3.connect(f"file:{KB_DB}?mode=ro", uri=True, timeout=30)


def pt_day():
    r = subprocess.run(
        ["date", "+%F"],
        env={**os.environ, "TZ": "America/Los_Angeles"},
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def ban_watch():
    """СИГНАЛ О ДНЕВНОМ БАНЕ КЛЮЧА (заказ юзера 2026-07-21).

    Ключ садится в бан только пройдя ВСЮ лестницу (60/300/1800/6000) — это редкое и
    важное событие: значит ключ не ожил за ~2.3 часа. Шлём в ТГ разбор: сколько он
    сегодня отработал, сколько словил 429, что Google писал в теле — для статистики
    причин (тело у Google немое, копим наблюдения).
    """
    seen = time.time()  # стартуем «с этого момента», старое не переигрываем
    while True:
        time.sleep(30)
        try:
            c = kb()
            rows = c.execute(
                "SELECT ts, consumer, key_hash, model FROM request_log "
                "WHERE event='day_ban' AND ts>? ORDER BY ts",
                (seen,),
            ).fetchall()
            for ts, cons, kh, mdl in rows:
                seen = max(seen, ts)
                cnt = (
                    c.execute(
                        "SELECT count FROM usage WHERE key_hash=? AND model=? AND pt_day=?",
                        (kh, mdl, pt_day()),
                    ).fetchone()
                    or [0]
                )[0]
                n429 = c.execute(
                    "SELECT COUNT(*) FROM request_log WHERE key_hash=? AND status=429 "
                    "AND ts>?",
                    (kh, ts - 86400),
                ).fetchone()[0]
                bodies = ""
                try:  # последнее, что Google сказал — вдруг там не пустой RESOURCE_EXHAUSTED
                    with open(f"{BRAIN}/error_bodies.log", encoding="utf-8") as f:
                        tail = [x for x in f.readlines()[-40:] if "\t429\t" in x]
                    if tail:
                        bodies = tail[-1].split("\t")[-1].strip()[:300]
                except Exception:
                    pass
                say(
                    f"⛔ КЛЮЧ {kh[:8]} В ДНЕВНОМ БАНЕ (прошёл всю лестницу отдыха)\n"
                    f"рот: {cons} | модель: {mdl}\n"
                    f"за сегодня попыток на ключе: {cnt} | 429 за сутки: {n429}\n"
                    f"последнее тело 429: {bodies or '—'}\n"
                    f"вернётся сам со сменой PT-дня (~10:00 МСК)."
                )
                log(f"СИГНАЛ: дневной бан ключа {kh[:8]} (рот {cons})")
            c.close()
        except Exception as e:
            log("ban_watch сбой:", type(e).__name__, e)


def brain_stats():
    """Снимок мозга за PT-день: попытки всего/по ртам, макс-ключ, 429."""
    day = pt_day()
    try:
        c = kb()
        total = c.execute(
            "SELECT COALESCE(SUM(count),0) FROM usage WHERE pt_day=?", (day,)
        ).fetchone()[0]
        mouths = c.execute(
            "SELECT consumer, count FROM consumer_usage WHERE pt_day=? ORDER BY count DESC",
            (day,),
        ).fetchall()
        kmax = c.execute(
            "SELECT COALESCE(MAX(count),0) FROM usage WHERE pt_day=?", (day,)
        ).fetchone()[0]
        n429 = c.execute(
            "SELECT COUNT(*) FROM request_log WHERE status=429 AND ts>?",
            (time.time() - 3600,),
        ).fetchone()[0]
        c.close()
        return total, mouths, kmax, n429
    except Exception as e:
        return 0, [], 0, f"?({e})"


LANGS = ["en", "es", "pt", "de", "fr", "it", "zh", "ja", "ko", "ar", "hi", "th", "tr"]

# Широкие рубрики-категории экстрактора (исходные метки мух). facet-фолбэк при сбое carve
# называет вид ЭТИМИ рубриками вместо конкретных подпунктов — по ним и распознаём откат в
# СТАРЫХ данных (без метки fails). Список — эвристика: промах = гео не подсветится, не беда
# (новый facet запишет fails и подсветит точно). Не жёсткая логика, а подсказка для ремонта.
_RUBRICS = {
    "Документы и виза",
    "Транспорт",
    "Жильё",
    "Банк и деньги",
    "Здоровье",
    "Обмен и переводы",
    "Билеты и развлечения",
    "Безопасность",
    "Покупки",
    "SIM и интернет",
    "Работа и налоги",
    "Связь и интернет",
    "Еда и рестораны",
}


def pipeline_state():
    """ЧТО НЕ СДЕЛАНО по тракту «тема и подтема» (§0.19) — считается из данных, не из памяти.

    Порядок: разметка (мухи → перевод, тема, подтема) → списки (мухи темы → страницы) →
    адреса → сборка → переводы. Работа шага = НЕСДЕЛАННОЕ, поэтому шаг с ✅ и есть готовый.
    """
    import glob
    import sys as _sys

    st = {"mark": [], "mark_n": 0, "svod": [], "geos": 0, "views": 0, "langs": []}
    if BUILDER not in _sys.path:
        _sys.path.insert(0, BUILDER)
    try:
        import facet as _facet
    except Exception as e:  # без билдера состояние не посчитать — честно скажем
        st["error"] = str(e)
        return st

    # ── что уже размечено, по гео ──────────────────────────────────────────────────────
    tagged = {}
    for fn in sorted(glob.glob(f"{BRAIN}/tags/*.json")):
        geo = os.path.basename(fn)[:-5]
        if geo.endswith("_fails"):  # файл сбоев — не гео
            continue
        try:
            tagged[geo] = {r["id"] for r in json.load(open(fn, encoding="utf-8"))}
        except Exception:
            tagged[geo] = set()

    # ── мухи в базе минус размеченные = работа шага разметки ───────────────────────────
    try:
        m = sqlite3.connect(f"file:{_facet.DB}?mode=ro", uri=True, timeout=30)
        rows = m.execute(
            "SELECT country, id, ai_lesson FROM extracted_patterns "
            "WHERE country IS NOT NULL AND ai_lesson IS NOT NULL AND length(ai_lesson)>?",
            (_facet.MIN_LEN,),
        ).fetchall()
        m.close()
        by_geo = {}
        for country, fid, lesson in rows:
            if _facet.is_junk(lesson):
                continue
            for g in _facet.geo_codes(country):
                by_geo.setdefault(g, set()).add(fid)
        for g, ids in sorted(by_geo.items(), key=lambda kv: -len(kv[1])):
            left = len(ids - tagged.get(g, set()))
            if left:
                st["mark"].append({"geo": g, "n": left})
                st["mark_n"] += left
    except Exception as e:
        st["error"] = f"база мух недоступна: {e}"

    # ── списки: разметка есть, а корпус старше её или отсутствует ──────────────────────
    for geo, ids in sorted(tagged.items()):
        if not ids:
            continue
        corpus = f"{BRAIN}/out_facet/{geo}.json"
        tags_fn = f"{BRAIN}/tags/{geo}.json"
        if not os.path.exists(corpus) or os.path.getmtime(corpus) < os.path.getmtime(
            tags_fn
        ):
            st["svod"].append(geo)

    # ── корпус: сколько гео и страниц уже собрано ──────────────────────────────────────
    for fn in sorted(glob.glob(f"{BRAIN}/out_facet/*.json")):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        st["geos"] += 1
        st["views"] += len(d.get("views_by_task") or [])
    return st


def state_card():
    """Карточка состояния: что есть и что просрочено. Без выводов, только числа."""
    s = pipeline_state()
    if s.get("error"):
        return f"⚠️ {s['error']}", None
    lines = [f"📦 корпус: {s['geos']} гео, {s['views']} страниц"]
    if s["mark"]:
        worst = ", ".join(f"{x['geo']}({x['n']})" for x in s["mark"][:6])
        lines.append(f"1) разметка: {len(s['mark'])} гео, {s['mark_n']} мух — {worst}")
    if s["svod"]:
        lines.append(f"2) списки: {len(s['svod'])} гео — {', '.join(s['svod'][:6])}")
    if not s["mark"] and not s["svod"]:
        lines.append("всё разобрано: можно собирать сайт")
    return chr(10).join(lines), None


def pipeline_steps(s):
    """Шаги В ПОРЯДКЕ ИСПОЛНЕНИЯ: [{kind, jobs, label}]. Пустой jobs = делать нечего."""
    return [
        {
            "kind": "mark",
            "jobs": [("mark", x["geo"]) for x in s["mark"]],
            "label": (
                f"1. Разметка — {len(s['mark'])} гео, {s['mark_n']} мух"
                if s["mark"]
                else "1. Разметка"
            ),
            "note": "",
        },
        {
            "kind": "svod",
            "jobs": [("svod", g) for g in s["svod"]],
            "label": (
                f"2. Списки — {len(s['svod'])} гео" if s["svod"] else "2. Списки"
            ),
            "note": "",
        },
        {
            "kind": "build",
            "jobs": [("build", None)] if s["geos"] else [],
            "label": f"3. Сборка сайта — {s['views']} страниц",
            "note": "",
        },
        {
            "kind": "translate",
            "jobs": [],
            "label": "4. Переводы",
            "note": "запускать после сборки",
        },
    ]


def facet_queue(s):
    """Гео шага разметки в порядке исполнения — для строк-кнопок под шагом."""
    return [{"geo": x["geo"], "n": x["n"]} for x in s["mark"]]


class Job:
    """Одна задача = один subprocess дубля. Глобально не больше одной."""

    def __init__(self):
        self.proc = None
        self.kind = None
        self.geo = None
        self.job_id = None  # id своей строки в jobs (не MAX(id) — тот уедет в чужую)
        self.t0 = 0.0
        self.base_attempts = 0
        self.last_report_at = 0
        self.tail = ""
        self.live_msg = None  # id сообщения с живым прогрессом (правим его же)
        self.lock = threading.Lock()
        self.chain = []  # очередь шагов полного цикла: [(kind, geo), ...]
        self.tries = {}  # (шаг, гео) → сколько раз уже пробовали в этой фазе цикла

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind, geo=None, _chain=False):
        with self.lock:
            if self.busy():
                say(f"занято: {self.kind} уже бежит. Сначала ⛔ СТОП.")
                return
            if not _chain:
                self.chain = []  # ручной запуск отменяет недобеганную цепочку
            # Пара «гео:аргумент» (например `mark cz:50`) — второе значение необязательное.
            # ⛔ Нет пары — аргумент просто ВЫКИДЫВАЕМ из команды. Раньше здесь стоял поиск
            # устаревшей полки, и `mark cz` уходил в него: пульт отвечал «полок из старой
            # таксономии нет» вместо запуска. Ветка умерла вместе со своей кнопкой.
            arg = None
            if geo and ":" in geo:
                geo, arg = geo.split(":", 1)
            argv = [a.replace("{geo}", geo or "") for a in MENU[kind][1]]
            if arg:
                argv = [a.replace("{shelf}", arg) for a in argv]
            else:
                argv = [a for a in argv if "{shelf}" not in a]
            for f in STOP_FLAGS:  # прошлый стоп не должен глушить новый заказ
                if os.path.exists(f):
                    os.remove(f)
            j = jobs_conn()
            cur = j.execute(
                "INSERT INTO jobs(ts,kind,args,status) VALUES(?,?,?,?)",
                (time.time(), kind, json.dumps(argv), "running"),
            )
            # ⭐ СВОЯ строка журнала (2026-07-25): раньше исход писали по MAX(id), а к тому
            # моменту следующий шаг цепочки/ретрая уже вставил СВОЮ — и результат уезжал в
            # чужую запись. Журнал врал ровно там, где нужен для разбора.
            self.job_id = cur.lastrowid
            j.commit()
            j.close()
            self.kind, self.geo, self.t0 = kind, geo, time.time()
            self.base_attempts = brain_stats()[0]
            self.last_report_at = 0
            self.tail = ""
            # полный лог задачи: BRAIN/combine_logs/<ts>_<kind>.log (переживает контейнер)
            os.makedirs(os.path.join(BRAIN, "combine_logs"), exist_ok=True)
            self.logpath = os.path.join(
                BRAIN, "combine_logs", f"{int(self.t0)}_{kind}.log"
            )
            self.proc = subprocess.Popen(
                argv,
                cwd=BRAIN,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            log(f"ЗАПУСК {kind} pid={self.proc.pid} argv={argv} лог={self.logpath}")
            # ⭐ СНИМОК СВОЕЙ ЗАДАЧИ (2026-07-25): _pump работает с ЛОКАЛЬНЫМИ копиями, а не
            # с self.*. Иначе гонка: рот вышел → busy() уже False → юзер жмёт кнопку →
            # start() подменяет self.proc/kind/job_id → старый _pump дожидается ЧУЖОГО
            # процесса и отчитывается за него.
            threading.Thread(
                target=self._pump,
                args=(
                    self.proc,
                    self.job_id,
                    kind,
                    geo,
                    self.logpath,
                    self.base_attempts,
                ),
                daemon=True,
            ).start()
            self.live_msg = say(  # это сообщение будем ПРАВИТЬ живым прогрессом
                f"▶️ пошёл: {kind}" + (f" ({geo})" if geo else "") + "\nразогрев…",
                stop_btn=True,
            )

    def _pump(self, proc, job_id, kind, geo, logpath, base_attempts):
        t0 = time.time()
        with open(logpath, "a", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)  # полный вывод рта — в файл лога
                lf.flush()
                print(f"[{kind}] {line}", end="")  # и в docker logs
                line = line.strip()
                if line:
                    self.tail = line  # последняя строка — в отчёты (живая, общая)
        rc = proc.wait()
        spent = brain_stats()[0] - base_attempts
        mins = (time.time() - t0) / 60
        # ЗНАЧОК ПО СУТИ, А НЕ ПО КОДУ ВЫХОДА: процесс с откатом внутри выходит с кодом 0
        # («сделал что мог»), и зелёная галочка врала бы — брак виден только в тексте.
        # Есть «НЕУДАЧ» в выводе → ⚠️, чтобы в ленте цикла брак было видно значком.
        icon = "💀" if rc != 0 else ("⚠️" if "НЕУДАЧ" in (self.tail or "") else "✅")
        j = jobs_conn()
        j.execute(
            "UPDATE jobs SET status=?, note=? WHERE id=?",
            (f"exit={rc}", self.tail[-300:], job_id),
        )
        j.commit()
        j.close()
        say(
            f"{icon} {kind}: код {rc}, {mins:.0f} мин, попыток ~{spent}\n"
            f"последнее: {self.tail[-300:] or '—'}\n"
            f"лог: {logpath}"
        )
        # ⚠️ НЕУДАЧА ВНУТРИ ШАГА (carve не разобрал → гео собрано откатом). Процесс вышел
        # с кодом 0 — по коду не отличить. Смотрим вывод: провал транзиентный (модель не
        # ответила), поэтому ПЕРЕПРОБУЕМ ЭТУ ЖЕ ФАЗУ, до MAX_PHASE_TRIES. Не рвём цикл на
        # случайности (юзер: «глупо рвать»), но и не наслаиваем kratko/переводы на брак.
        stopped = any(os.path.exists(f) for f in STOP_FLAGS)
        if rc == 0 and "НЕУДАЧ" not in (self.tail or ""):
            # ⭐ УСПЕХ ПРОЩАЕТ (2026-07-25). Счётчик попыток жил до конца жизни пульта:
            # гео споткнулось дважды утром, вечером прошло, потом споткнулось ОДИН раз —
            # и сразу «не прошёл 3 раза подряд, цикл остановлен». Считать надо ПОДРЯД
            # идущие неудачи, а чистый проход обрывает серию (та же болезнь, что лечили
            # в лестнице ключей: наказание переживало эпизод).
            self.tries.pop((kind, geo), None)
        if "НЕУДАЧ" in (self.tail or "") and not stopped:
            key = (kind, geo)
            self.tries[key] = self.tries.get(key, 1) + 1
            if self.tries[key] <= MAX_PHASE_TRIES:
                say(
                    f"⚠️ {kind}"
                    + (f" ({geo})" if geo else "")
                    + f": неудача внутри шага — попытка {self.tries[key]}/{MAX_PHASE_TRIES}\n"
                    f"{self.tail[-200:]}"
                )
                self.start(kind, geo, _chain=bool(self.chain))
                return
            # три раза подряд — это не случайность. СТОП-МАШИНА с разбором.
            n = len(self.chain)
            self.chain = []
            card, _ = state_card()
            tg(
                "sendMessage",
                chat_id=CHAT,
                text=(
                    f"🛑 СТОП: {kind}"
                    + (f" ({geo})" if geo else "")
                    + f" не прошёл {MAX_PHASE_TRIES} раза подряд.\n\n"
                    f"ЧТО СЛУЧИЛОСЬ: {self.tail[-250:]}\n\n"
                    f"ЦИКЛ ОСТАНОВЛЕН (пропущено шагов: {n}) — дальше не идём, чтобы не "
                    f"класть короткие ответы и переводы поверх неразобранного.\n\n"
                    f"ЧТО ДЕЛАТЬ: посмотреть лог {logpath}; если модель молчит — "
                    f"подождать и нажать ремонт вручную; если стабильно — смотреть данные гео.\n\n"
                    f"{card}"
                ),
                reply_markup={
                    "inline_keyboard": [
                        [
                            {
                                "text": f"🔧 повторить {kind}"
                                + (f" {geo}" if geo else ""),
                                "callback_data": f"run:{kind}"
                                + (f":{geo}" if geo else ""),
                            }
                        ],
                        [{"text": "☰ меню", "callback_data": "menu"}],
                    ]
                },
            )
            return
        # цепочка полного цикла: следующий шаг только если предыдущий вышел чисто
        # и стоп не нажат (нажатый стоп = юзер сказал «хватит», цепочка рвётся)
        if self.chain and rc == 0 and not any(os.path.exists(f) for f in STOP_FLAGS):
            kind, geo = self.chain.pop(0)
            say(f"⛓ цикл: следующий шаг — {kind}" + (f" ({geo})" if geo else ""))
            self.start(kind, geo, _chain=True)
            return
        if self.chain:
            n = len(self.chain)
            self.chain = []
            say(f"⛓ цикл прерван: осталось {n} шагов. Запусти заново, когда решишь.")
            return
        # ⭐ ОДИНОЧНАЯ ЗАДАЧА ЗАВЕРШЕНА → показать НОВОЕ состояние тракта + что дальше,
        # с кнопками. Иначе после «готово» юзер не знает, что делать (его прямая жалоба).
        card, _ = state_card()
        # «Что дальше» берём из ТОЙ ЖЕ вертикали, что и меню, — иначе после задачи пульт
        # звал бы один шаг, а меню показывало другой (в прежней версии 'failed' вообще
        # не был ротом из MENU и требовал отдельной ветки).
        nxt_step = next(
            (st for st in pipeline_steps(pipeline_state()) if st["jobs"]), None
        )
        if nxt_step:
            rows = [
                [
                    {
                        "text": f"➡️ дальше: {nxt_step['label']}",
                        "callback_data": f"run:{nxt_step['kind']}",
                    }
                ]
            ]
            rows.append([{"text": "☰ меню", "callback_data": "menu"}])
            tg(
                "sendMessage",
                chat_id=CHAT,
                text="📊 после задачи:\n" + card,
                reply_markup={"inline_keyboard": rows},
            )
        else:
            say(
                "📊 после задачи:\n"
                + card
                + "\n\n🎉 тракт готов — можно шипить (ship с десктопа)."
            )

    def stop(self):
        """ВЕЖЛИВЫЙ стоп. У ртов есть свои точки чистого выхода (facet — между мухами,
        dedup — между видами, lang_runner — между задачами): там они ДОСОХРАНЯЮТ
        сделанное. Раньше я ставил флаг и ТУТ ЖЕ бил SIGTERM — рот не успевал дойти
        до своей проверки, и вызовы к Gemini сгорали впустую. Теперь: флаг → ждём →
        и только упрямого дожимаем сигналом.
        """
        log(f"СТОП запрошен | бежит={self.kind if self.busy() else 'ничего'}")
        for f in STOP_FLAGS:
            open(f, "w").close()
        if not self.busy():
            say("⛔ стоп: живых задач нет, флаг поставлен на всякий.")
            return
        say(
            f"⛔ стоп принят: {self.kind} дожёвывает текущий шаг и сохраняет "
            f"(до {GRACE_S // 60} мин). Финальный отчёт придёт."
        )
        threading.Thread(target=self._escalate, daemon=True).start()

    def _escalate(self):
        """Дать роту доработать по-хорошему; не вышел — SIGTERM, совсем упрямый — SIGKILL."""
        t0 = time.time()
        while self.busy() and time.time() - t0 < GRACE_S:
            time.sleep(2)
        if not self.busy():
            log(f"стоп: вышел сам за {time.time() - t0:.0f}с — данные сохранены")
            return
        log(f"стоп: не вышел за {GRACE_S}с → SIGTERM")
        say(f"⚠️ {self.kind} не вышел по-хорошему за {GRACE_S // 60} мин — шлю SIGTERM.")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception as e:
            log("SIGTERM не прошёл:", e)
        t1 = time.time()
        while self.busy() and time.time() - t1 < 30:
            time.sleep(2)
        if self.busy():
            log("стоп: не умер и от SIGTERM → SIGKILL")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception as e:
                log("SIGKILL не прошёл:", e)

    def status(self):
        total, mouths, kmax, n429 = brain_stats()
        m = ", ".join(f"{c} {n}" for c, n in mouths) or "тишина"
        run = (
            f"бежит {self.kind}, {(time.time() - self.t0) / 60:.0f} мин, "
            f"последнее: {self.tail[-200:] or '—'}"
            if self.busy()
            else "задач нет"
        )
        say(
            f"📊 {run}\nPT-день: попыток {total} | макс-ключ {kmax}/440 | 429 за час: {n429}\n"
            f"рты: {m}",
            stop_btn=self.busy(),
        )

    def reporter(self):
        """Два слоя в ТГ, чтобы НЕ надо было сидеть в докплой-логах и при этом не спамить:
        (1) ЖИВАЯ строка — правим одно и то же сообщение каждые LIVE_EVERY сек;
        (2) ВЕХА — новое сообщение каждые REPORT_EVERY попыток мозга (канон юзера п.2/4).
        """
        while True:
            time.sleep(LIVE_EVERY)
            if not self.busy():
                continue
            total, mouths, kmax, n429 = brain_stats()
            spent = total - self.base_attempts
            mins = (time.time() - self.t0) / 60
            rate = spent / mins if mins else 0
            head = (
                f"⚙️ {self.kind} идёт {mins:.0f} мин | попыток {spent} | {rate:.0f}/мин\n"
                f"{self.tail[-200:] or 'разогрев…'}"
            )
            edit(self.live_msg, head)  # живая строка — всегда актуальна
            if spent - self.last_report_at >= REPORT_EVERY:  # веха
                self.last_report_at = spent
                m = ", ".join(f"{c} {n}" for c, n in mouths)
                say(
                    f"⚙️ {self.kind} | попыток за задачу: {spent} | {rate:.0f}/мин\n"
                    f"прогресс: {self.tail[-200:] or '—'}\n"
                    f"ключи: макс {kmax}/440 | 429 за час: {n429}\nрты дня: {m}",
                    stop_btn=True,
                )


def send_menu(job):
    """Меню = карточка состояния + ВЕРТИКАЛЬ ТРАКТА. Юзер не должен держать порядок в
    голове — пульт считает его сам.

    ПРАВИЛА (юзер 2026-07-27), из них меню однозначно:
      1. вертикаль = порядок исполнения: что выше, то делается раньше;
      2. верхняя кнопка делает ровно всё, что под ней — без исключений;
      3. одна кнопка = один смысл (прежние 🆕/🔧 дублировали шаг 0 и висели НАД ним,
         из-за чего вертикаль расходилась с нумерацией);
      4. шаг, где делать нечего, не исчезает, а стоит на своём месте с ✅ — позиции
         не скачут, и меню не надо перечитывать заново каждый раз.
    """
    card, _ = state_card()
    s = pipeline_state()
    steps = pipeline_steps(s)
    total = sum(len(st["jobs"]) for st in steps)
    rows = []
    if total:
        rows.append(
            [
                {
                    "text": f"▶️ ВСЁ ПО ПОРЯДКУ — {total} шагов",
                    "callback_data": "run:cycle",
                }
            ]
        )
    first = next((st["kind"] for st in steps if st["jobs"]), None)
    for st in steps:
        if not st["jobs"]:  # сделано — место держим, стрелку не ставим
            rows.append([{"text": st["label"] + " ✅", "callback_data": "menu"}])
            continue
        mark = "➡️ " if st["kind"] == first else "　　"  # ровно ОДНА стрелка на меню
        rows.append(
            [{"text": mark + st["label"], "callback_data": f"run:{st['kind']}"}]
        )
        if st["note"]:
            rows.append([{"text": f"　　　└ {st['note']}", "callback_data": "menu"}])
        # ⛔ Строки по гео берём из очереди шага, а не из ключа `geos` у шага: шаги нового
        # тракта его не носят, и стартовое меню падало на KeyError.
        for x in (facet_queue(s) if st["kind"] == "mark" else [])[:8]:
            rows.append(
                [
                    {
                        "text": "　　　└ %s — %d мух" % (x["geo"], x["n"]),
                        "callback_data": f"run:mark:{x['geo']}",
                    }
                ]
            )
    tg("sendMessage", chat_id=CHAT, text=card, reply_markup={"inline_keyboard": rows})
    log("меню отправлено | шагов:", total, "| первый:", first)


def start_cycle(job):
    """Полный цикл = ВСЯ вертикаль тракта подряд. Worst-case пишем ДО запуска.

    ⛔ Цепочка берётся из `pipeline_steps` — из ТОГО ЖЕ списка, по которому рисуется меню.
    Прежде она собиралась здесь отдельно и БЕЗ шага 0: кнопка обещала «по порядку», а
    разметку пропускала, и страницы собирались по непротегованным мухам.
    """
    s = pipeline_state()
    steps = pipeline_steps(s)
    chain = [j for st in steps for j in st["jobs"]]
    if not chain:
        say("цикл не нужен: всё готово, можно шипить.")
        return
    # ИСПОЛНЕНИЕ ВСЛУХ (worst-case, не «выглядит ок»): считаем ДО запуска и вслух.
    fq = facet_queue(s)
    est = (
        sum(x["n"] for x in fq)  # разметка: батч 25, но worst-case — запрос на муху
        + len(s["no_shelf"]) * 70
        + s["no_kratko"]
        + s["no_branch"]  # ветвление: ~1 запрос на страницу-гиганта
        + sum((m + st_) * 3 for _, m, st_ in s["langs"])
    )
    plan = " → ".join(
        f"{st['label'].split(' — ')[0]}×{len(st['jobs'])}" for st in steps if st["jobs"]
    )
    say(
        f"⛓ весь тракт: {len(chain)} шагов\n"
        f"порядок: {plan}\n"
        f"ГРУБАЯ оценка расхода: ~{est} запросов (при 12 ключах × 440 = 5280/день)\n"
        f"остановить можно в любой момент — ⛔ СТОП рвёт цепочку."
    )
    first = chain.pop(0)
    job.chain = chain
    job.start(first[0], first[1], _chain=True)


def main():
    log(f"СТАРТ пульта | админ={CHAT} | BRAIN={BRAIN} | отчёт каждые {REPORT_EVERY}")
    log("меню:", ", ".join(MENU))
    # long-poll и webhook несовместимы: если у токена висит webhook (остался от другого
    # бота/прежней конфигурации), getUpdates отдаёт Conflict и команды не доходят.
    w = tg("deleteWebhook", drop_pending_updates=False)
    log("deleteWebhook:", w.get("ok"), w.get("description", ""))
    me = tg("getMe").get("result", {})
    log(f"я бот: @{me.get('username')} (id={me.get('id')})")
    job = Job()
    threading.Thread(target=job.reporter, daemon=True).start()
    threading.Thread(target=ban_watch, daemon=True).start()  # сигнал о банах ключей
    say("🟢 комбайн-пульт на связи. /combine — меню, /status, /stop")
    # Рестарт контейнера убил рот вместе с PID-namespace (пульт = PID 1 — проверено
    # NSpid). Задача оборвалась молча, а канон обещает отчёт при ЛЮБОМ исходе.
    for kind, ts in close_interrupted_jobs():
        say(
            f"⚠️ прошлый прогон «{kind}» оборван рестартом пульта "
            f"(запущен был {time.strftime('%d.%m %H:%M', time.localtime(ts))}).\n"
            f"Сделанное сохранено до последней точки выхода рта; продолжить — кнопкой ниже."
        )
    # АВТОНОМНО: при старте сразу показываем реальное состояние тракта + кнопки, чтобы
    # не держать его в голове и не жать /combine вручную (заказ юзера 07-24).
    try:
        send_menu(job)
    except Exception as e:
        log("стартовое меню не отправилось:", type(e).__name__, e)
    offset = 0
    while True:
        r = tg("getUpdates", offset=offset, timeout=30)
        for u in r.get("result", []):
            offset = u["update_id"] + 1
            cb = u.get("callback_query")
            if cb:
                if cb["from"]["id"] != CHAT:
                    log(f"ОТКАЗ кнопка от чужого id={cb['from']['id']} (админ {CHAT})")
                    continue
                log("←КНОПКА:", cb["data"])
                tg("answerCallbackQuery", callback_query_id=cb["id"])
                data = cb["data"]
                if data == "stop":
                    job.stop()
                elif data == "menu":
                    send_menu(job)
                elif data == "run:cycle":
                    start_cycle(job)
                elif data.startswith("run:"):
                    _, kind, geo = (data + ":").split(":")[:3]
                    # КНОПКА ШАГА БЕЗ ГЕО = ВСЕ РАБОТЫ ЭТОГО ШАГА, и берём их из
                    # `pipeline_steps` — того же списка, что рисует меню и собирает цикл.
                    #
                    # ⛔ Раньше здесь была ветка НА КАЖДЫЙ РОТ: своя для `facet` (цепочка из
                    # facet_queue), своя для `assign` (из no_shelf), а всё остальное падало
                    # в `job.start(kind, geo or None)`. То есть правило «как развернуть шаг
                    # в цепочку» жило третьей копией — и новый шаг её не получил: кнопка
                    # «3. Адреса страниц» ушла в общую ветку без гео, вышло
                    # `--stamp-keys ""` и падение на пути `out_facet/.json` (08.08). Шаг был
                    # добавлен в реестр ртов, в метрику и в вертикаль — в четвёртое место
                    # нет. Теперь мест ОДНО: добавил шаг в pipeline_steps — кнопка работает.
                    #
                    # `all`/`new` — псевдонимы пустого гео: их шлют кнопки СТАРЫХ сообщений,
                    # которые Telegram хранит вечно, и молча ронять их нельзя.
                    if not geo or geo in ("all", "new"):
                        st = next(
                            (
                                x
                                for x in pipeline_steps(pipeline_state())
                                if x["kind"] == kind
                            ),
                            None,
                        )
                        jobs = (st or {}).get("jobs") or []
                        if not jobs:
                            say(f"шагу «{kind}» делать нечего.")
                        else:
                            job.chain = jobs[1:]
                            job.start(jobs[0][0], jobs[0][1], _chain=True)
                    else:
                        job.start(kind, geo)
                continue
            msg = u.get("message") or {}
            src = msg.get("from", {}).get("id")
            if src != CHAT:  # не юзер — в ТГ молчим (канон), но в лог пишем ВСЕГДА
                log(
                    f"ОТКАЗ сообщение от чужого id={src} (админ {CHAT}): {msg.get('text')}"
                )
                continue
            text = (msg.get("text") or "").strip()
            log("←КОМАНДА:", text)
            if text in ("/combine", "/start"):
                send_menu(job)
            elif text in ("/stop", "/combine_stop"):
                job.stop()
            elif text in ("/status", "/combine_status"):
                job.status()
            else:
                parts = text.split()
                if parts and parts[0] in MENU:
                    job.start(parts[0], parts[1] if len(parts) > 1 else None)


if __name__ == "__main__":
    main()
