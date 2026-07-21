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
JOBS_DB = os.path.join(BRAIN, "combine_jobs.db")
KB_DB = os.path.join(BRAIN, "keybroker.db")
# два флага: facet-рты чтут RUNNER_STOP, lang_runner — LANG_RUNNER_STOP
STOP_FLAGS = [
    os.path.join(BRAIN, "RUNNER_STOP"),
    os.path.join(BRAIN, "LANG_RUNNER_STOP"),
]

# Меню: kind → (кнопка, argv дубля; {geo} подставляется). cwd=BRAIN — данные хоста.
MENU = {
    "kratko": (
        "Kratko (дожим)",
        ["python", f"{BUILDER}/dedup.py", "--all", "--kratko"],
    ),
    "translate": ("Переводы (очередь)", ["python", f"{BUILDER}/lang_runner.py"]),
    "facet": ("Facet+carve <гео>", ["python", f"{BUILDER}/facet.py", "{geo}"]),
    "assign": (
        "Хвост→полки <гео>",
        ["python", f"{BUILDER}/facet.py", "{geo}", "--assign-tail"],
    ),
}


def log(*a):
    """Всё, что делает пульт — в stdout контейнера (вкладка Logs в Dokploy)."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tg(method, **kw):
    try:
        r = requests.post(f"{API}/{method}", json=kw, timeout=35).json()
        if not r.get("ok", True):
            log("TG-ОШИБКА", method, r.get("description"))
        return r
    except Exception as e:
        log("TG-СБОЙ", method, type(e).__name__, e)
        return {}


def say(text, stop_btn=False):
    kw = {"chat_id": CHAT, "text": text}
    if stop_btn:
        kw["reply_markup"] = {
            "inline_keyboard": [[{"text": "⛔ СТОП", "callback_data": "stop"}]]
        }
    log("→ЮЗЕРУ:", text.replace("\n", " | ")[:200])
    tg("sendMessage", **kw)


def jobs_conn():
    c = sqlite3.connect(JOBS_DB, timeout=30)
    c.execute(
        "CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL, kind TEXT, args TEXT, status TEXT, note TEXT)"
    )
    return c


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


class Job:
    """Одна задача = один subprocess дубля. Глобально не больше одной."""

    def __init__(self):
        self.proc = None
        self.kind = None
        self.t0 = 0.0
        self.base_attempts = 0
        self.last_report_at = 0
        self.tail = ""
        self.lock = threading.Lock()

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind, geo=None):
        with self.lock:
            if self.busy():
                say(f"занято: {self.kind} уже бежит. Сначала ⛔ СТОП.")
                return
            argv = [a.replace("{geo}", geo or "") for a in MENU[kind][1]]
            for f in STOP_FLAGS:  # прошлый стоп не должен глушить новый заказ
                if os.path.exists(f):
                    os.remove(f)
            j = jobs_conn()
            j.execute(
                "INSERT INTO jobs(ts,kind,args,status) VALUES(?,?,?,?)",
                (time.time(), kind, json.dumps(argv), "running"),
            )
            j.commit()
            j.close()
            self.kind, self.t0 = kind, time.time()
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
            threading.Thread(target=self._pump, daemon=True).start()
            say(f"▶️ пошёл: {kind}" + (f" ({geo})" if geo else ""), stop_btn=True)

    def _pump(self):
        with open(self.logpath, "a", encoding="utf-8") as lf:
            for line in self.proc.stdout:
                lf.write(line)  # полный вывод рта — в файл лога
                lf.flush()
                print(f"[{self.kind}] {line}", end="")  # и в docker logs
                line = line.strip()
                if line:
                    self.tail = line  # последняя строка — в отчёты
        rc = self.proc.wait()
        spent = brain_stats()[0] - self.base_attempts
        mins = (time.time() - self.t0) / 60
        icon = "✅" if rc == 0 else "💀"
        j = jobs_conn()
        j.execute(
            "UPDATE jobs SET status=?, note=? WHERE id=(SELECT MAX(id) FROM jobs)",
            (f"exit={rc}", self.tail[-300:]),
        )
        j.commit()
        j.close()
        say(
            f"{icon} {self.kind}: код {rc}, {mins:.0f} мин, попыток ~{spent}\n"
            f"последнее: {self.tail[-300:] or '—'}\n"
            f"лог: {self.logpath}"
        )

    def stop(self):
        log(f"СТОП запрошен | бежит={self.kind if self.busy() else 'ничего'}")
        for f in STOP_FLAGS:  # рты сами встают между мухами
            open(f, "w").close()
        if self.busy():
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            say(
                "⛔ стоп: флаг поставлен, процессу послан SIGTERM. Жду финальный отчёт."
            )
        else:
            say("⛔ стоп: живых задач нет, флаг поставлен на всякий.")

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
        """Каждые REPORT_EVERY попыток мозга — отчёт с прогрессом (канон юзера п.2/4)."""
        while True:
            time.sleep(20)
            if not self.busy():
                continue
            total, mouths, kmax, n429 = brain_stats()
            spent = total - self.base_attempts
            if spent - self.last_report_at < REPORT_EVERY:
                continue
            self.last_report_at = spent
            mins = (time.time() - self.t0) / 60
            rate = spent / mins if mins else 0
            m = ", ".join(f"{c} {n}" for c, n in mouths)
            say(
                f"⚙️ {self.kind} | попыток за задачу: {spent} | {rate:.0f}/мин\n"
                f"прогресс: {self.tail[-200:] or '—'}\n"
                f"ключи: макс {kmax}/440 | 429 за час: {n429}\nрты дня: {m}",
                stop_btn=True,
            )


def main():
    log(f"СТАРТ пульта | админ={CHAT} | BRAIN={BRAIN} | отчёт каждые {REPORT_EVERY}")
    log("меню:", ", ".join(MENU))
    job = Job()
    threading.Thread(target=job.reporter, daemon=True).start()
    say("🟢 комбайн-пульт на связи. /combine — меню, /status, /stop")
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
                elif data.startswith("run:"):
                    _, kind, geo = (data + ":").split(":")[:3]
                    if kind in ("facet", "assign") and not geo:
                        say(f"гео не задано: пришли текстом `{kind} br`")
                    else:
                        job.start(kind, geo or None)
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
                rows = [
                    [{"text": label, "callback_data": f"run:{kind}"}]
                    for kind, (label, _) in MENU.items()
                ]
                tg(
                    "sendMessage",
                    chat_id=CHAT,
                    text="что запускаем?",
                    reply_markup={"inline_keyboard": rows},
                )
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
