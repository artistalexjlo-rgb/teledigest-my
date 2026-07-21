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
# сколько ждём вежливого выхода рта после стоп-флага (один вызов к Gemini + запись;
# у lang_runner шаг крупнее — гео×язык, потому не секунды)
GRACE_S = int(os.environ.get("COMBINE_GRACE_S", "180"))
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
MENU = {
    "kratko": (
        "Kratko (дожим)",
        ["python", "-u", f"{BUILDER}/dedup.py", "--all", "--kratko"],
    ),
    "translate": ("Переводы (очередь)", ["python", "-u", f"{BUILDER}/lang_runner.py"]),
    "facet": ("Facet+carve <гео>", ["python", "-u", f"{BUILDER}/facet.py", "{geo}"]),
    "assign": (
        "Хвост→полки <гео>",
        ["python", "-u", f"{BUILDER}/facet.py", "{geo}", "--assign-tail"],
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


def edit(msg_id, text, stop_btn=True):
    """Живой прогресс = ПРАВКА одного сообщения, не поток новых (юзер не должен сидеть
    в докплой-логах, но и спамить чат раз в 3 секунды нельзя)."""
    if not msg_id:
        return
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


def pipeline_state():
    """ЧТО СЕЙЧАС ПРОСРОЧЕНО — считается из данных, не из моей памяти.

    Порядок тракта: мухи → facet(+carve) → assign(хвост→полки) → kratko(короткий
    ответ) → translate(языки) → ship(с десктопа). Каждый шаг кормит следующий,
    поэтому чинить надо в этом порядке — иначе переводы поедут без kratko.
    """
    import glob

    st = {"geos": 0, "views": 0, "no_kratko": 0, "no_shelf": [], "langs": []}
    try:
        files = sorted(glob.glob(f"{BRAIN}/out_facet/*.json"))
        st["geos"] = len(files)
        for f in files:
            geo = os.path.basename(f)[:-5]
            d = json.load(open(f, encoding="utf-8"))
            vs = [
                v
                for v in d.get("views_by_task", [])
                if len(v.get("groups") or v.get("items") or []) >= 4
            ]
            st["views"] += len(vs)
            st["no_kratko"] += sum(1 for v in vs if not v.get("kratko"))
            if not (d.get("shelves") or []):
                st["no_shelf"].append(geo)
        for lang in LANGS:
            d = f"{BRAIN}/out_facet_{lang}"
            have = glob.glob(f"{d}/*.json")
            stale = sum(
                1
                for p in have
                if os.path.getmtime(p)
                < os.path.getmtime(f"{BRAIN}/out_facet/{os.path.basename(p)}")
            )
            miss = st["geos"] - len(have)
            if miss or stale:
                st["langs"].append((lang, miss, stale))
    except Exception as e:
        st["error"] = f"{type(e).__name__}: {e}"
    return st


def state_card():
    """Карточка состояния + подсказка «что дожать СЕЙЧАС» (первый непустой шаг)."""
    s = pipeline_state()
    if s.get("error"):
        return f"⚠️ не смог прочитать данные: {s['error']}", None
    lines = [f"📦 корпус: {s['geos']} гео, {s['views']} страниц-видов"]
    todo = []
    if s["no_shelf"]:
        lines.append(f"1) хвост не разложен по полкам: {len(s['no_shelf'])} гео")
        todo.append("assign")
    else:
        lines.append("1) хвост→полки: ✅ все гео")
    if s["no_kratko"]:
        lines.append(f"2) без короткого ответа: {s['no_kratko']} видов")
        todo.append("kratko")
    else:
        lines.append("2) короткие ответы: ✅ все виды")
    if s["langs"]:
        worst = ", ".join(
            f"{lang}(нет {m}, устар {st_})" for lang, m, st_ in s["langs"][:5]
        )
        lines.append(f"3) переводы: {len(s['langs'])} языков не готовы — {worst}")
        todo.append("translate")
    else:
        lines.append("3) переводы: ✅ все языки свежие")
    lines.append(
        "\n➡️ СЕЙЧАС НАДО: "
        + (
            {
                "assign": "разложить хвост (assign)",
                "kratko": "дожать kratko",
                "translate": "гнать переводы",
            }[todo[0]]
            if todo
            else "ничего — можно шипить (ship с десктопа)"
        )
    )
    lines.append(
        "порядок жёсткий: assign → kratko → translate → ship.\n"
        "переводы ПОСЛЕ kratko, иначе языки останутся без коротких ответов."
    )
    return "\n".join(lines), todo


class Job:
    """Одна задача = один subprocess дубля. Глобально не больше одной."""

    def __init__(self):
        self.proc = None
        self.kind = None
        self.t0 = 0.0
        self.base_attempts = 0
        self.last_report_at = 0
        self.tail = ""
        self.live_msg = None  # id сообщения с живым прогрессом (правим его же)
        self.lock = threading.Lock()
        self.chain = []  # очередь шагов полного цикла: [(kind, geo), ...]

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind, geo=None, _chain=False):
        with self.lock:
            if self.busy():
                say(f"занято: {self.kind} уже бежит. Сначала ⛔ СТОП.")
                return
            if not _chain:
                self.chain = []  # ручной запуск отменяет недобеганную цепочку
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
            self.live_msg = say(  # это сообщение будем ПРАВИТЬ живым прогрессом
                f"▶️ пошёл: {kind}" + (f" ({geo})" if geo else "") + "\nразогрев…",
                stop_btn=True,
            )

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
        # цепочка полного цикла: следующий шаг только если предыдущий вышел чисто
        # и стоп не нажат (нажатый стоп = юзер сказал «хватит», цепочка рвётся)
        if self.chain and rc == 0 and not any(os.path.exists(f) for f in STOP_FLAGS):
            kind, geo = self.chain.pop(0)
            say(f"⛓ цикл: следующий шаг — {kind}" + (f" ({geo})" if geo else ""))
            self.start(kind, geo, _chain=True)
        elif self.chain:
            n = len(self.chain)
            self.chain = []
            say(f"⛓ цикл прерван: осталось {n} шагов. Запусти заново, когда решишь.")

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
    """Меню = карточка состояния + кнопки С ЧИСЛАМИ + полный цикл. Юзер не должен
    держать состояние тракта в голове — пульт считает его сам."""
    card, todo = state_card()
    s = pipeline_state()
    labels = {
        "assign": f"1. Хвост→полки ({len(s['no_shelf'])} гео)",
        "kratko": f"2. Kratko ({s['no_kratko']} видов)",
        "translate": f"3. Переводы ({len(s['langs'])} языков)",
        "facet": "0. Facet+carve <гео> (новые мухи)",
    }
    rows = []
    if todo:
        rows.append(
            [{"text": "▶️ ПОЛНЫЙ ЦИКЛ по порядку", "callback_data": "run:cycle"}]
        )
    for kind in ("facet", "assign", "kratko", "translate"):
        mark = "➡️ " if todo and kind == todo[0] else ""
        rows.append([{"text": mark + labels[kind], "callback_data": f"run:{kind}"}])
    tg("sendMessage", chat_id=CHAT, text=card, reply_markup={"inline_keyboard": rows})
    log("меню отправлено | надо:", todo)


def start_cycle(job):
    """Полный цикл = очередь шагов в жёстком порядке. Worst-case пишем ДО запуска."""
    s = pipeline_state()
    chain = []
    for geo in s["no_shelf"]:
        chain.append(("assign", geo))
    if s["no_kratko"]:
        chain.append(("kratko", None))
    if s["langs"]:
        chain.append(("translate", None))
    if not chain:
        say("цикл не нужен: всё готово, можно шипить.")
        return
    # ИСПОЛНЕНИЕ ВСЛУХ (worst-case, не «выглядит ок»):
    est = (
        len(s["no_shelf"]) * 70
        + s["no_kratko"]
        + sum((m + st_) * 3 for _, m, st_ in s["langs"])
    )
    say(
        f"⛓ полный цикл: {len(chain)} шагов\n"
        f"порядок: {', '.join(k for k, _ in chain[:3])}…\n"
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
                elif data == "run:cycle":
                    start_cycle(job)
                elif data.startswith("run:"):
                    _, kind, geo = (data + ":").split(":")[:3]
                    if kind == "facet" and not geo:
                        say("facet нужен гео: пришли текстом `facet br`")
                    elif kind == "assign" and not geo:
                        s = pipeline_state()  # без гео — разложить все, где полок нет
                        if not s["no_shelf"]:
                            say("хвост разложен везде, assign не нужен.")
                        else:
                            job.chain = [("assign", g) for g in s["no_shelf"][1:]]
                            job.start("assign", s["no_shelf"][0], _chain=True)
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
