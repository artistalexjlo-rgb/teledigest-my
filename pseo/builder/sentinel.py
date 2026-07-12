"""
sentinel.py — ДЕТЕРМИНИРОВАННЫЙ сторож (не LLM-«Вася»). Правила, считаются дёшево,
срабатывают сами. Гоняется по cron часто (дёшево). Чирик в тех-канал ТОЛЬКО при срабатывании.

Проверки (все — правила, без интеллекта):
  1. Живой сайт: обход внутренних /ru/ ссылок от главной → каждая 200? (тухлые/битые ссылки)
  2. Билдер: status.json — МАССОВЫЙ провал (сбоев >=3 И >=30%) или завис (ts>6ч). Единичный
     сбой/парк — НЕ фатально, в лог, не пинг. Движок в канал не пишет вовсе (только sentinel).
  3. Квота: max_key близко к кэпу (билдер объедает прод-резерв).
LLM тут НЕ нужен — это HTTP-коды и пороги. Эскалация в дайджест только по факту срабатывания.

Запуск (VPS): /root/embed_ab/venv/bin/python sentinel.py   (cron, напр. каждые 6ч)
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
DRY = "--dry" in sys.argv  # диагностика: считать, но НЕ чирикать

SITE = "https://info.multyspeak.online"
DB = "/home/teledigest/data/messages_fts.db"
STATUS = "/root/pseo_builder/status.json"
CHIRP = "/root/embed_ab/chirp.sh"
QUOTA_CAP = 500
QUOTA_ALERT = 460  # max_key выше → билдер ест прод-резерв
STALE_HOURS = 6


def fetch(url):
    """curl (не urllib) — Cloudflare бот-защита режет Python-UA (403), curl проходит."""
    try:
        r = subprocess.run(
            ["curl", "-sS", "-m", "20", url, "-w", "\n__ST__%{http_code}"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        out = r.stdout
        body, _, code = out.rpartition("__ST__")
        return (int(code) if code.strip().isdigit() else 0), body
    except Exception:
        return 0, ""


def crawl():
    """BFS по внутренним /ru/ ссылкам от главной. Возвращает {url: status}, broken=[]."""
    seen, broken, queue = {}, [], ["/ru/"]
    while queue and len(seen) < 400:
        path = queue.pop(0)
        if path in seen:
            continue
        status, body = fetch(SITE + path)
        seen[path] = status
        if status != 200:
            broken.append((path, status))
            continue
        for href in re.findall(r'href="(/ru/[^"#?]*)"', body):
            if not href.endswith("/"):
                continue
            if href not in seen:
                queue.append(href)
    return seen, broken


def check_builder():
    issues = []
    try:
        st = json.loads(open(STATUS, encoding="utf-8").read())
    except Exception:
        return ["status.json нечитаем — билдер не рапортует"]
    # Фатально = МАССОВЫЙ провал (систематика), не 1-2 ячейки (это рутина, resume/парк).
    # Порог: сбоев >=3 И >=30% батча. Единичный сбой — в лог, не пинг. Парк — не сбой вовсе.
    failed, total = st.get("failed", 0), max(st.get("total", 1), 1)
    if failed >= 3 and failed / total >= 0.30:
        issues.append(
            f"билдер: МАССОВЫЙ провал {failed}/{total} ({', '.join(st.get('fails', [])[:4])}) — систематика"
        )
    # state=stopped НЕ алярмим: STOP инициирует человек, он и так знает.
    if st.get("state") == "running":
        try:
            ts = datetime.strptime(st["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            age = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age > STALE_HOURS:
                issues.append(
                    f"билдер завис: running, но статус {age:.0f}ч не обновлялся"
                )
        except Exception:
            pass
    return issues


def check_quota():
    try:
        out = subprocess.check_output(
            [
                "sqlite3",
                DB,
                "SELECT COALESCE(MAX(count),0) FROM gemini_quota "
                "WHERE date_utc=date('now') AND model='gemini-3.1-flash-lite';",
            ],
            text=True,
            timeout=15,
        ).strip()
        mx = int(out or 0)
    except Exception:
        return []
    if mx >= QUOTA_ALERT:
        return [f"квота: max_key={mx}/{QUOTA_CAP} — билдер у прод-резерва, притормози"]
    return []


def main():
    issues = []
    seen, broken = crawl()
    if broken:
        lst = ", ".join(f"{p}→{s}" for p, s in broken[:6])
        issues.append(f"битых ссылок {len(broken)}/{len(seen)}: {lst}")
    issues += check_builder()
    issues += check_quota()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if issues:
        msg = "🛡 [pseo-sentinel] " + " | ".join(issues)
        if not DRY:
            try:
                subprocess.run(["sh", CHIRP, msg[:600]], timeout=30)
            except Exception:
                pass
        print(f"{ts} ALERT {len(issues)}: {issues}")
    else:
        print(f"{ts} OK (страниц {len(seen)}, битых 0)")


if __name__ == "__main__":
    main()
