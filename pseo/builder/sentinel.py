"""
sentinel.py — ДЕТЕРМИНИРОВАННЫЙ сторож (не LLM-«Вася»). Правила, считаются дёшево,
срабатывают сами. Гоняется по cron часто (дёшево). Чирик в тех-канал ТОЛЬКО при срабатывании.

Проверки (все — правила, без интеллекта):
  1. Живой сайт: обход внутренних /ru/ ссылок от главной → каждая 200? (тухлые/битые ссылки)
     + контент-целостность каждой 200-страницы: есть <h1>, тело не пустое, нет '�'.
  2. Sitemap-свип: живой /sitemap.xml → КАЖДЫЙ <loc> отвечает 200 (ловит orphans, недоехавший
     CF-билд, потерянные файлы — краул их не видит). + robots.txt доступен.
(runner- и квота-проверки вырезаны 2026-07-20: раннера нет как класса, всё по отмашке.)
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
STATUS = "/root/pseo_builder/status.json"
CHIRP = "/root/embed_ab/chirp.sh"
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
    """BFS по внутренним /ru/ ссылкам от главной. Возвращает {url:status}, broken, damaged
    (200, но контент битый: нет h1 / тело <400 симв / '�')."""
    seen, broken, damaged, queue = {}, [], [], ["/ru/"]
    while queue and len(seen) < 400:
        path = queue.pop(0)
        if path in seen:
            continue
        status, body = fetch(SITE + path)
        seen[path] = status
        if status != 200:
            broken.append((path, status))
            continue
        if "<h1>" not in body or len(re.sub(r"<[^>]+>", "", body)) < 400 or "�" in body:
            damaged.append(path)
        for href in re.findall(r'href="(/ru/[^"#?]*)"', body):
            if not href.endswith("/"):
                continue
            if href not in seen:
                queue.append(href)
    return seen, broken, damaged


def head(url):
    """Быстрый статус без тела (для sitemap-свипа)."""
    try:
        r = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-m", "12", "-w", "%{http_code}", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def check_sitemap(crawled):
    """Живой sitemap: доступен, не пуст, КАЖДЫЙ loc → 200 (не-краулнутые добираем head'ом).
    Ловит orphans/недоехавший CF-билд. + robots.txt доступен."""
    issues = []
    st, body = fetch(SITE + "/sitemap.xml")
    if st != 200 or "<loc>" not in body:
        return [f"sitemap: недоступен/пуст (HTTP {st})"]
    locs = re.findall(r"<loc>([^<]+)</loc>", body)
    dead = []
    for u in locs[:300]:
        path = u.replace(SITE, "")
        code = crawled.get(path) or head(u)
        if code != 200:
            dead.append(f"{path}→{code}")
    if dead:
        issues.append(
            f"sitemap: {len(dead)}/{len(locs)} мёртвых loc: {', '.join(dead[:5])}"
        )
    if head(SITE + "/robots.txt") != 200:
        issues.append("robots.txt недоступен")
    return issues


# check_runner ВЫРЕЗАН 2026-07-20: раннера нет как класса (всё по отмашке).
# check_quota ВЫРЕЗАН 2026-07-20 (юзер: неактуально): сценарий «самоходный билдер
# объедает прод-резерв» умер вместе с раннером; расход по отмашке виден в keybroker.


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


def main():
    issues = []
    seen, broken, damaged = crawl()
    if broken:
        lst = ", ".join(f"{p}→{s}" for p, s in broken[:6])
        issues.append(f"битых ссылок {len(broken)}/{len(seen)}: {lst}")
    if damaged:
        issues.append(
            f"битый контент (200, но пусто/без h1/�): {len(damaged)}: {damaged[:4]}"
        )
    issues += check_sitemap(seen)
    issues += check_builder()

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
