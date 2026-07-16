"""runner.py — НЕПРЕРЫВНАЯ авто-сборка страниц в ТЕМПЕ, в рамках квот (Ferrari).
Бережное потребление ВШИТО:
  - DAILY_FLIES: не больше N новых мух в СУТКИ (растягивает объём, не грызёт всё разом);
  - CHUNK: порция новых мух на гео за проход (facet резюмируемый — догоняет постепенно);
  - окно 20-23 UTC (пауза), резерв ключей (в gemini_json), STOP-флаг, статус-лампа.
Дневной бюджет выбран → спит до смены суток. Гео с полностью тегнутыми мухами (и без новых
данных) — скип. Живой процесс (nohup/systemd).
"""

import json
import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone

PY = "/root/embed_ab/venv/bin/python"
DB = "/home/teledigest/data/messages_fts.db"
HERE = "/root/pseo_builder"
STATUS = f"{HERE}/runner_status.json"
STOP = f"{HERE}/RUNNER_STOP"
STAMPS = f"{HERE}/runner_stamps.json"
BUDGET = f"{HERE}/runner_budget.json"
MIN_FLIES = 40
MIN_LEN = (
    140  # тот же порог длины ai_lesson, что в facet (для счёта корпуса в progress)
)
CHUNK = 50  # новых мух на гео за проход (пейсинг)
PASS_SLEEP = 1200  # пауза между проходами, сек
# КВОТУ держит МОЗГ (keybroker): per-ключ RPD-кап (RPD−RESERVE) + per-рот кап. Раннерского
# pool_state (чтение gemini_quota) БОЛЬШЕ НЕТ — второго квота-источника быть не должно.
# Окна экстракции тоже НЕТ: коэкзистенцию держит резерв мозга (60/ключ) + per-ключ шаг + abuse.


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def geos():
    m = sqlite3.connect(DB)
    rows = m.execute(
        "SELECT country, COUNT(*) c, MAX(extracted_at) mx FROM extracted_patterns "
        "WHERE country!='any' AND country IS NOT NULL AND ai_lesson IS NOT NULL "
        "GROUP BY country HAVING c>=? ORDER BY c DESC",
        (MIN_FLIES,),
    ).fetchall()
    m.close()
    return rows


def progress():
    """Сводка прогресса для лампы: сколько мух тегнуто / корпус / %, гео готово, страниц-срезов."""
    import glob

    tagged = 0
    for f in glob.glob(f"{HERE}/tags/*.json"):
        try:
            tagged += len(json.load(open(f, encoding="utf-8")))
        except Exception:
            pass
    m = sqlite3.connect(DB, timeout=5)
    corpus = m.execute(
        "SELECT COUNT(*) FROM extracted_patterns WHERE country!='any' AND country IS NOT NULL "
        "AND ai_lesson IS NOT NULL AND length(ai_lesson)>?",
        (MIN_LEN,),
    ).fetchone()[0]
    m.close()
    return {
        "мух_тегнуто": tagged,
        "корпус": corpus,
        "готово_%": f"{tagged * 100 // max(corpus, 1)}%",
        "гео_с_фактами": len(glob.glob(f"{HERE}/out_facet/*.json")),
        "гео_с_вопросами": len(glob.glob(f"{HERE}/out_questions/*.json")),
    }


def load(p, d):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return d


def save(p, o):
    json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def build_facet(geo, chunk):
    """facet ≤chunk новых мух. Возвращает сколько НОВЫХ реально тегнуто (парсит вывод)."""
    r = subprocess.run(
        [PY, "facet.py", geo, "--limit", str(chunk)],
        cwd=HERE,
        env={**os.environ, "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
    )
    if "прод-окно" in (r.stderr or ""):
        return -1, None  # окно
    m = re.search(r"\+(\d+) новых", r.stdout or "")
    rem = re.search(r"remaining=(\d+)", r.stdout or "")
    return (int(m.group(1)) if m else 0), (int(rem.group(1)) if rem else None)


def build_questions(geo):
    subprocess.run(
        [PY, "questions_page.py", geo, "--limit", "120"],
        cwd=HERE,
        env={**os.environ, "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
    )


def main():
    while True:
        if os.path.exists(STOP):
            save(STATUS, {"state": "stopped", "ts": now()})
            return

        stamps = load(STAMPS, {})
        for geo, cnt, mx in geos():
            if os.path.exists(STOP):
                break
            if stamps.get(geo) == mx:
                continue  # гео полностью тегнут при этих данных → скип
            # КВОТА — на мозге: build_facet зовёт рты через keybroker.call, тот сам режет на капе.
            new_n, remaining = build_facet(geo, CHUNK)
            if new_n < 0:
                break  # окно
            if new_n > 0:
                build_questions(geo)  # вопросы дёшевы, обновим срез
            if remaining == 0:  # stamp ТОЛЬКО по честному остатку (сбои мух ≠ исчерпан)
                stamps[geo] = mx
                save(STAMPS, stamps)
            save(
                STATUS,
                {
                    "state": "building",
                    "geo": geo,
                    "new": new_n,
                    "прогресс": progress(),
                    "ts": now(),
                },
            )
        save(
            STATUS,
            {
                "state": "idle",
                "geos_stamped": len(stamps),
                "прогресс": progress(),
                "ts": now(),
            },
        )
        time.sleep(PASS_SLEEP)


if __name__ == "__main__":
    main()
