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
BLOCK_UTC = (20, 23)
MIN_FLIES = 40
MIN_LEN = (
    140  # тот же порог длины ai_lesson, что в facet (для счёта корпуса в progress)
)
CHUNK = 50  # новых мух на гео за проход (пейсинг)
PASS_SLEEP = 1200  # пауза между проходами, сек
# КВОТА-ОСОЗНАННЫЙ потолок: не число мух, а доля дневного пула flash-lite.
# Читаем gemini_quota (билдер+вечерняя экстракция суммарно). Дошли до SOFT → стоп до суток,
# оставив запас экстракции. Резерв 80/ключ и так в gemini_json (прод не голодает). Модель RPD 500.
POOL_SOFT_FRAC = 0.55  # жрём до 55% пула, ~45% + резерв оставляем вечерней экстракции
RESERVE = 120  # выровнено с build.py: RPD-запас под потолок 500/проект
RPD = 500


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def in_window():
    return BLOCK_UTC[0] <= datetime.now(timezone.utc).hour < BLOCK_UTC[1]


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


def pool_state():
    """Сегодняшний расход flash-lite (билдер+экстракция) и мягкий потолок пула.
    Потолок = число_ключей × (RPD − резерв) × доля. Возвращает (used, soft)."""
    m = sqlite3.connect(DB, timeout=5)
    keys, used = m.execute(
        "SELECT COUNT(DISTINCT key_hash), COALESCE(SUM(count),0) FROM gemini_quota "
        "WHERE model LIKE '%flash-lite%' AND date_utc=date('now')"
    ).fetchone()
    m.close()
    keys = keys or 13  # если сегодня ещё пусто — берём известное число ключей
    soft = int(keys * (RPD - RESERVE) * POOL_SOFT_FRAC)
    return used, soft


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
        if in_window():
            save(STATUS, {"state": "paused-window", "ts": now()})
            time.sleep(600)
            continue

        used, soft = pool_state()
        if (
            used >= soft
        ):  # общий расход пула дошёл до мягкого потолка → отступаем до суток
            save(
                STATUS,
                {
                    "state": "pool-soft-reached",
                    "pool_used": used,
                    "soft": soft,
                    "ts": now(),
                },
            )
            time.sleep(1800)
            continue

        stamps = load(STAMPS, {})
        for geo, cnt, mx in geos():
            if os.path.exists(STOP) or in_window():
                break
            if stamps.get(geo) == mx:
                continue  # гео полностью тегнут при этих данных → скип
            used, soft = pool_state()
            if used >= soft:
                break  # пул почти выбран (билдер+экстракция) → стоп до суток
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
                    "pool_used": used,
                    "soft": soft,
                    "прогресс": progress(),
                    "ts": now(),
                },
            )
        used, soft = pool_state()
        save(
            STATUS,
            {
                "state": "idle",
                "pool_used": used,
                "soft": soft,
                "geos_stamped": len(stamps),
                "прогресс": progress(),
                "ts": now(),
            },
        )
        time.sleep(PASS_SLEEP)


if __name__ == "__main__":
    main()
