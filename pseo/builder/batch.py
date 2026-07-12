"""
batch.py — прогон билдера по списку ячеек (Ferrari: медленно, сам, с докатом) +
НАБЛЮДАЕМОСТЬ: статус-файл (glance), аудит-лог, чистый STOP-флаг, чирик при сбое.

Читает cells.json = [{geo, tag, slug, intent_name}], по каждой зовёт build.py.
RESUME: если out/<geo>_<slug>.json уже есть — пропускает (докат после обрыва/лимита).
STOP: файл ./STOP → грациозный выход (не pkill). Тайм-окно разделителя — пауза.

Наблюдаемость (для cron-сторожа и юзера, см. process_autonomous_auditor_bounds):
  status.json — {state, ts, built, skipped, failed, gate_drops, last_cell, pending}
  audit.log   — построчно: что собрано, сколько FAQ, что focus/gate выкинули
  чирик       — ТОЛЬКО при сбое/стопе (не спам «всё ок»); эскалацию делает сторож.

Запуск (VPS, фоном):
    LC_ALL=C.UTF-8 nohup /root/embed_ab/venv/bin/python -u batch.py cells.json > batch.log 2>&1 &
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

PY = "/root/embed_ab/venv/bin/python"
CHIRP = "/root/embed_ab/chirp.sh"  # переиспользуем маячок sweeper'а (→ тех-канал)
STATUS = "status.json"
AUDIT = "audit.log"
STOP = "STOP"
GAP = 8

# Прод-окно вечернего «разделителя»: старт ~20:30 UTC (23:30 МСК) на тех же flash-ключах.
BLOCK_UTC = (20, 23)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def in_prod_window():
    return BLOCK_UTC[0] <= datetime.now(timezone.utc).hour < BLOCK_UTC[1]


def wait_out_prod_window(st):
    while in_prod_window():
        st["state"] = "paused-prod-window"
        write_status(st)
        print(f"  ⏸ прод-окно ({_now()}) — пауза", flush=True)
        time.sleep(600)


def write_status(st):
    st["ts"] = _now()
    try:
        with open(STATUS, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print("status write err:", e, flush=True)


def audit(line):
    try:
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(f"{_now()}\t{line}\n")
    except Exception:
        pass


def chirp(msg):
    try:
        subprocess.run(["sh", CHIRP, msg], timeout=30)
    except Exception as e:
        print("chirp err:", e, flush=True)


def main():
    cells = json.loads(open(sys.argv[1], encoding="utf-8").read())
    st = {
        "state": "running",
        "started": _now(),
        "total": len(cells),
        "built": 0,
        "skipped": 0,
        "failed": 0,
        "gate_drops": 0,
        "last_cell": None,
        "pending": len(cells),
        "fails": [],
    }
    write_status(st)
    audit(f"BATCH START total={len(cells)}")

    for i, c in enumerate(cells):
        if os.path.exists(STOP):
            st["state"] = "stopped"
            write_status(st)
            audit("STOP flag — грациозный выход")
            chirp(
                f"⏹ [pseo-builder] STOP: собрано {st['built']}, ошибок {st['failed']}. Требует внимания."
            )
            return
        wait_out_prod_window(st)
        st["state"] = "running"
        cell = f"{c['geo']}/{c['slug']}"
        st["last_cell"] = cell
        out = f"out/{c['geo']}_{c['slug']}.json"
        if os.path.exists(out):
            st["skipped"] += 1
            st["pending"] -= 1
            write_status(st)
            continue
        print(f"[{i+1}/{len(cells)}] build {cell} …", flush=True)
        r = subprocess.run(
            [
                PY,
                "build.py",
                c["geo"],
                c["tag"],
                c["slug"],
                c["intent_name"],
                "--limit",
                str(c.get("limit", 70)),
            ],
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
            },
            capture_output=True,
            text=True,
        )
        so = r.stdout or ""
        drops = len(re.findall(r"(topic-focus|грунт-гейт) выкинул", so))
        faqs = re.search(r"\((\d+) FAQ\)", so)
        st["gate_drops"] += drops
        st["pending"] -= 1
        if os.path.exists(out):
            st["built"] += 1
            audit(f"OK {cell} faq={faqs.group(1) if faqs else '?'} drops={drops}")
        else:
            st["failed"] += 1
            st["fails"].append(cell)
            audit(f"FAIL {cell} :: {r.stderr.strip()[-160:]}")
            print("  FAIL:", r.stderr.strip()[-200:], flush=True)
        write_status(st)
        time.sleep(GAP)

    st["state"] = "done"
    write_status(st)
    audit(
        f"BATCH DONE built={st['built']} skip={st['skipped']} fail={st['failed']} gate_drops={st['gate_drops']}"
    )
    # Чирик ТОЛЬКО если есть на что смотреть (сбои). Чистый успех — тихо, статус-файл скажет.
    if st["failed"]:
        chirp(
            f"⚠️ [pseo-builder] готово с ошибками: собрано {st['built']}, "
            f"ошибок {st['failed']} ({', '.join(st['fails'][:5])}). Глянь."
        )
    print(
        f"\nИТОГ: built={st['built']} skip={st['skipped']} fail={st['failed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
