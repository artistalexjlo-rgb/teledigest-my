"""Короткая нога: доказать, что мозг координирует ключи МЕЖДУ ПРОЦЕССАМИ.
4 отдельных процесса молотят 5 общих dummy-ключей через acquire(). Инвариант:
для каждого ключа два подряд grant'а не ближе RPM_STEP → осьминог не может насрать в колодец.
Гоняем на изолированной test-db с маленьким шагом (env), реальные ключи не нужны.
"""

import multiprocessing as mp
import time
from collections import defaultdict

import keybroker as kb

KEYS = ["dummy-key-%d" % i for i in range(5)]
MODEL = "gemini-3.1-flash-lite"
N_PER_WORKER = 12


def worker(wid, out):
    grants = []
    for _ in range(N_PER_WORKER):
        key, wait = kb.acquire("w%d" % wid, "background", MODEL, KEYS)
        if key is None:
            if wait and wait > 0:
                time.sleep(min(wait, 2))
                continue
            break  # exhausted
        grants.append((key, time.time()))
        kb.report("w%d" % wid, key, MODEL, 200)
    out.put(grants)


if __name__ == "__main__":
    kb.init()
    c = kb._conn()
    c.executescript(
        "DELETE FROM key_clock; DELETE FROM usage; DELETE FROM request_log;"
    )
    c.commit()
    c.close()

    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, q)) for i in range(4)]
    t0 = time.time()
    for p in procs:
        p.start()
    allg = []
    for _ in procs:
        allg += q.get()
    for p in procs:
        p.join()
    dur = time.time() - t0

    bykey = defaultdict(list)
    for k, ts in allg:
        bykey[k].append(ts)
    STEP = kb.step_for(MODEL)
    violations, mingap = 0, 999.0
    for k, tss in bykey.items():
        tss.sort()
        for a, b in zip(tss, tss[1:]):
            g = b - a
            mingap = min(mingap, g)
            if g < STEP - 0.05:
                violations += 1

    print(
        "step=%.2fs  grants=%d  keys=%d  dur=%.1fs" % (STEP, len(allg), len(bykey), dur)
    )
    print("min gap per key = %.3fs  → violations(<step)=%d" % (mingap, violations))
    print(
        "VERDICT:",
        (
            "OK — осьминог укрощён, clock общий"
            if violations == 0
            else "FAIL — burst прорвался"
        ),
    )
    print("request_log stats:", kb.stats())
