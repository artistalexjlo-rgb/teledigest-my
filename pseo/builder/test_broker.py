"""Короткая нога: доказать, что мозг координирует ключи МЕЖДУ ПРОЦЕССАМИ.
4 отдельных процесса молотят 5 общих dummy-ключей через acquire(). Реальные ключи не нужны.

ИНВАРИАНТ ОЧЕРЕДИ (сквозной нумерации оборотов больше нет): выдаём следующего ПО СПИСКУ за
курсором → между двумя выдачами ОДНОГО ключа обязаны пройти ВСЕ остальные. Значит любое окно
из N подряд идущих грантов содержит N РАЗНЫХ ключей. Проверяем сам порядок выдачи, а не
номер оборота.

⚠️ Этот инвариант держат ОБЕ схемы — и позиционная, и сортировка по давности, — поэтому
подмену базы он поймать НЕ МОЖЕТ и однажды её пропустил. Позицию ключа стережёт test_order.py.

Порядок берём из request_log (это отметка САМОГО мозга в момент гранта), а не из времени в
воркере: между acquire() и записью в воркере процесс могут переключить, и порядок соврёт.

⚠️ Прежняя проверка была ПУСТОЙ: `STEP = kb.ROUND_PAUSE`, а ROUND_PAUSE=0, поэтому условие
`g < STEP-0.05` не срабатывало НИКОГДА — инвариант не проверялся ни разу.

⚠️ Требует KB_DB на ОТДЕЛЬНУЮ базу: тест чистит таблицы.
⚠️ И KB_GROUP_SIZE=0: тест про круг по ВСЕМУ пулу. Со звеньями окно молча ужимается до
   размера звена, и проверка «N разных в окне из N» становится слабее, ничего об этом не
   сказав. Без явного нуля тест откажется стартовать.
KB_GRANT_MAX=0.2 KB_GRANT_MIN=0.1 KB_GROUP_SIZE=0 KB_DB=/tmp/kb_test.db python test_broker.py
"""

import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import keybroker as kb

KEYS = ["dummy-key-%d" % i for i in range(5)]
MODEL = "gemini-3.1-flash-lite"
N_PER_WORKER = 12


def worker(wid, out):
    grants = []
    guard = 0  # считаем ГРАНТЫ, а не итерации: ожидание в очереди попытку не тратит
    while len(grants) < N_PER_WORKER and guard < N_PER_WORKER * 400:
        guard += 1
        key, wait = kb.acquire("w%d" % wid, "background", MODEL, KEYS)
        if key is None:
            if wait and wait > 0:
                time.sleep(min(wait, 2))
                continue
            if wait == 0.0:  # очередь: такт не прошёл
                time.sleep(0.05)
                continue
            break  # -1.0: бан/кап у всех
        grants.append((key, time.time()))
        kb.report("w%d" % wid, key, MODEL, 200)
    out.put(grants)


if __name__ == "__main__":
    if not os.environ.get("KB_DB"):
        sys.exit(
            "ОТКАЗ: задай KB_DB на тестовую базу — тест чистит таблицы, боевую не трогаем"
        )
    if kb.GROUP_SIZE:
        sys.exit(
            "ОТКАЗ: тест про круг по ВСЕМУ пулу, а стоит KB_GROUP_SIZE=%d. Запускай с "
            "KB_GROUP_SIZE=0; звенья проверяет test_group.py" % kb.GROUP_SIZE
        )
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

    # порядок выдачи — из журнала мозга
    c = kb._conn()
    seq = [
        r[0]
        for r in c.execute(
            "SELECT key_hash FROM request_log WHERE event='grant' ORDER BY ts"
        )
    ]
    c.close()

    n_keys = len(set(seq))
    violations = 0
    for i in range(len(seq) - n_keys + 1):
        if len(set(seq[i : i + n_keys])) != n_keys:
            violations += 1  # ключ повторился раньше, чем очередь обошла остальных

    bykey = defaultdict(int)
    for kh in seq:
        bykey[kh] += 1
    spread = (min(bykey.values()), max(bykey.values())) if bykey else (0, 0)

    print("grants=%d  keys=%d  dur=%.1fs" % (len(seq), n_keys, dur))
    print("выдач на ключ: min=%d max=%d (перекос = плохо)" % spread)
    print("окно из %d подряд → повторов внутри окна: %d" % (n_keys, violations))
    print(
        "VERDICT:",
        (
            "OK — очередь держит: ключ не повторяется, пока не пройдут остальные"
            if violations == 0
            else "FAIL — очередь налипает на часть ключей"
        ),
    )
    print("request_log stats:", kb.stats())
