"""Песочница (без реальных ключей): проверка per-key 429-эскалации мозга.
Ожидаем: 1-й 429→~300с, повтор→~1800с, 200→сброс. (Глоб abuse-пауза вычищена, §2.5.)"""

import os
import time

import keybroker as k

M = "gemini-3.1-flash-lite"
KEYS = ["A", "B", "C", "D"]


def cd(key):
    c = k._conn()
    r = c.execute(
        "SELECT cooldown_until, was_cd FROM key_clock WHERE key_hash=?", (k._kh(key),)
    ).fetchone()
    c.close()
    return (round(r[0] - time.time()) if r else None, r[1] if r else None)


if __name__ == "__main__":
    try:
        os.remove(k.DB)
    except OSError:
        pass
    k.init()
    ok = True

    k.report("t", "A", M, 429)
    r = cd("A")
    print("1-й 429 на A → cooldown/was:", r, "(ожид ~300, 1)")
    ok &= 290 <= r[0] <= 310 and r[1] == 1

    k.report("t", "A", M, 429)
    r = cd("A")
    print("повтор 429 на A →", r, "(ожид ~1800, 1)")
    ok &= 1790 <= r[0] <= 1810 and r[1] == 1

    k.report("t", "A", M, 200)
    r = cd("A")
    print("200 на A (прощение = разбан) →", r, "(ожид 0, 0)")
    ok &= r[0] <= 0 and r[1] == 0

    # per-key независимость: 429 по B/C/D остужают ИХ, не весь пул (abuse-паузы нет)
    for kk in ("B", "C", "D"):
        k.report("t", kk, M, 429)
    res = k.acquire("t", "background", M, KEYS)  # A прощён → выдаётся, пул НЕ встал
    print(
        "acquire после 3×429 по B/C/D →",
        (res[0], res[1]),
        "(ожид: выдан ключ, НЕ пауза)",
    )
    ok &= res[0] is not None

    print("VERDICT:", "OK — per-key эскалация, глоб-паузы нет" if ok else "FAIL")
