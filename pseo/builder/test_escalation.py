"""Песочница (без реальных ключей): проверка 429-эскалации мозга против эталона.
Ожидаем: 1-й 429→~300с, повтор→~1800с, 200→сброс, ≥3×429/60с→глоб abuse-пауза."""

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
    print("200 на A (прощение) →", r, "(ожид 0, 0)")
    ok &= r[0] <= 0 and r[1] == 0

    k.report("t", "B", M, 429)
    k.report("t", "C", M, 429)
    k.report(
        "t", "D", M, 429
    )  # 3-й 429 за окно (A уже простился, но лог 429 считает все)
    c = k._conn()
    ap = c.execute("SELECT abuse_pause_until FROM broker_global WHERE id=1").fetchone()
    c.close()
    left = round(ap[0] - time.time()) if ap else -1
    print("abuse_pause_until через ≥3×429 →", left, "с (ожид ~1800)")
    ok &= 1700 <= left <= 1810

    res = k.acquire("t", "background", M, KEYS)
    print(
        "acquire во время abuse →",
        (res[0], round(res[1]) if res[1] else res[1]),
        "(ожид (None, ~<=1800))",
    )
    ok &= res[0] is None and res[1] and res[1] > 1000

    print("VERDICT:", "OK — эскалация по эталону" if ok else "FAIL")
