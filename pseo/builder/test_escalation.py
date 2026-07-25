"""Песочница (без реальных ключей): per-key 429-эскалация мозга + давность серии.

Лестница (канон юзера 07-21): 1-й 429 = только МЕТКА без наказания, дальше 60/300/1800/6000,
успех = полное прощение. Давность (07-25): ступень принадлежит СЕРИИ отказов, а не ключу —
тихо прожил дольше только что отсиженного, и следующий отказ снова считается первым.

⚠️ Прежняя версия теста ждала «1-й 429 → ~300с» и читала колонку was_cd — то и другое
протухло (правило сменилось 07-21, колонка снесена 07-25).

⚠️ Требует KB_DB на ОТДЕЛЬНУЮ базу: тест СНОСИТ файл. Без KB_DB не запускается, чтобы не
снести боевой keybroker.db.
"""

import os
import sys
import time

import keybroker as k

M = "gemini-3.1-flash-lite"
KEYS = ["A", "B", "C", "D"]


def state(key):
    """(секунд до конца кулдауна, ступень лестницы, метка) — как их видит мозг."""
    c = k._conn()
    r = c.execute(
        "SELECT cooldown_until, cd_level, struck FROM key_clock WHERE key_hash=?",
        (k._kh(key),),
    ).fetchone()
    c.close()
    if not r:
        return (None, None, None)
    return (round((r[0] or 0) - time.time()), r[1] or 0, r[2] or 0)


if __name__ == "__main__":
    if not os.environ.get("KB_DB"):
        sys.exit(
            "ОТКАЗ: задай KB_DB на тестовую базу — тест сносит файл, боевую не трогаем"
        )
    try:
        os.remove(k.DB)
    except OSError:
        pass
    k.init()
    ok = True

    k.report("t", "A", M, 429)
    s = state("A")
    print("1-й 429 на A → (cd, ступень, метка):", s, "(ожид: cd<=0, 0, 1)")
    ok &= s[0] <= 0 and s[1] == 0 and s[2] == 1

    k.report("t", "A", M, 429)
    s = state("A")
    print("2-й 429 на A →", s, "(ожид: ~60, ступень 1)")
    ok &= 55 <= s[0] <= 65 and s[1] == 1

    k.report("t", "A", M, 429)
    s = state("A")
    print("3-й 429 на A →", s, "(ожид: ~300, ступень 2)")
    ok &= 290 <= s[0] <= 310 and s[1] == 2

    k.report("t", "A", M, 200)
    s = state("A")
    print("200 на A (прощение) →", s, "(ожид: cd<=0, 0, 0)")
    ok &= s[0] <= 0 and s[1] == 0 and s[2] == 0

    # ДАВНОСТЬ СЕРИИ. Ключ B: метка, потом ступень 1 (отсидка 60с).
    k.report("t", "B", M, 429)  # метка
    k.report("t", "B", M, 429)  # ступень 1 → кулдаун ~60с
    s = state("B")
    print("2×429 по B →", s, "(ожид: ~60, ступень 1)")
    ok &= 55 <= s[0] <= 65 and s[1] == 1

    # отсидел 60с и сразу отказал снова → ТА ЖЕ серия, лестница вверх (300с)
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute(  # отматываем: отсидка кончилась 5с назад
        "UPDATE key_clock SET cooldown_until=? WHERE key_hash=?",
        (time.time() - 5, k._kh("B")),
    )
    c.execute("COMMIT")
    c.close()
    k.report("t", "B", M, 429)
    s = state("B")
    print("отказ ЧЕРЕЗ 5с после отсидки →", s, "(ожид: ~300, ступень 2 — серия та же)")
    ok &= 290 <= s[0] <= 310 and s[1] == 2

    # а если после отсидки прожил тихо ДОЛЬШЕ отсиженного — серия рассосалась,
    # следующий отказ снова ПЕРВЫЙ (бесплатная метка), без всякого прохода-чистильщика
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute(  # отсидка (300с) кончилась 400с назад — дольше, чем длилась
        "UPDATE key_clock SET cooldown_until=? WHERE key_hash=?",
        (time.time() - 400, k._kh("B")),
    )
    c.execute("COMMIT")
    c.close()
    k.report("t", "B", M, 429)
    s = state("B")
    print(
        "отказ через 400с тишины →",
        s,
        "(ожид: cd<=0, ступень 0, метка 1 — серия новая)",
    )
    ok &= s[0] <= 0 and s[1] == 0 and s[2] == 1

    # per-key независимость: отказы по C/D остужают ИХ, пул не встаёт (глоб-паузы нет)
    for kk in ("C", "D"):
        k.report("t", kk, M, 429)
    res = k.acquire("t", "background", M, KEYS)
    print("acquire после отказов по B/C/D →", (res[0], res[1]), "(ожид: выдан ключ)")
    ok &= res[0] is not None

    # ⭐ РЕГРЕСС-ГВОЗДЬ (баг вскрыт 07-24, стоил 31 мухи за один прогон):
    # «все ключи отдыхают» ≠ «бюджет выбран». Мозг обязан вернуть ПОЛОЖИТЕЛЬНОЕ ожидание,
    # иначе call() сдаётся навсегда при живой дневной квоте.
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute("UPDATE key_clock SET cooldown_until=?", (time.time() + 45,))
    c.execute("COMMIT")
    c.close()
    res = k.acquire("t", "background", M, KEYS)
    print("ВСЕ ключи в кулдауне →", res, "(ожид: (None, ~45) — ЖДАТЬ, не -1.0)")
    ok &= res[0] is None and res[1] is not None and 40 <= res[1] <= 50

    # а вот когда все на дневном бане — честный -1.0, работать правда нечем
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute("UPDATE key_clock SET cooldown_until=0")
    for kk in KEYS:
        c.execute(
            "INSERT INTO usage(key_hash, model, pt_day, count, banned) VALUES(?,?,?,0,1) "
            "ON CONFLICT(key_hash, model, pt_day) DO UPDATE SET banned=1",
            (k._kh(kk), M, k._pt_day()),
        )
    c.execute("COMMIT")
    c.close()
    res = k.acquire("t", "background", M, KEYS)
    print("ВСЕ ключи забанены →", res, "(ожид: (None, -1.0) — сдаться честно)")
    ok &= res[0] is None and res[1] == -1.0

    print("VERDICT:", "OK — лестница и давность серии работают" if ok else "FAIL")
