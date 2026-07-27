"""ЗВЕНЬЯ: дежурит срез пула, между звеньями пауза.

Требование юзера (2026-07-27): «надо сделать как ротацию с паузой 4-пауза-4-пауза-4-пауза
-начало всех 12 ключей», уточнение — «но не быстро, какое-то время чтоб работали».
То есть звено меняется ПО ВРЕМЕНИ, а не по числу запросов.

Зачем вообще: замером 27.07 установлено, что стена 429 приходит от ЧИСЛА РАЗНЫХ КЛЮЧЕЙ,
засветившихся с адреса, а не от темпа. Два прогона на 13 запросах/мин: 4 ключа — ноль
отказов по 7-8 запросов на ключ; 12 ключей — те же ключи сыплются со 2-3-го обращения.

⚠️ Требует KB_DB на ОТДЕЛЬНУЮ базу: тест сносит файл. Такт душим в ноль — он тут не при чём.
  KB_DB=/tmp/kb_group.db KB_GRANT_MAX=0.01 KB_GRANT_MIN=0.01 \
  KB_GROUP_SIZE=4 KB_GROUP_WORK=3 KB_GROUP_PAUSE=2 python test_group.py
"""

import os
import sys
import time

import keybroker as k

M = "gemini-3.1-flash-lite"
KEYS = ["key-%d" % i for i in range(12)]


def grant():
    """Один грант. (None, 0.0) — очередь по такту, ждём. Возврат: (номер ключа, ожидание)."""
    for _ in range(4000):
        key, wait = k.acquire("t", "background", M, KEYS)
        if key is not None:
            return KEYS.index(key), 0.0
        if wait == 0.0:
            time.sleep(0.002)
            continue
        return None, wait  # пауза звена или кулдаун — наверх, это предмет теста
    raise SystemExit("не дождались гранта")


def cooldown(idx, secs):
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute(
        "INSERT INTO key_clock(key_hash, cooldown_until) VALUES(?,?) "
        "ON CONFLICT(key_hash) DO UPDATE SET cooldown_until=excluded.cooldown_until",
        (k._kh(KEYS[idx]), time.time() + secs),
    )
    c.execute("COMMIT")
    c.close()


def ban(idx):
    """Дневной кап/бан — в отличие от кулдауна это НАСОВСЕМ до конца PT-суток."""
    c = k._conn()
    c.execute("BEGIN IMMEDIATE")
    c.execute(
        "INSERT INTO usage(key_hash, model, pt_day, count, banned) VALUES(?,?,?,0,1) "
        "ON CONFLICT(key_hash, model, pt_day) DO UPDATE SET banned=1",
        (k._kh(KEYS[idx]), M, k._pt_day()),
    )
    c.execute("COMMIT")
    c.close()


def ok(cond, what, got=""):
    print("%-58s %-22s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    if not os.environ.get("KB_DB"):
        sys.exit("ОТКАЗ: задай KB_DB на тестовую базу — тест сносит файл")
    if k.GROUP_SIZE != 4:
        sys.exit("ОТКАЗ: тест писан под KB_GROUP_SIZE=4, а стоит %d" % k.GROUP_SIZE)
    try:
        os.remove(k.DB)
    except OSError:
        pass
    k.init()
    good = True

    # 1. Первое звено — только ключи 0-3, и по кругу внутри него, а не по всему пулу.
    seen = [grant()[0] for _ in range(8)]
    good &= ok(
        seen == [0, 1, 2, 3, 0, 1, 2, 3],
        "1. дежурит первое звено, круг внутри него",
        str(seen),
    )

    # 2. Отработав GROUP_WORK, звено уходит на паузу: выдачи нет, отдаётся ОЖИДАНИЕ.
    #    Это ключевое: рот на положительном ожидании спит и попыток вызова НЕ тратит.
    time.sleep(k.GROUP_WORK)
    keyno, wait = grant()
    good &= ok(
        keyno is None and 0 < wait <= k.GROUP_PAUSE + 0.2,
        "2. между звеньями ПАУЗА, а не выдача",
        "ожидание %.2fс" % (wait or 0),
    )

    # 3. Пауза кончилась — заступает ВТОРОЕ звено (4-7), первое молчит.
    time.sleep(wait + 0.05)
    seen = [grant()[0] for _ in range(4)]
    good &= ok(seen == [4, 5, 6, 7], "3. заступило второе звено", str(seen))

    # 4. Третье звено (8-11) — и дальше круг замыкается на первое, а не идёт в никуда.
    time.sleep(k.GROUP_WORK + k.GROUP_PAUSE)
    seen = [grant()[0] for _ in range(4)]
    good &= ok(seen == [8, 9, 10, 11], "4. заступило третье звено", str(seen))
    time.sleep(k.GROUP_WORK + k.GROUP_PAUSE)
    seen = [grant()[0] for _ in range(4)]
    good &= ok(seen == [0, 1, 2, 3], "5. круг замкнулся на первое звено", str(seen))

    # 6. ⭐ Ключи звена в КУЛДАУНЕ → ждём СВОЁ звено, а не убегаем в соседнее.
    #    Убежать значило бы засветить лишние ключи — ровно то, от чего звенья и заведены.
    for i in range(4):
        cooldown(i, 30)
    keyno, wait = grant()
    good &= ok(
        keyno is None and 0 < wait <= 30.5,
        "6. звено в кулдауне → ЖДЁМ его, не убегаем",
        "ожидание %.1fс" % (wait or 0),
    )

    # 7. ⛔ Ключи звена МЕРТВЫ насовсем (бан/кап) → ход уходит следующему звену немедленно.
    #    Тут нельзя отдать -1.0: для рта это «сдавайся до завтра», а соседи живые.
    #    Ровно так 24.07 вылетела 31 муха при расходе 6-8 из 440.
    for i in range(4):
        cooldown(i, -1)  # кулдаун снят, но...
        ban(i)  # ...ключ забанен насовсем
    seen = [grant()[0] for _ in range(4)]
    good &= ok(
        seen == [4, 5, 6, 7] and all(x is not None for x in seen),
        "7. мёртвое звено → сразу следующее, НЕ капитуляция",
        str(seen),
    )

    print(
        "\nVERDICT:",
        "OK — звенья дежурят по очереди, пауза держит" if good else "FAIL",
    )
    sys.exit(0 if good else 1)
