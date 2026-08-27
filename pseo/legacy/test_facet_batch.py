"""Батч разметки facet: сшивка по индексу и отсев порчи.

Без ключей и сети: call() заглушён и отдаёт заранее заготовленный ответ. Проверяем ровно
то, ради чего таблица и валидация вводились, — что модель не может испортить вывод тихо.

Случаи:
  1. чистый ответ                 → все 25 разобраны, id совпали
  2. строки ПЕРЕПУТАНЫ местами    → сшивка по индексу, не по порядку → всё равно верно
  3. одна строка на 5 колонок     → эта муха в дед-леттер, остальные целы
  4. пустой perevod / нет zadachi → в дед-леттер (без задачи вид не построить)
  5. плохая роль сущности         → чинится на «обстоятельство», муха НЕ теряется
  6. модель вернула не rows       → вся пачка = инфра, мух не винить
  7. call вернул None трижды      → инфра, ровно 3 попытки, не больше
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import facet  # noqa: E402

CHUNK = [(1000 + i, "муха %d" % i) for i in range(facet.FACET_BATCH)]


def row(i, per="перевод", zad=None, ent=None, mesto=None, uslovie=None):
    return [
        str(i),
        per,
        zad if zad is not None else ["задача"],
        ent or [],
        mesto,
        uslovie,
    ]


def stub(answer, calls=None):
    """Подменяем сосок: считаем обращения и отдаём заготовку."""

    def _c(user, sysprompt, consumer=None, **kw):
        if calls is not None:
            calls.append(consumer)
        return answer

    facet.call = _c


def check(name, cond):
    print("%-58s %s" % (name, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    ok = True

    # 1. чистый ответ
    stub({"rows": [row(i) for i in range(facet.FACET_BATCH)]})
    recs, bad, reason = facet.facet_many(CHUNK)
    ok &= check(
        "1. чистая пачка: разобраны все, id на месте",
        len(recs) == facet.FACET_BATCH
        and not bad
        and reason is None
        and [r["id"] for r in recs] == [f for f, _ in CHUNK],
    )

    # 2. строки перепутаны — сшивка обязана идти по индексу
    shuffled = [row(i, per="перевод-%d" % i) for i in range(facet.FACET_BATCH)][::-1]
    stub({"rows": shuffled})
    recs, bad, reason = facet.facet_many(CHUNK)
    свои = all(r["perevod"] == "перевод-%d" % j for j, r in enumerate(recs))
    ok &= check(
        "2. строки задом наперёд: сшивка по индексу, не по порядку",
        len(recs) == facet.FACET_BATCH and свои,
    )

    # 3. одна строка короче на колонку
    rows = [row(i) for i in range(facet.FACET_BATCH)]
    rows[7] = rows[7][:5]
    stub({"rows": rows})
    recs, bad, reason = facet.facet_many(CHUNK)
    ok &= check(
        "3. строка на 5 колонок: только эта муха в дед-леттер",
        len(recs) == facet.FACET_BATCH - 1 and bad == [CHUNK[7][0]],
    )

    # 4. пустой перевод и пустые задачи
    rows = [row(i) for i in range(facet.FACET_BATCH)]
    rows[3] = row(3, per="   ")
    rows[4] = row(4, zad=[])
    stub({"rows": rows})
    recs, bad, reason = facet.facet_many(CHUNK)
    ok &= check(
        "4. пустой perevod / нет zadachi: обе в дед-леттер",
        len(recs) == facet.FACET_BATCH - 2
        and sorted(bad) == sorted([CHUNK[3][0], CHUNK[4][0]]),
    )

    # 5. чужая роль — чиним, но муху не теряем
    rows = [row(i) for i in range(facet.FACET_BATCH)]
    rows[2] = row(2, ent=[["CPF", "goal"], ["Busbud", "обход"]])
    stub({"rows": rows})
    recs, bad, reason = facet.facet_many(CHUNK)
    r2 = next(r for r in recs if r["id"] == CHUNK[2][0])
    ok &= check(
        "5. неизвестная роль → «обстоятельство», муха цела",
        not bad
        and r2["sushnosti"][0]["rol"] == "обстоятельство"
        and r2["sushnosti"][1]["rol"] == "обход",
    )

    # 6. ответ не той формы
    stub({"items": []})
    recs, bad, reason = facet.facet_many(CHUNK)
    ok &= check(
        "6. нет rows: вся пачка = инфра, мух не винить",
        not recs and not bad and reason is not None,
    )

    # 7. пул молчит — ровно три попытки
    calls = []
    stub(None, calls)
    recs, bad, reason = facet.facet_many(CHUNK)
    ok &= check(
        "7. call=None: инфра и РОВНО 3 попытки пачки",
        reason is not None and len(calls) == 3 and not bad,
    )

    print(
        "\nЭКОНОМИЯ: %d мух за 1 запрос вместо %d запросов"
        % (facet.FACET_BATCH, facet.FACET_BATCH)
    )
    print("VERDICT:", "OK — порча не проходит молча" if ok else "FAIL")
    sys.exit(0 if ok else 1)
