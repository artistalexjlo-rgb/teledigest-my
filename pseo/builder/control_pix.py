"""
control_pix.py — КОНТРОЛЬ метода до масштаба (BUILDER_RULES §0.3, «метод до стройки»).

Мишень: 94 br-мухи с литеральным «PIX» = ground truth (в них PIX точно есть).
Мерим RECALL фасет-разметки: на скольких из них facet_one реально ставит сущность PIX?
Промахи (PIX в тексте, но не в тегах) = дыра recall'а, которую обязан добрать ВЕКТОР.
Цифра, не «на глаз». Держит планку → корпус; дырявит → чиним разметку/семантику.

Запуск (VPS): /root/embed_ab/venv/bin/python control_pix.py > control_pix_out.log 2>&1
"""

import sqlite3

from facet import DB, facet_one

PIX_HINT = ("pix", "пикс")


def has_pix_tag(r):
    for e in r["sushnosti"]:
        if any(h in e["imya"].lower() for h in PIX_HINT):
            return True
    return any(any(h in z.lower() for h in PIX_HINT) for z in r["zadachi"])


def main():
    m = sqlite3.connect(DB)
    rows = m.execute(
        "SELECT id, ai_lesson FROM extracted_patterns "
        "WHERE country='br' AND ai_lesson LIKE '%PIX%' AND length(ai_lesson)>140 ORDER BY id"
    ).fetchall()
    m.close()
    total = len(rows)
    tagged, misses, dropped = 0, [], 0
    for fid, lesson in rows:
        r = facet_one(fid, lesson)
        if not r:
            dropped += 1
            continue
        if has_pix_tag(r):
            tagged += 1
        else:
            misses.append((fid, r["zadachi"], [e["imya"] for e in r["sushnosti"]]))

    proc = total - dropped
    print("\n=== КОНТРОЛЬ PIX ===")
    print(f"мишень (литеральный PIX в br): {total}")
    print(f"обработано (не отвалилось): {proc}")
    print(
        f"тег PIX поставлен: {tagged}/{proc}  recall={tagged/proc:.0%}" if proc else "0"
    )
    print(f"промахи (PIX в тексте, тега нет) — ДОБИРАЕТ ВЕКТОР: {len(misses)}")
    for fid, zad, ent in misses[:15]:
        print(f"  - {fid[:8]} задачи={zad} сущности={ent}")


if __name__ == "__main__":
    main()
