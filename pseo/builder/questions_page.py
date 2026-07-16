"""questions_page.py — страница «что спрашивают в чатах» из вопросов-нарраций (бывший junk).
НЕ дамп: причёсывает наррации в живые вопросы + группирует по темам → редакторская история.
Ответов НЕ даём (не выдумываем) — посыл «а Луки-то знает». Воронка, не SEO-лошадь.
Запуск (VPS, вне окна 20-23): /root/embed_ab/venv/bin/python questions_page.py <geo> [--limit N]
"""

import json
import sqlite3
import sys

from facet import DB, is_junk
from keybroker import call

SYS = (
    "Вот НАРРАЦИИ-запросы из чатов (модель пересказала, что человек спрашивал). Для КАЖДОЙ верни:\n"
    '  "q" — короткий ЖИВОЙ вопрос на русском, как человек РЕАЛЬНО спросил бы в чате '
    "(разговорно, по делу, без канцелярита; сохрани суть и специфику);\n"
    '  "tema" — тема 2-4 слова для группировки (например «Аренда жилья», «Виза и документы», '
    "«Обмен денег», «Транспорт», «Связь», «Покупки», «Здоровье», «С детьми», «С питомцем»).\n"
    'Верни СТРОГО JSON: {"items":[{"q":"...","tema":"..."}, ...]} — по одному на каждую входную наррацию, в том же порядке.'
)


def load_questions(geo, limit):
    m = sqlite3.connect(DB)
    rows = [
        r[0]
        for r in m.execute(
            "SELECT ai_lesson FROM extracted_patterns WHERE country=? AND ai_lesson IS NOT NULL "
            "AND length(ai_lesson)>140 ORDER BY id",
            (geo,),
        )
    ]
    m.close()
    q = [t for t in rows if is_junk(t)]
    return q[:limit]


def run(geo, limit=45):
    narr = load_questions(geo, limit)
    items = []
    for i in range(0, len(narr), 15):
        batch = narr[i : i + 15]
        payload = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch))
        out = call(payload, SYS, consumer="questions")
        for it in (out or {}).get("items", []):
            if it.get("q") and it.get("tema"):
                items.append({"q": it["q"].strip(), "tema": it["tema"].strip()})
        print(f"  батч {i // 15 + 1}: +{len(items)} вопросов", flush=True)

    groups = {}
    for it in items:
        groups.setdefault(it["tema"], []).append(it["q"])
    # только темы, где ≥4 вопроса (осмысленный раздел, не одиночка)
    groups = {k: v for k, v in groups.items() if len(v) >= 4}

    import os

    os.makedirs("out_questions", exist_ok=True)
    page = {
        "geo": geo,
        "groups": [{"tema": k, "questions": v} for k, v in groups.items()],
    }
    with open(f"out_questions/{geo}.json", "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=1)
    print(
        f"\n{geo}: вопросов {len(items)}, тем-разделов {len(groups)} → out_questions/{geo}.json",
        flush=True,
    )


if __name__ == "__main__":
    geo = sys.argv[1] if len(sys.argv) > 1 else "br"
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 45
    run(geo, lim)
