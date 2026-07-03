"""
demand.py — снимает матрицу спроса с VPS в demand.json = {geo: {topic: count}}.
Источник правды для «СКОРО»: обещаем тему только если под неё РЕАЛЬНО есть данные.
Запускать перед wire.py (в деплой-пайплайне). Требует SSH на VPS.

ВАЖНО: в SQL строковый литерал — ОДИНАРНЫЕ кавычки ('id'), т.к. "id" = имя колонки
(в extracted_patterns есть колонка id) → "country=\"id\"" молча даёт 0 строк.
"""
import json
import pathlib
import subprocess

BASE = pathlib.Path(__file__).parent
VPS = "root@199.195.252.114"

# Грязный tag (669 значений) → канонические темы. Travel НЕ тема (помойка, Этап 4).
TAG2TOPIC = {
    "Finance": "finance",
    "Bureaucracy": "bureaucracy", "Visa": "bureaucracy", "Immigration": "bureaucracy",
    "Safety": "safety",
    "Shopping": "shopping", "Commerce": "shopping",
    "Health": "health",
    "Transport": "transport", "Logistics": "transport",
    "Housing": "housing", "Real Estate": "housing",
}


def pull():
    sql = ("SELECT country || '|' || tag || '|' || COUNT(*) "
           "FROM extracted_patterns "
           "WHERE ai_lesson IS NOT NULL AND length(ai_lesson) > 140 "
           "GROUP BY country, tag;")
    cmd = ["ssh", "-o", "ConnectTimeout=25", VPS,
           f'sqlite3 /home/teledigest/data/messages_fts.db "{sql}"']
    out = subprocess.check_output(cmd, text=True, encoding="utf-8")
    demand = {}
    for ln in out.splitlines():
        parts = ln.strip().split("|")
        if len(parts) != 3:
            continue
        geo, tag, cnt = parts[0], parts[1], int(parts[2])
        topic = TAG2TOPIC.get(tag)
        if not topic:
            continue  # не каноническая тема (Travel/шум) — не в спрос
        d = demand.setdefault(geo, {})
        d[topic] = d.get(topic, 0) + cnt
    (BASE / "demand.json").write_text(
        json.dumps(demand, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return demand


if __name__ == "__main__":
    d = pull()
    print(f"demand.json: {len(d)} гео")
    for geo in sorted(d, key=lambda g: -sum(d[g].values()))[:12]:
        top = sorted(d[geo].items(), key=lambda kv: -kv[1])
        print(f"  {geo}: " + ", ".join(f"{t}={c}" for t, c in top))
