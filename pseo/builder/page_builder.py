"""page_builder.py — БИЛДЕР как ИНСТРУМЕНТ. Одна пара (гео, категория) → одна страница.

Молоко (капля) ВПРЫСКИВАЕТСЯ: drop(user, sysprompt) -> dict|None.
Инструмент НЕ знает, откуда капля — ключей, квот, пейсинга, cooldown, HTTP тут НЕТ.
Всё это держит сиська (keybroker); она же кормит. См. __main__: drop = keybroker.call.

Базовая цепочка:
  вход (гео, категория)
    1. СЫРЬЁ     из таксономии — готовые мухи по секциям (не кластерим, не классифицируем)
    2. на секцию: КАПЛЯ → FAQ   (дедуп + перечисление пунктов, per-item grounding)
    3. ГРУНТ-ГЕЙТ claude — ОТДЕЛЬНЫЙ ресурс, не молоко → выкинуть необоснованные FAQ
    4. КАПЛЯ → ШАПКА            (title / h1 / meta / короткий ответ)
    5. СОБРАТЬ   page.json → out/{гео}_{категория}.json
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time

import numpy as np

# --- данные (только чтение готового; никакого добывания ключей) ---
DB = "/home/teledigest/data/messages_fts.db"  # extracted_patterns (мухи)
VEC = "/root/embed_ab/local_vec.db"  # векторы мух (дедуп)
TAXDB = "/root/pseo_builder/taxonomy.db"  # индекс свипера (секции)

# --- грунт-гейт: локальный claude -p (подписка ЮЗЕРА — ОТДЕЛЬНЫЙ общий ресурс) ---
CLAUDE_BIN = "/root/.local/bin/claude"
CLAUDE_ENV = "/root/.claude_env"
GATE_PACE = (
    6.0  # пауза после гейта — беречь подписку юзера (НЕ молоко сиськи, свой ресурс)
)

GEO_NAME = {
    "br": "Бразилия",
    "vn": "Вьетнам",
    "me": "Черногория",
    "id": "Индонезия",
    "gr": "Греция",
    "kr": "Южная Корея",
    "ph": "Филиппины",
    "de": "Германия",
    "gb": "Великобритания",
    "bg": "Болгария",
    "jp": "Япония",
    "by": "Беларусь",
    "fr": "Франция",
    "au": "Австралия",
    "ar": "Аргентина",
    "hu": "Венгрия",
    "at": "Австрия",
    "ru": "Россия",
    "cl": "Чили",
    "fi": "Финляндия",
}
INTENT_NAME = {
    "documents": "Документы и визы",
    "money": "Деньги и банки",
    "housing": "Жильё и аренда",
    "health": "Здоровье и медицина",
    "transport": "Транспорт",
    "safety": "Безопасность",
    "connectivity": "Связь и интернет",
    "shopping": "Покупки и цены",
    "food": "Еда",
    "attractions": "Что посмотреть",
    "culture": "Язык и культура",
    "work": "Работа",
    "education": "Образование",
    "shipping": "Посылки и почта",
}


# --------------------------------------------------------------------------- 1. СЫРЬЁ
def load_taxonomy_group(geo, category):
    """Секции категории из индекса свипера: {section: [{id,title,lesson,vec}]}.
    Организация уже в индексе (taxonomy.py) — билдер её не пересчитывает."""
    t = sqlite3.connect(TAXDB)
    rows = t.execute(
        "SELECT pattern_id, section FROM tax WHERE geo=? AND category=?",
        (geo, category),
    ).fetchall()
    t.close()
    if not rows:
        return {}
    vsel = sqlite3.connect(VEC)
    vmap = {}
    for pid, _ in rows:
        r = vsel.execute(
            "SELECT v FROM vec WHERE doc_id=? AND dim=1024", (pid,)
        ).fetchone()
        if r:
            vmap[pid] = np.frombuffer(r[0], dtype=np.float32)
    vsel.close()
    m = sqlite3.connect(DB)
    secs = {}
    for pid, sec in rows:
        if pid not in vmap:
            continue
        r = m.execute(
            "SELECT title, ai_lesson FROM extracted_patterns WHERE id=?", (pid,)
        ).fetchone()
        if not r or not r[1]:
            continue
        secs.setdefault(sec, []).append(
            {"id": pid, "title": r[0] or "", "lesson": r[1], "vec": vmap[pid]}
        )
    m.close()
    return secs


# -------------------------------------------------------------------- 2. FAQ (капля)
def dedupe(group, thresh=0.90):
    """Схлопнуть near-дубли (косинус >= thresh), сохранить РАЗНЫЕ пункты."""
    kept, keptv = [], []
    for p in group:
        v = p["vec"]
        v = v / (np.linalg.norm(v) + 1e-9)
        if all(float(v @ kv) < thresh for kv in keptv):
            kept.append(p)
            keptv.append(v)
    return kept


def verify(answer, cited_sources):
    """Детерминированный грунт-чек: числа и Latin-токены ответа есть в источниках?
    Ловит класс «Сбер-335» (выдуманное число/имя). Возврат: список неподтверждённых."""
    src = " ".join(cited_sources).lower()
    src_nums = set(re.findall(r"\d[\d.,]*", src))
    txt = re.sub(r"<[^>]+>", " ", answer)

    def norm(n):
        return n.rstrip(".,").replace(",", "")

    bad = []
    for num in re.findall(r"\d[\d.,]*", txt):
        nn = norm(num)
        if len(nn) < 2:
            continue
        if not any(norm(s).startswith(nn) or nn in norm(s) for s in src_nums):
            bad.append(num)
    for tok in set(re.findall(r"[A-Za-z][A-Za-z0-9.&]{2,}", txt)):
        t = tok.strip(".-/&").lower()
        if len(t) < 3:
            continue
        if t not in src:
            bad.append(tok)
    return bad


def plural(n):
    if n == 1:
        return "живой ответ"
    if 2 <= n <= 4:
        return "живых ответа"
    return "живых ответов"


WRITE_SYS = (
    "Тема «{topic}» ({geo}). На вход — набор УЖЕ-ГОТОВЫХ фактов (реальные советы, англ., "
    "каждый с ID). Это НЕ сырьё для пересказа — готовые пункты. НЕ сливай в один ответ, "
    "НЕ обобщай, НЕ усредняй.\n"
    "ПЕРЕВЕДИ и РАЗЛОЖИ:\n"
    "1) сформулируй ОДИН вопрос, который эти факты закрывают;\n"
    "2) ответ = ПЕРЕЧЕНЬ пунктов: каждый пункт — ОТДЕЛЬНЫЙ способ/случай/условие из СВОИХ "
    "фактов, на русском, живым языком; между собой пункты НЕ смешивать;\n"
    "3) число/название/цену пиши ТОЛЬКО в пункте, где оно реально в его источнике "
    "(перенос числа из пункта в пункт ЗАПРЕЩЁН);\n"
    "4) явные дубли схлопни, но РАЗНЫЕ условия — РАЗНЫЕ пункты; ничего не выдумывай.\n"
    'Верни JSON: {{"q":..,"items":[{{"point":"<рус, один факт с условием, можно <b>/'
    '<span class=\\"num\\">>","ids":["P0",..]}}],"a_plain":"<кратко 1-2 предложения>"}}'
)


def write_faq(group, geo_name, topic, drop):
    """Секция мух → один FAQ. КАПЛЯ переводит и раскладывает; per-item grounding режет
    выдуманные числа/имена (число проверяется против ИСТОЧНИКОВ СВОЕГО пункта)."""
    group = dedupe(group)[:20]  # разные пункты, кап под токены
    cand = [{"id": f"P{i}", "text": g["lesson"][:400]} for i, g in enumerate(group)]
    user = "ФАКТЫ:\n" + "\n".join(f'{c["id"]}: {c["text"]}' for c in cand)
    out = drop(user, WRITE_SYS.format(geo=geo_name, topic=topic))
    items = out.get("items") if out else None
    if not items:
        return None
    idmap = {f"P{i}": g for i, g in enumerate(group)}
    kept, allbad = [], []
    for it in items:
        pt = (it.get("point") or "").strip()
        if len(pt) < 4:
            continue
        cited = [c for c in it.get("ids", []) if c in idmap]
        src = [idmap[c]["lesson"] for c in cited] or [g["lesson"] for g in group]
        allbad += verify(pt, src)
        kept.append(pt)
    if not kept:
        return None
    a = "<ul>" + "".join(f"<li>{pt}</li>" for pt in kept) + "</ul>"
    a_plain = (out.get("a_plain") or " ".join(kept))[:400]
    n = len(kept)
    return {
        "q": out.get("q", ""),
        "a": a,
        "a_plain": a_plain,
        "n": n,
        "n_word": plural(n),
        "_unsupported": allbad,
    }


# --------------------------------------------------------------- 3. ГРУНТ-ГЕЙТ (claude)
def _claude_env():
    env = dict(os.environ)
    env.update({"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "PYTHONUTF8": "1"})
    try:
        for ln in open(CLAUDE_ENV, encoding="utf-8"):
            ln = ln.strip()
            if ln.startswith("export "):
                ln = ln[7:]
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def gate(faqs, pats, geo_name, intent):
    """claude -p сверяет FAQ с источниками, роняет выдуманное/оффтоп. ОДИН вызов на
    страницу, fail-open (claude недоступен → ничего не роняем)."""
    src = "\n".join(f"- {p['title']}: {p['lesson'][:400]}" for p in pats[:60])
    pg = "\n".join(f"{i}. Q: {f['q']} | A: {f['a_plain']}" for i, f in enumerate(faqs))
    prompt = (
        f"Грунт-чек справочной страницы про «{intent}» ({geo_name}).\n"
        f"ИСТОЧНИКИ (реальные советы, англ.):\n{src}\n\n"
        f"FAQ СТРАНИЦЫ (рус.):\n{pg}\n\n"
        'Верни ТОЛЬКО JSON: {"drop":[индексы FAQ с выдуманными/искажёнными фактами ИЛИ не по '
        f'теме «{intent}»], "note":"1 строка"}}. Кириллица vs англ. источник — сверяй по СМЫСЛУ '
        "и числам, не по буквальному совпадению. Пустой drop если всё чисто."
    )
    try:
        r = subprocess.run(
            [CLAUDE_BIN, "-p", prompt],
            env=_claude_env(),
            capture_output=True,
            text=True,
            timeout=150,
        )
        m = re.search(r"\{.*\}", r.stdout, re.S)
        dropped_idx = set(json.loads(m.group(0)).get("drop", [])) if m else set()
    except Exception as e:
        print("  gate fail-open (claude недоступен):", str(e)[:80], flush=True)
        return faqs, []
    time.sleep(GATE_PACE)
    dropped = [
        faqs[i]["q"] for i in dropped_idx if isinstance(i, int) and i < len(faqs)
    ]
    kept = [f for i, f in enumerate(faqs) if i not in dropped_idx]
    return kept, dropped


# ------------------------------------------------------------------ 4. ШАПКА (капля)
SYNTH_SYS = (
    "Дан список вопросов-ответов справочной страницы про «{geo}», тема «{topic}». "
    "Напиши шапку СТРОГО по содержанию ответов, без новых фактов. "
    "short_answer ОБЯЗАН начинаться с самого конкретного практического факта "
    "(число/название/действие), как живой совет от бывалого. ЗАПРЕЩЕНЫ вводные-вода: "
    "«управление финансами требует…», «мы собрали актуальную информацию», «важно учитывать нюансы». "
    'Не обобщай — сразу по делу. JSON: {{"title":.. (SEO, страна+тема, <=70 симв),"h1":..,'
    '"meta_desc":.. (~150 симв),"short_answer":.. (2-4 предложения, <strong> на ключевом)}}'
)


def synth(faqs, geo_name, topic, drop):
    body = "\n".join(f'- {f["q"]} {f["a_plain"]}' for f in faqs)
    out = drop(body, SYNTH_SYS.format(geo=geo_name, topic=topic))
    return out or {}


# --------------------------------------------------------------- 5. СБОРКА / ЦЕПОЧКА
def build_page(geo, category, drop):
    """Полная цепочка инструмента. drop — капля (user, sysprompt) -> dict|None.
    Возврат: page-dict или None (нечего публиковать)."""
    intent_name = INTENT_NAME.get(category, category)
    geo_name = GEO_NAME.get(geo, geo)

    secs = load_taxonomy_group(geo, category)  # 1. сырьё
    if not secs:
        print("нет мух в таксономии для пары — запусти taxonomy.py")
        return None
    print(
        f"{geo}/{category}: {sum(len(v) for v in secs.values())} мух, секций {len(secs)}"
    )

    faqs, allpats = [], []
    for sec_id in sorted(secs, key=lambda s: -len(secs[s])):
        group = secs[sec_id]
        allpats += group
        f = write_faq(group, geo_name, intent_name, drop)  # 2. капля → FAQ
        if not f:
            print(f"  секция {sec_id}: skip")
            continue
        print(f"  секция {sec_id}: {f['n']} пунктов  {f['q'][:48]}")
        faqs.append(f)
    if not faqs:
        print("ноль FAQ")
        return None

    faqs, gate_dropped = gate(faqs, allpats, geo_name, intent_name)  # 3. грунт-гейт
    if gate_dropped:
        print("  грунт-гейт выкинул:", " | ".join(d[:40] for d in gate_dropped))
    if len(faqs) < 2:
        print(f"парк: {len(faqs)} FAQ (<2) — не публикуем")
        return None

    head = synth(faqs, geo_name, intent_name, drop)  # 4. капля → шапка
    return {  # 5. сборка
        "lang": "ru",
        "geo": geo,
        "intent": category,
        "geo_name": geo_name,
        "intent_name": intent_name,
        "path": f"/ru/{geo}/{category}/",
        "updated": "07.2026",
        "title": head.get("title", f"{intent_name} в {geo_name} — Luky"),
        "meta_desc": head.get("meta_desc", ""),
        "h1": head.get("h1", f"{intent_name} в {geo_name}"),
        "short_answer": head.get("short_answer", ""),
        "faqs": [{k: f[k] for k in ("q", "a", "a_plain", "n", "n_word")} for f in faqs],
        "chips": [],
        "cta_text": (
            "Твоя ситуация особая? Спроси ассистента Luky на своём языке — без воды. "
            "Базу пополняем каждый день: если по твоему вопросу есть живой опыт, он его найдёт."
        ),
    }


if __name__ == "__main__":
    geo, category = sys.argv[1], sys.argv[2]
    # КОРМИТ СИСЬКА: капля = единственная дверь мозга. Инструмент её только зовёт.
    from keybroker import call as drop

    page = build_page(geo, category, drop)
    if not page:
        sys.exit(1)
    os.makedirs("out", exist_ok=True)
    fn = f"out/{geo}_{category}.json"
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(page, fh, ensure_ascii=False, indent=1)
    print(f"-> {fn}  ({len(page['faqs'])} FAQ)")
