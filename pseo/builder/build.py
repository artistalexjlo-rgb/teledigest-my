"""
build.py — pSEO builder v1 (runs on VPS host via bge-m3 venv).

Конвейер на ОДНУ страницу-единицу (набор паттернов одной фокусной темы):
  1. cluster   — bge-m3 векторы из local_vec.db → под-вопросы (FAQ-группы)
  2. write     — gemini-3.1-flash-lite на КАЖДУЮ группу: сожми заданные сниппеты
                 + процитируй заданные ID (писатель НЕ доверенный, формат узкий)
  3. verify    — детерминированно: числа/Latin-названия из ответа есть в
                 процитированных источниках? нет → флаг unsupported
  4. synth     — flash-lite: title/h1/meta/short_answer строго по ответам
  5. emit      — page-data JSON (схема pseo/render.py) в out/<geo>_<topic>.json

Умный adversarial-чекер на смысловом остатке — СЛОЙ 2 (тут нет). v1 ловит
числовые/именные выдумки детерминированно (класс «Сбер-335»).

Запуск (на VPS):
    /root/embed_ab/venv/bin/python build.py <geo> <tag> <intent_slug> "<intent_name>" [--limit N]
Ключи Gemini берутся из env контейнера bots-grab (docker exec printenv).
"""

import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import KMeans

# У VPS битый/медленный IPv6, а urllib (в отличие от curl) не делает happy-eyeballs —
# пробует AAAA первым и виснет ~40с на КАЖДЫЙ вызов, потом падает на IPv4. Форсируем IPv4.
_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [
    r for r in _orig_gai(*a, **k) if r[0] == socket.AF_INET
]

DB = "/home/teledigest/data/messages_fts.db"

# Общий бюджет ключей с ПРОДОМ (вечерняя экстракция на тех же flash-ключах).
# Билдер пишет расход в gemini_quota (тот же key_hash=sha1[:16]) и оставляет РЕЗЕРВ.
# См. feedback_shared_key_budget_discipline.
RESERVE = 80  # оставить проду на (ключ, модель)/день
RPD = {
    "gemini-3.1-flash-lite": 500,
    "gemini-2.5-flash-lite": 500,
    "gemini-2.5-flash": 250,
}
VEC = "/root/embed_ab/local_vec.db"
MODEL = "gemini-3.1-flash-lite"
SLEEP = 1.6  # пейсинг RPM free-tier

# Грунт-гейт: локальный claude -p на VPS (подписка ЮЗЕРА — общий ресурс, не грузить).
# 1 вызов на СТРАНИЦУ, последовательно, с паузой, fail-open. См. feedback_shared_key_budget.
CLAUDE_BIN = "/root/.local/bin/claude"
CLAUDE_ENV = "/root/.claude_env"
GATE_PACE = 6.0  # пауза после вызова гейта — беречь подписку юзера

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


# --------------------------------------------------------------------------- keys
def get_keys():
    cid = subprocess.check_output(
        "docker ps --format '{{.Names}}' | grep bots-grab | head -1",
        shell=True,
        text=True,
    ).strip()
    env = subprocess.check_output(["docker", "exec", cid, "printenv"], text=True)
    keys = [
        ln.split("=", 1)[1]
        for ln in env.splitlines()
        if ln.startswith("GEMINI_API_KEY") and "=" in ln
    ]
    if not keys:
        sys.exit("no GEMINI keys in container env")
    return keys


_KEYS = None
_key_ready = {}  # key -> ts когда ключ снова можно дёргать (пейсинг под RPM/ключ)
KEY_INTERVAL = (
    6.0  # мин интервал между вызовами ОДНОГО ключа (~10 RPM/ключ, под free-tier)
)
# Ключи делят проект → RPM ОБЩИЙ. Per-key пейсинга мало — нужен глобальный аггрегат-лимит,
# иначе 13 ключей × частота прошибают проектный RPM → 429-шторм. ~7 вызовов/мин суммарно.
GLOBAL_MIN = 8.0
GLOBAL_ON_429 = (
    15.0  # 429 = проектный RPM выбран → глобально выждать окно (не долбить ключи)
)
_last_global = [0.0]


def _kh(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _today_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _quota_all(model):
    """{key_hash: (count, banned)} за сегодня — ОДИН запрос (не 13 коннектов на вызов;
    прод-БД пишется ботом → множественные коннекты виснут на локе). busy_timeout короткий.
    """
    try:
        con = sqlite3.connect(DB, timeout=3)
        con.execute("PRAGMA busy_timeout=2000")
        rows = con.execute(
            "SELECT key_hash, count, banned FROM gemini_quota WHERE model=? AND date_utc=?",
            (model, _today_utc()),
        ).fetchall()
        con.close()
        return {r[0]: (int(r[1] or 0), bool(r[2])) for r in rows}
    except Exception:
        return {}


def _quota_inc(key, model):
    """Записать наш расход в ОБЩИЙ учёт — чтобы ротатор экстракции его видел."""
    try:
        con = sqlite3.connect(DB, timeout=5)
        con.execute("PRAGMA busy_timeout=4000")
        con.execute(
            "INSERT INTO gemini_quota(key_hash,model,date_utc,count) VALUES(?,?,?,1) "
            "ON CONFLICT(key_hash,model,date_utc) DO UPDATE SET count=count+1",
            (_kh(key), model, _today_utc()),
        )
        con.commit()
        con.close()
    except Exception as e:
        print("  quota_inc err", str(e)[:80])


def _cap(model):
    return RPD.get(model, 250) - RESERVE


def _acquire_key(model):
    """Взять самый 'остывший' ключ, у которого сегодня ещё есть бюджет (под резервом
    для прода) и который не забанен. Один ключ не дёргается чаще KEY_INTERVAL (RPM).
    None — если по всем ключам бюджет модели выбран (резерв проду) → стоп."""
    global _KEYS
    if _KEYS is None:
        _KEYS = get_keys()
        for k in _KEYS:
            _key_ready[k] = 0.0
    cap = _cap(model)
    q = _quota_all(model)
    elig = [
        k
        for k in _KEYS
        if q.get(_kh(k), (0, False))[0] < cap and not q.get(_kh(k), (0, False))[1]
    ]
    if not elig:
        return None
    key = min(elig, key=lambda k: _key_ready[k])
    wait = _key_ready[key] - time.time()
    if wait > 0:
        time.sleep(wait)
    # глобальный аггрегат-интервал (общий RPM проекта, а не только per-key)
    gwait = _last_global[0] + GLOBAL_MIN - time.time()
    if gwait > 0:
        time.sleep(gwait)
    _last_global[0] = time.time()
    _key_ready[key] = time.time() + KEY_INTERVAL
    return key


def gemini_json(user, sysprompt, model=MODEL, timeout=60):
    """generateContent в JSON. Проактивный per-key пейсинг под RPM + общий бюджет:
    читает gemini_quota (стоп на резерве проду), пишет расход туда же. 429 не ожидается;
    если прилетел — ключ в кул-даун."""
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": sysprompt}]},
        "generationConfig": {"responseMimeType": "application/json"},
    }
    data = json.dumps(payload).encode()
    for _ in range(4):  # ограниченные ретраи — вызов не должен висеть вечно
        key = _acquire_key(model)
        if key is None:
            print(
                f"  бюджет модели {model} выбран (резерв {RESERVE}/ключ проду) — стоп"
            )
            return None
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={key}"
        )
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            api = json.loads(r.read())
            _quota_inc(key, model)  # успех — записать наш расход в общий учёт
            raw = api["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(re.sub(r"```json|```", "", raw).strip())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # проектный RPM выбран — глобально выждать окно (долбить ключи бесполезно)
                _last_global[0] = time.time() + GLOBAL_ON_429
                continue
            print("  HTTP", e.code, str(e)[:120])
            return None
        except Exception as e:
            print("  err", str(e)[:120])
            return None
    print("  429 держится после ретраев — пропуск FAQ")
    return None


# ----------------------------------------------------------------------- data load
def load_cell(geo, tag, limit):
    m = __import__("sqlite3").connect(DB)
    rows = m.execute(
        "SELECT id,title,ai_lesson FROM extracted_patterns "
        "WHERE country=? AND tag=? AND ai_lesson IS NOT NULL AND length(ai_lesson)>140 "
        "ORDER BY length(ai_lesson) DESC LIMIT ?",
        (geo, tag, limit),
    ).fetchall()
    return [{"id": r[0], "title": r[1], "lesson": r[2]} for r in rows]


def load_vectors(pats):
    v = __import__("sqlite3").connect(VEC)
    out = []
    for p in pats:
        r = v.execute(
            "SELECT v FROM vec WHERE doc_id=? AND dim=1024", (p["id"],)
        ).fetchone()
        if r:
            p["vec"] = np.frombuffer(r[0], dtype=np.float32)
            out.append(p)
    return out


# ------------------------------------------------------------------------ clustering
def cluster(pats, k):
    X = np.vstack([p["vec"] for p in pats])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
    groups = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        cen = km.cluster_centers_[c]
        cen = cen / (np.linalg.norm(cen) + 1e-9)
        order = idx[np.argsort(-(X[idx] @ cen))]
        groups.append([pats[i] for i in order[:6]])  # топ-6 у центра = кандидаты
    return groups


# -------------------------------------------------------------------------- verify
def verify(answer, cited_sources):
    """Детерминированно: числа и Latin-токены ответа есть в источниках?
    (Кириллица EN→RU не проверяется substring'ом — ловим числа + Latin-имена,
    это и есть класс «Сбер-335».)"""
    src = " ".join(cited_sources).lower()
    src_nums = set(re.findall(r"\d[\d.,]*", src))
    txt = re.sub(
        r"<[^>]+>", " ", answer
    )  # срезать HTML-разметку (span/class/num — не факты)

    def norm(n):
        return n.rstrip(".,").replace(",", "")

    bad = []
    for num in re.findall(r"\d[\d.,]*", txt):
        nn = norm(num)
        if len(nn) < 2:
            continue  # одиночные цифры (1-2-3 в списках) не считаем
        if not any(norm(s).startswith(nn) or nn in norm(s) for s in src_nums):
            bad.append(num)
    for tok in set(re.findall(r"[A-Za-z][A-Za-z0-9.&]{2,}", txt)):
        t = tok.strip(".-/&").lower()
        if len(t) < 3:
            continue
        if t not in src:
            bad.append(tok)
    return bad


# ---------------------------------------------------------------------------- write
WRITE_SYS = (
    "Ты пишешь FAQ-блок справочной страницы про «{geo}», тема «{topic}». На вход — "
    "сниппеты с ID (реальные советы на английском). Сформулируй ОДИН вопрос, который "
    "эти сниппеты покрывают, и ответь на русском СТРОГО по ним. ОБЯЗАТЕЛЬНО вставляй "
    "конкретные числа/названия из сниппетов (цены, проценты, названия банков/приложений/мест) — "
    "не обобщай вместо конкретики. Живой язык, не телеграф через двоеточия. Не добавляй "
    'фактов вне сниппетов. Верни JSON: {{"q":..,"a":.. (можно <b> и <span class=\\"num\\">),'
    '"a_plain":.. (короткий plain),"cited_ids":["P0",..]}}'
)


def plural(n):
    if n == 1:
        return "живой ответ"
    if 2 <= n <= 4:
        return "живых ответа"
    return "живых ответов"


def write_faq(group, geo_name, topic):
    cand = [{"id": f"P{i}", "text": g["lesson"][:420]} for i, g in enumerate(group)]
    user = "СНИППЕТЫ:\n" + "\n".join(f'{c["id"]}: {c["text"]}' for c in cand)
    out = gemini_json(user, WRITE_SYS.format(geo=geo_name, topic=topic))
    time.sleep(SLEEP)
    if not out or "a" not in out:
        return None
    idmap = {f"P{i}": g for i, g in enumerate(group)}
    cited = [c for c in out.get("cited_ids", []) if c in idmap]
    sources = [idmap[c]["lesson"] for c in cited] or [g["lesson"] for g in group]
    bad = verify(out["a"], sources)
    n = max(len(cited), 1)
    return {
        "q": out.get("q", ""),
        "a": out["a"],
        "a_plain": out.get("a_plain", ""),
        "n": n,
        "n_word": plural(n),
        "_unsupported": bad,
    }


# ---------------------------------------------------------------------------- synth
SYNTH_SYS = (
    "Дан список вопросов-ответов справочной страницы про «{geo}», тема «{topic}». "
    "Напиши шапку СТРОГО по содержанию ответов, без новых фактов. "
    "short_answer ОБЯЗАН начинаться с самого конкретного практического факта "
    "(число/название/действие), как живой совет от бывалого. ЗАПРЕЩЕНЫ вводные-вода: "
    "«управление финансами требует…», «мы собрали актуальную информацию», «важно учитывать нюансы». "
    'Не обобщай — сразу по делу. JSON: {{"title":.. (SEO, страна+тема, <=70 симв),"h1":..,'
    '"meta_desc":.. (~150 симв),"short_answer":.. (2-4 предложения, <strong> на ключевом)}}'
)

# Топик-фокус: умный проход (2.5-flash) выкидывает FAQ не по теме (мис-тег).
FOCUS_SYS = (
    "Страница узко про «{topic}» ({geo}). Дан список вопросов с номерами. Верни JSON "
    '{{"keep":[номера строго по теме «{topic}»],"drop":[номера не по теме — другая '
    "рубрика: покупки/цены на технику, недвижимость-инвестиции, транспорт и т.п.]}}. "
    "Будь строг: лучше выкинуть пограничное, чем размыть фокус страницы."
)


def focus_filter(faqs, geo_name, topic):
    qlist = "\n".join(f"{i}. {f['q']}" for i, f in enumerate(faqs))
    out = gemini_json(
        qlist, FOCUS_SYS.format(geo=geo_name, topic=topic), model="gemini-2.5-flash"
    )
    time.sleep(SLEEP)
    if not out or "keep" not in out:
        return faqs, []  # фильтр не отработал — не режем вслепую
    keep = set(out.get("keep", []))
    dropped = [faqs[i]["q"] for i in range(len(faqs)) if i not in keep]
    return [f for i, f in enumerate(faqs) if i in keep], dropped


def synth(faqs, geo_name, topic):
    body = "\n".join(f'- {f["q"]} {f["a_plain"]}' for f in faqs)
    out = gemini_json(body, SYNTH_SYS.format(geo=geo_name, topic=topic))
    time.sleep(SLEEP)
    return out or {}


def _claude_env():
    env = dict(os.environ)
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
    """Слой-2б: локальный claude -p сверяет FAQ с источниками, роняет выдуманные/оффтоп.
    ОДИН вызов на страницу, fail-open (упал → ничего не роняем, помечаем note)."""
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
        drop = set(json.loads(m.group(0)).get("drop", [])) if m else set()
    except Exception as e:
        print("  gate fail-open (claude недоступен):", str(e)[:80], flush=True)
        return faqs, []
    time.sleep(GATE_PACE)  # беречь подписку юзера
    dropped = [faqs[i]["q"] for i in drop if isinstance(i, int) and i < len(faqs)]
    kept = [f for i, f in enumerate(faqs) if i not in drop]
    return kept, dropped


# ----------------------------------------------------------------------------- main
def main():
    geo, tag, slug, intent_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 70
    )
    geo_name = GEO_NAME.get(geo, geo)

    pats = load_cell(geo, tag, limit)
    pats = load_vectors(pats)
    print(f"{geo}/{tag}: {len(pats)} паттернов с вектором")
    if len(pats) < 8:
        sys.exit("слишком мало векторов — нечего кластеровать")
    k = min(7, max(4, len(pats) // 9))
    groups = cluster(pats, k)
    print(f"кластеров: {len(groups)}")

    faqs = []
    for i, g in enumerate(groups):
        f = write_faq(g, geo_name, intent_name)
        if not f:
            print(f"  [{i}] write FAIL")
            continue
        flag = f"⚠ unsupported={f['_unsupported']}" if f["_unsupported"] else "ok"
        print(f"  [{i}] n={f['n']} {flag}  {f['q'][:50]}")
        faqs.append(f)
    if not faqs:
        sys.exit("ноль FAQ")

    # Слой 2а — топик-фокус: выкинуть мис-тегнутые FAQ (не по теме)
    faqs, dropped = focus_filter(faqs, geo_name, intent_name)
    if dropped:
        print("  topic-focus выкинул:", " | ".join(d[:40] for d in dropped))
    if not faqs:
        sys.exit("после топик-фокуса ноль FAQ")

    # Слой 2б — грунт-гейт (локальный claude -p): выкинуть выдуманное/искажённое
    faqs, gate_dropped = gate(faqs, pats, geo_name, intent_name)
    if gate_dropped:
        print("  грунт-гейт выкинул:", " | ".join(d[:40] for d in gate_dropped))
    if not faqs:
        sys.exit("после грунт-гейта ноль FAQ")

    head = synth(faqs, geo_name, intent_name)
    SIBL = [
        ("bureaucracy", "🛂", "Документы и виза"),
        ("housing", "🏠", "Жильё и аренда"),
        ("safety", "🛡", "Безопасность"),
        ("finance", "💰", "Финансы"),
        ("transport", "🚕", "Транспорт"),
        ("health", "🩺", "Медицина"),
    ]
    chips = [
        {"icon": ic, "label": lb, "url": f"/ru/{geo}/{s}/", "soon": True}
        for s, ic, lb in SIBL
        if s != slug
    ][:5]
    page = {
        "lang": "ru",
        "geo": geo,
        "intent": slug,
        "geo_name": geo_name,
        "intent_name": intent_name,
        "path": f"/ru/{geo}/{slug}/",
        "updated": "06.2026",
        "title": head.get("title", f"{intent_name} в {geo_name} — Luky"),
        "meta_desc": head.get("meta_desc", ""),
        "h1": head.get("h1", f"{intent_name} в {geo_name}"),
        "short_answer": head.get("short_answer", ""),
        "faqs": [{k: f[k] for k in ("q", "a", "a_plain", "n", "n_word")} for f in faqs],
        "chips": chips,
        "cta_text": (
            "Твоя ситуация особая? Спроси ассистента Luky на своём языке — без воды. "
            "Базу пополняем каждый день: если по твоему вопросу есть живой опыт, он его найдёт."
        ),
        "_audit": {
            "faq_unsupported": {
                i: f["_unsupported"] for i, f in enumerate(faqs) if f["_unsupported"]
            }
        },
    }
    import os

    os.makedirs("out", exist_ok=True)
    fn = f"out/{geo}_{slug}.json"
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(page, fh, ensure_ascii=False, indent=1)
    print(f"-> {fn}  ({len(faqs)} FAQ)")


if __name__ == "__main__":
    main()
