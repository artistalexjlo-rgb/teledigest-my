"""builder.py — pSEO builder (КАРКАС, ФАЗА 3).

Архитектура (см. memory roadmap_pseo_build_plan):
  load_qa → taxonomy → extract_facts → gate → render → freshness → (deploy)
СМЫСЛОВЫЕ шаги (taxonomy/extract/translate/judge) на VPS делает Claude Code headless
(`claude -p`, подписка), НЕ Gemini. Здесь — оркестрация + швы. Запускать НА VPS.

СТАТУС: каркас. Data-слой и LLM-слой — за чёткими швами:
  * load_qa: на VPS → Qdrant scroll; локально → --fixture <json>.
  * claude_agent: на VPS → subprocess `claude -p`; в --dry-run → детерминированный стаб.
Проверяемо локально: `python builder.py --geo br --dry-run` → out/staging/<path>/index.html.

Уведомления/лампа → пост в тех-канал дайджеста (notify(), пока stdout).
Управление = превью+деплой сейчас; веб-панель билдера потом. Ботов/CLI-команд НЕТ.
"""

import argparse
import hashlib
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
import render as R  # noqa: E402  (render_page/build, единый шаблон-fill)

# ── config (config-as-code; на VPS вынести в config/builder.py) ──────────────
CFG = {
    "qdrant_collection": "wisdom_base",
    "gate_min_facts": 6,          # планка на ГЛУБИНУ (не 4-минимум) — «больше контента»
    "gate_min_cluster": 8,        # тема живёт только при достаточном фактаже
    "languages": ["ru"],          # на VPS: широкий список (генерим broad + гейты + прунинг)
    "staging_dir": BASE / "out" / "staging",
    "built_hashes": BASE / "out" / ".built_hashes.json",
}


# ── уведомления (лампа) → тех-канал дайджеста (пока stdout) ───────────────────
def notify(msg: str) -> None:
    # TODO(VPS): POST в существующий тех-канал дайджеста (one-way). НЕ новый бот.
    print(f"[pseo] {msg}")


# ── LLM-слой: Claude Code agent (VPS) / стаб (dry-run) ───────────────────────
def claude_agent(prompt: str, schema: dict | None = None, *, dry: bool, stub=None):
    """Смысловой шаг через Claude Code headless. На VPS — subprocess `claude -p`
    со structured-output по schema; вызовы ПЕЙСИТЬ (батч/пауза, не упереть подписку).
    В dry-run возвращает детерминированный стаб (для проверки плумбинга)."""
    if dry:
        return stub
    # TODO(VPS): import subprocess; claude -p prompt --output-format json (schema) → parse.
    raise NotImplementedError("claude_agent: подключить `claude -p` на VPS")


# ── 1. load_qa: Qdrant (VPS) / fixture (локально) ────────────────────────────
def load_qa(geo: str, fixture: pathlib.Path | None):
    if fixture:
        rows = json.loads(fixture.read_text(encoding="utf-8"))
        return [r for r in rows if r.get("geo") == geo]
    # TODO(VPS): Qdrant scroll по CFG["qdrant_collection"], payload country==geo,
    # with_vectors=True. Поле опыта: `ai_lesson`, читать через `embedded_text` как
    # primary (см. memory gotcha_wisdom_payload_fields).
    raise NotImplementedError("load_qa: подключить Qdrant scroll на VPS (или дать --fixture)")


# ── 2. taxonomy: кластеры по эмбеддингам + Claude-нейминг (ШИРОКО, не чеклист) ─
def taxonomy(entries: list, geo: str, *, dry: bool) -> dict:
    """{intent_key: {"name":..., "entries":[...]}}. На VPS: HDBSCAN/kmeans по векторам
    → claude_agent именует кластеры в темы+интенты (широко). Кластер живёт при ≥min."""
    if dry:
        # стаб: группируем по уже размеченному полю фикстуры `topic`
        buckets: dict = {}
        for e in entries:
            buckets.setdefault(e["topic"], []).append(e)
        return {k: {"name": k.title(), "entries": v} for k, v in buckets.items()}
    clusters = _cluster_by_vectors(entries)  # TODO(VPS)
    return claude_agent(_taxonomy_prompt(clusters, geo), schema=None, dry=False)


def _cluster_by_vectors(entries):  # TODO(VPS): HDBSCAN на e["vector"]
    raise NotImplementedError


def _taxonomy_prompt(clusters, geo):
    return f"Назови кластеры опыта по {geo} в широкие темы+интенты (НЕ экспат-чеклист), как формулируют люди. Кластеры: {clusters}"


# ── 3. extract_facts: Claude чистит/дедупит, МАКСИМУМ контента ───────────────
def extract_facts(topic_entries: list, *, dry: bool) -> list:
    """[{q, a, a_plain, n, n_word, fact_hash, value_tag, content_date, source_count}].
    На VPS: claude_agent читает кластер → чистые дедуп-Q&A по-максимуму."""
    if dry:
        facts, seen = [], set()
        for e in topic_entries:
            text = (e.get("embedded_text") or e.get("ai_lesson") or "").strip()
            if not text:
                continue
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)
            facts.append({
                "q": e.get("q", "Вопрос"),
                "a": text, "a_plain": text,
                "n": e.get("source_count", 1), "n_word": "ответа",
                "fact_hash": h,
                "value_tag": e.get("value_tag", "hot_intent"),
                "content_date": e.get("content_date", ""),
                "source_count": e.get("source_count", 1),
            })
        return facts
    return claude_agent(_extract_prompt(topic_entries), schema=None, dry=False)


def _extract_prompt(entries):
    return f"Извлеки из этих сообщений МАКСИМУМ чистых дедуплицированных Q&A (живой опыт, без воды): {entries}"


# ── 4. gate: глубина + связность + не near-dup ───────────────────────────────
def gate(facts: list) -> tuple[bool, str]:
    if len(facts) < CFG["gate_min_facts"]:
        return False, f"тонко ({len(facts)}<{CFG['gate_min_facts']})"
    # TODO(VPS): coherence (centroid) + near-dup vs page_summaries + Claude-судья (Ferrari).
    return True, "ok"


# ── 5/6. сборка page-data под render.py + стейджинг ──────────────────────────
def build_page_data(geo, geo_name, intent_key, intent_name, facts, lang, updated):
    chips = []  # перелинковка на соседние темы — заполнится из taxonomy на полном прогоне
    return {
        "lang": lang, "geo": geo, "intent": intent_key,
        "geo_name": geo_name, "intent_name": intent_name,
        "path": f"/{lang}/{geo}/{intent_key}/",
        "updated": updated,  # реальная дата прогона (freshness)
        "title": f"{intent_name} в {geo_name} — живой опыт · Luky",
        "meta_desc": f"{intent_name}: живой опыт из чатов сообществ по теме «{intent_name}» в {geo_name}. Без воды.",
        "h1": f"{intent_name} в {geo_name}",
        "short_answer": facts[0]["a"] if facts else "",
        "faqs": facts,
        "chips": chips,
    }


def render_and_stage(page: dict) -> pathlib.Path:
    html = R.render_page(page)
    out = CFG["staging_dir"] / page["path"].strip("/") / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ── 7. freshness: пересобирать только изменившиеся темы ──────────────────────
def _load_hashes() -> dict:
    p = CFG["built_hashes"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_hashes(h: dict) -> None:
    CFG["built_hashes"].parent.mkdir(parents=True, exist_ok=True)
    CFG["built_hashes"].write_text(json.dumps(h, ensure_ascii=False, indent=0), encoding="utf-8")


def topic_hash(facts: list) -> str:
    return hashlib.md5("".join(sorted(f["fact_hash"] for f in facts)).encode()).hexdigest()


# ── оркестрация ──────────────────────────────────────────────────────────────
def run(geo: str, *, dry: bool, fixture: pathlib.Path | None, updated: str):
    notify(f"start geo={geo} dry={dry}")
    entries = load_qa(geo, fixture)
    geo_name = entries[0].get("geo_name", geo.upper()) if entries else geo.upper()
    tax = taxonomy(entries, geo, dry=dry)
    hashes = _load_hashes()
    stats = {"collected": len(entries), "built": 0, "gated": 0, "skipped_fresh": 0, "review": []}

    for ikey, t in tax.items():
        if len(t["entries"]) < CFG["gate_min_cluster"]:
            stats["gated"] += 1
            continue
        facts = extract_facts(t["entries"], dry=dry)
        ok, why = gate(facts)
        if not ok:
            stats["gated"] += 1
            notify(f"gate отсёк {geo}/{ikey}: {why}")
            continue
        h = topic_hash(facts)
        if hashes.get(f"{geo}/{ikey}") == h:
            stats["skipped_fresh"] += 1
            continue  # не изменилось — не регенерим
        for lang in CFG["languages"]:
            # TODO(VPS): lang!=source → claude_agent перевод (нативно) + гейт качества.
            page = build_page_data(geo, geo_name, ikey, t["name"], facts, lang, updated)
            out = render_and_stage(page)
            stats["built"] += 1
            stats["review"].append(str(out))
        hashes[f"{geo}/{ikey}"] = h

    _save_hashes(hashes)
    notify(f"done geo={geo}: собрано {stats['collected']} / тем-в-ревью {stats['built']} / "
           f"гейт отсёк {stats['gated']} / свежих-скип {stats['skipped_fresh']}")
    # TODO(VPS): первый прогон — НЕ автопаблиш; ревью в превью → деплой (git push в multyspeak-pages).
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="br")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fixture", type=pathlib.Path, default=None)
    ap.add_argument("--updated", default="06.2026")  # на VPS: реальная дата прогона
    a = ap.parse_args()
    if a.dry_run and not a.fixture:
        a.fixture = BASE / "fixtures" / "wisdom_br_sample.json"
    run(a.geo, dry=a.dry_run, fixture=a.fixture, updated=a.updated)


if __name__ == "__main__":
    main()
