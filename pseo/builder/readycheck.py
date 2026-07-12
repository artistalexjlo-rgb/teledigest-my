"""readycheck.py — СИГНАЛ готовности к деплою. Рендерит все data/ (октагон-шаблон) и
ДЕТЕРМИНИРОВАННО валидирует: битые внутр-ссылки / пустые страницы / кодировка / sitemap.
Дизайн проверять не нужно (гарантирован шаблоном) — тут структура+контент-целостность.

Выхлоп: ready.json + печать «готово N, проблем K». Ноль LLM.
Запуск: python readycheck.py   (из pseo/)
"""

import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pseo
OUT = f"{BASE}/out"


def main():
    # 1) рендер всего через настоящий шаблон
    subprocess.run(
        [sys.executable, f"{BASE}/render.py", "--all", "2026-07-06"],
        cwd=BASE,
        capture_output=True,
        text=True,
    )

    pages, broken, empty, moji = [], [], [], []
    for root, _, files in os.walk(f"{OUT}/ru"):
        for fn in files:
            if fn != "index.html":
                continue
            fp = os.path.join(root, fn)
            rel = (
                "/"
                + os.path.relpath(fp, OUT).replace("\\", "/").rsplit("/", 1)[0]
                + "/"
            )
            html = open(fp, encoding="utf-8").read()
            pages.append(rel)
            # пустая: нет h1 или тела < 400 символов
            if "<h1>" not in html or len(re.sub(r"<[^>]+>", "", html)) < 400:
                empty.append(rel)
            # кодировка: replacement char
            if "�" in html:
                moji.append(rel)
            # битые внутр-ссылки
            for href in set(re.findall(r'href="(/ru/[^"#]*)"', html)):
                tgt = os.path.join(OUT, href.strip("/"), "index.html")
                if not os.path.exists(tgt):
                    broken.append((rel, href))

    # sitemap валиден + все loc резолвятся
    sm = f"{OUT}/sitemap.xml"
    sm_locs = (
        re.findall(r"<loc>([^<]+)</loc>", open(sm, encoding="utf-8").read())
        if os.path.exists(sm)
        else []
    )

    problems = len(broken) + len(empty) + len(moji)
    ready = len(pages) - len(
        {p for p in empty} | {p for p, _ in broken} | {p for p in moji}
    )
    rep = {
        "страниц_всего": len(pages),
        "готово_к_деплою": ready,
        "проблем": problems,
        "битых_ссылок": len(broken),
        "пустых": len(empty),
        "кодировка_бита": len(moji),
        "sitemap_url": len(sm_locs),
    }
    json.dump(
        rep,
        open(f"{BASE}/ready.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if broken:
        print("\n⛔ битые ссылки (первые 10):")
        for src, h in broken[:10]:
            print(f"  {src} → {h}")
    if empty:
        print(f"\n⚠ пустые (первые 10): {empty[:10]}")
    if moji:
        print(f"\n⚠ кодировка (первые 10): {moji[:10]}")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
