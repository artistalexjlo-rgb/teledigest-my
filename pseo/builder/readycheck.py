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
sys.path.insert(0, BASE)
from render import (  # noqa: E402  каталог вывода объявлен ОДИН раз — в render.py
    OUT as _OUT,
)

# ⛔ Своего `f"{BASE}/out"` здесь быть не должно. Гейт обязан смотреть ТУДА, КУДА ПИСАЛ
# рендер: в пульте вывод уходит в примонтированный каталог (PSEO_OUT), и со своей копией
# пути гейт проверял бы пустой `/app/out` и рапортовал «готово 0» при собранном сайте.
OUT = str(_OUT)


def scan(out=None):
    """Обойти собранное дерево и вернуть (страницы, битые ссылки, пустые, кракозябры).

    ⭐ Вынесено из `main()`, чтобы сторож звал ЭТУ проверку, а не свою копию правил:
    копия правила в двух местах — болезнь, на которой этот проект горел не раз. Рендер
    остаётся в `main()`: он тяжёлый и тесту не нужен.
    """
    out = out or OUT
    pages, broken, empty, moji = [], [], [], []
    for root, _, files in os.walk(f"{out}/ru"):
        for fn in files:
            if fn != "index.html":
                continue
            fp = os.path.join(root, fn)
            rel = (
                "/"
                + os.path.relpath(fp, out).replace("\\", "/").rsplit("/", 1)[0]
                + "/"
            )
            html = open(fp, encoding="utf-8").read()
            pages.append(rel)
            # Служебные страницы (шлюз клика `/go/luky/`) помечены noindex и тонкие ПО
            # ЗАМЫСЛУ: они не контент, а пересылка. Правило «пустая» к ним не применяем,
            # иначе гейт вечно рапортует дефект, которого нет.
            utility = 'name="robots" content="noindex' in html
            # пустая: нет h1 или тела < 400 символов
            if not utility and (
                "<h1>" not in html or len(re.sub(r"<[^>]+>", "", html)) < 400
            ):
                empty.append(rel)
            # кодировка: replacement char
            if "�" in html:
                moji.append(rel)
            # битые внутр-ссылки
            # ⛔ СТРОКУ ЗАПРОСА ОТРЕЗАЕМ. Шлюз клика (шаг 6) стоит на каждой странице как
            # `/ru/go/luky/?geo=..&shelf=..`, и без отреза гейт искал бы каталог с `?` в
            # имени — то есть объявил бы битой КАЖДУЮ дверь сайта, все ~2000.
            for href in set(re.findall(r'href="(/ru/[^"#]*)"', html)):
                path = href.split("?", 1)[0]
                tgt = os.path.join(out, path.strip("/"), "index.html")
                if not os.path.exists(tgt):
                    broken.append((rel, href))
    return pages, broken, empty, moji


def main():
    # 1) рендер всего через настоящий шаблон
    subprocess.run(
        [sys.executable, f"{BASE}/render.py", "--all", "2026-07-06"],
        cwd=BASE,
        capture_output=True,
        text=True,
    )

    pages, broken, empty, moji = scan()

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
