"""readycheck.py — СИГНАЛ готовности к деплою. Рендерит все data/ и ДЕТЕРМИНИРОВАННО
валидирует: битые внутр-ссылки / пустые страницы / кодировка / sitemap, НА ВСЕХ ЯЗЫКАХ.
Дизайн проверять не нужно (гарантирован шаблоном) — тут структура+контент-целостность.

Выхлоп: ready.json + печать «готово N, проблем K». Ноль LLM.
Запуск: python readycheck.py   (из pseo/tract/)

⛔ Переписан заново, не перенесён (юзер 28.08). Старый (`pseo/legacy/readycheck.py`)
обходил ТОЛЬКО `out/ru` и ловил битые ссылки ТОЛЬКО вида `/ru/...` — на 14-языковом сайте
это значило: гейт мог сказать «проблем 0», когда сломан любой из тринадцати остальных
языков. Сторож на это — `test_readycheck.py`, написан ДО этой правки.
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # …/pseo/tract
ROOT = os.path.dirname(HERE)  # …/pseo
sys.path.insert(0, HERE)
# Каталог вывода объявлен ОДИН раз, в render.py — своего здесь не заводим.
from render import OUT as _OUT  # noqa: E402

# ⛔ Своего `f"{ROOT}/out"` здесь быть не должно. Гейт обязан смотреть ТУДА, КУДА ПИСАЛ
# рендер: в пульте вывод уходит в примонтированный каталог (PSEO_OUT), и со своей копией
# пути гейт проверял бы пустой каталог и рапортовал «готово 0» при собранном сайте.
OUT = str(_OUT)

# Каталоги верхнего уровня в `out/`, которые НЕ языки — общие ассеты (render.copy_assets).
NOT_A_LANGUAGE = {"assets"}

# Строка запроса у ссылки ОТРЕЗАЕТСЯ. Шлюз клика стоит на каждой странице как
# `/<язык>/go/luky/?geo=..&shelf=..`, и без отреза гейт искал бы каталог с `?` в имени —
# то есть объявил бы битой каждую дверь сайта.
_HREF = re.compile(r'href="(/[a-z]{2}/[^"#]*)"')


def scan(out=None):
    """Обойти собранное дерево, ЛЮБОЙ ЯЗЫК, и вернуть (страницы, битые, пустые, кракозябры).

    Вынесено из `main()`, чтобы сторож звал ЭТУ проверку, а не свою копию правил: копия
    правила в двух местах — болезнь, на которой этот проект горел не раз. Рендер остаётся
    в `main()`: он тяжёлый и тесту не нужен.
    """
    out = out or OUT
    pages, broken, empty, moji = [], [], [], []
    if not os.path.isdir(out):
        return pages, broken, empty, moji
    langs = sorted(
        d
        for d in os.listdir(out)
        if d not in NOT_A_LANGUAGE and os.path.isdir(os.path.join(out, d))
    )
    for lang in langs:
        for root, _, files in os.walk(os.path.join(out, lang)):
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
                # Служебные страницы (шлюз клика) помечены noindex и тонкие ПО ЗАМЫСЛУ:
                # они не контент, а пересылка. Правило «пустая» к ним не применяем.
                utility = 'name="robots" content="noindex' in html
                if not utility and (
                    "<h1>" not in html or len(re.sub(r"<[^>]+>", "", html)) < 400
                ):
                    empty.append(rel)
                if "�" in html:
                    moji.append(rel)
                for href in set(_HREF.findall(html)):
                    path = href.split("?", 1)[0]
                    tgt = os.path.join(out, path.strip("/"), "index.html")
                    if not os.path.exists(tgt):
                        broken.append((rel, href))
    return pages, broken, empty, moji


def main():
    subprocess.run(  # 1) рендер всего через настоящий шаблон
        [sys.executable, f"{HERE}/render.py", "--all"],
        cwd=HERE,
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
        open(f"{ROOT}/ready.json", "w", encoding="utf-8"),
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
