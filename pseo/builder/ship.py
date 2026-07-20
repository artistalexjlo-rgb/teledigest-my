"""ship.py — ОДНА команда деплой-тракта (десктоп, by-call). Идеальный тракт при объёме:
руками только триггер, всё остальное — конвейер с гейтом качества.

  1. pull   — забрать built-данные ВСЕХ гео с VPS (out_facet/, out_questions/);
  2. pages  — pages.py --all → portal-data (оба контура: факт-темы, вопрос-контур, хабы);
  3. check  — readycheck: рендер через октагон-шаблон + детерм-валидация;
  4. gate   — гео с проблемами (битые/пустые/кодировка) НЕ едут; чистые — едут;
  5. push   — чистые гео-блоки → pages-репо → git push → CF авто-деплой.

Запуск: python builder/ship.py [--dry] [--geo br,vn]   (из pseo/)
--dry = всё до push (посмотреть, что поедет). Дизайн гарантирован шаблоном, гейт держит структуру.
"""

import json
import os
import re
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pseo
BUILT = f"{BASE}/builder"
OUT = f"{BASE}/out"
PAGES_REPO = os.path.abspath(f"{BASE}/../../multyspeak-pages")
VPS = "root@199.195.252.114"
VPS_DIR = "/root/pseo_builder"


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def step_pull(only=None):
    """Забрать built-данные + runner_stamps (метка «гео дозрел») с VPS одним tar."""
    r = sh(
        [
            "ssh",
            "-o",
            "ConnectTimeout=25",
            VPS,
            f"cd {VPS_DIR} && tar cf - out_facet out_questions runner_stamps.json 2>/dev/null | base64 -w0",
        ]
    )
    if r.returncode != 0 or not r.stdout.strip():
        print("pull: ssh/tar не отдал данных")
        return False
    import base64
    import io
    import tarfile

    buf = io.BytesIO(base64.b64decode(r.stdout.strip()))
    with tarfile.open(fileobj=buf) as tf:
        tf.extractall(BUILT, filter="data")
    n_f = len([f for f in os.listdir(f"{BUILT}/out_facet") if f.endswith(".json")])
    n_q = len([f for f in os.listdir(f"{BUILT}/out_questions") if f.endswith(".json")])
    print(f"pull: факт-гео {n_f}, вопрос-гео {n_q}")
    return True


def step_pages(only=None):
    args = [sys.executable, f"{BUILT}/pages.py"] + (only or ["--all"])
    r = sh(args, cwd=BASE)
    print(
        (r.stdout or r.stderr).strip().splitlines()[-1]
        if (r.stdout or r.stderr)
        else "pages: ?"
    )
    return r.returncode == 0


def step_check():
    r = sh([sys.executable, f"{BUILT}/readycheck.py"], cwd=BASE)
    print(r.stdout.strip())
    try:
        return json.load(open(f"{BASE}/ready.json", encoding="utf-8"))
    except Exception:
        return None


def geo_of(path):  # '/ru/br/q/...' → 'br'
    m = re.match(r"/ru/([a-z]{2})/", path)
    return m.group(1) if m else None


def bad_geos():
    """Гео с проблемами — из деталей readycheck (перечитаем сами, дёшево)."""
    bad = set()
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
            g = geo_of(rel)
            if not g:
                continue
            if "�" in html or "<h1>" not in html:
                bad.add(g)
                continue
            for href in set(re.findall(r'href="(/ru/[^"#]*)"', html)):
                if not os.path.exists(os.path.join(OUT, href.strip("/"), "index.html")):
                    bad.add(g)
                    break
    return bad


def step_push(dry, only=None):
    bad = bad_geos()
    # completeness-гейт: НОВАЯ модель гео едет только когда блок ДОЗРЕЛ (runner stamps —
    # гео исчерпан при текущих данных). Частично-тегнутое гео = тонкая замена богатого старого
    # → держим. Гео БЕЗ built-данных (старые страницы, не тронуты pages.py) — едут как были.
    try:
        stamps = set(json.load(open(f"{BUILT}/runner_stamps.json", encoding="utf-8")))
    except Exception:
        stamps = set()
    built_geos = {
        f[:-5] for f in os.listdir(f"{BUILT}/out_facet") if f.endswith(".json")
    }
    immature = built_geos - stamps  # начали тегать, но не дозрели
    geos = sorted(
        {
            d
            for d in os.listdir(f"{OUT}/ru")
            if os.path.isdir(f"{OUT}/ru/{d}") and re.fullmatch(r"[a-z]{2}", d)
        }
    )
    if only:
        geos = [g for g in geos if g in only]
    go = [g for g in geos if g not in bad and g not in immature]
    print(f"gate: едут {len(go)} гео {go}")
    print(
        f"      задержаны-битые {sorted(bad & set(geos))}; недозревшие {sorted(immature & set(geos))}"
    )
    if dry:
        print("DRY — без push.")
        return True
    for g in go:
        dst = f"{PAGES_REPO}/ru/{g}"
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(f"{OUT}/ru/{g}", dst)
    # главная/about/sitemap/robots — общие
    for extra in ("index.html",):
        srcp = f"{OUT}/ru/{extra}"
        if os.path.exists(srcp):
            shutil.copy2(srcp, f"{PAGES_REPO}/ru/{extra}")
    # МУЛЬТИЯЗЫК: все не-ru языковые деревья целиком (стамп-гейт только для RU; чисты по readycheck)
    langs = [
        d
        for d in os.listdir(OUT)
        if d != "ru" and re.fullmatch(r"[a-z]{2}", d) and os.path.isdir(f"{OUT}/{d}")
    ]
    for lang in langs:
        dst = f"{PAGES_REPO}/{lang}"
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(f"{OUT}/{lang}", dst)
    for f in ("sitemap.xml", "robots.txt"):
        if os.path.exists(f"{OUT}/{f}"):
            shutil.copy2(f"{OUT}/{f}", f"{PAGES_REPO}/{f}")
    sh(["git", "add", "-A"], cwd=PAGES_REPO)
    msg = f"pSEO ship: {len(go)} geo blocks ({', '.join(go)})\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    sh(["git", "commit", "-m", msg], cwd=PAGES_REPO)
    r = sh(["git", "push", "origin", "main"], cwd=PAGES_REPO)
    print("push:", "OK (CF подхватит)" if r.returncode == 0 else r.stderr[-200:])
    return r.returncode == 0


MIRROR_REPO = os.path.abspath(f"{BASE}/../../multyspeak-pages-ru")  # клон, ветка ru


def step_mirror(dry):
    """ЗЕРКАЛО для Яндекса: пере-рендер ВСЕГО под mirror_domain (canonical/sitemap на
    info.multyspeak.ru — иначе Яндекс видит канониклы на чужой .online и не индексирует)
    → пуш в ветку `ru` репо pages → Dokploy на РФ-серваке отдаёт статикой.
    Гонять ПОСЛЕ обычного ship (data/ уже собран). Основной out/ пере-рендеривается
    обратно под .online в конце (иначе следующий ship уедет с .ru-канониклами).
    """
    sys.path.insert(0, BASE)
    from config.site import SITE as _S

    mirror = _S["mirror_domain"]
    env = {
        **os.environ,
        "PSEO_DOMAIN": mirror,
        "PSEO_CTA_URL": _S["mirror_cta_url"],  # дверь Luky = .ru (РФ без VPN)
    }
    r = subprocess.run(
        [sys.executable, "render.py", "--all"],
        cwd=BASE,
        env=env,
        capture_output=True,
        text=True,
    )
    print("mirror render:", (r.stdout or r.stderr).strip().splitlines()[-1])
    if r.returncode != 0:
        return False
    if not os.path.isdir(MIRROR_REPO):  # первый раз: клон того же origin, ветка ru
        origin = sh(
            ["git", "remote", "get-url", "origin"], cwd=PAGES_REPO
        ).stdout.strip()
        sh(["git", "clone", origin, MIRROR_REPO])
        sh(["git", "checkout", "-B", "ru"], cwd=MIRROR_REPO)
    if dry:
        print("DRY — зеркало отрендерено в out/, без push.")
    else:
        for d in os.listdir(MIRROR_REPO):  # чистим всё кроме .git
            if d == ".git":
                continue
            p = os.path.join(MIRROR_REPO, d)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        for d in os.listdir(OUT):
            src = os.path.join(OUT, d)
            dst = os.path.join(MIRROR_REPO, d)
            shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy2(src, dst)
        for extra in ("favicon.svg",):  # статика вне out/
            p = os.path.join(PAGES_REPO, extra)
            if os.path.exists(p):
                shutil.copy2(p, MIRROR_REPO)
        ms = os.path.join(
            BASE, "mirror_static"
        )  # только-зеркальное (яндекс-верификация)
        if os.path.isdir(ms):
            for fn in os.listdir(ms):
                shutil.copy2(os.path.join(ms, fn), MIRROR_REPO)
        sh(["git", "add", "-A"], cwd=MIRROR_REPO)
        sh(
            [
                "git",
                "commit",
                "-m",
                "pSEO ru-mirror ship\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
            ],
            cwd=MIRROR_REPO,
        )
        r2 = sh(["git", "push", "origin", "ru", "--force"], cwd=MIRROR_REPO)
        print("mirror push:", "OK" if r2.returncode == 0 else r2.stderr[-200:])
    # вернуть out/ под основной домен — следующий ship не должен уехать с .ru-канониклами
    r3 = sh([sys.executable, "render.py", "--all"], cwd=BASE)
    print("re-render .online:", "OK" if r3.returncode == 0 else "FAIL")
    return True


def main():
    if "--mirror" in sys.argv:
        step_mirror("--dry" in sys.argv)
        return
    dry = "--dry" in sys.argv
    only = None
    if "--geo" in sys.argv:
        only = sys.argv[sys.argv.index("--geo") + 1].split(",")
    print("== 1. pull ==")
    step_pull(only) or sys.exit(1)
    print("== 2. pages ==")
    step_pages(only) or sys.exit(1)
    print("== 3. check ==")
    rep = step_check()
    if not rep:
        sys.exit(1)
    print("== 4-5. gate+push ==")
    step_push(dry, only)


if __name__ == "__main__":
    main()
