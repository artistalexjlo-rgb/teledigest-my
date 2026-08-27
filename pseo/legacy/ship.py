"""ship.py — ОДНА команда деплой-тракта (десктоп, by-call). Идеальный тракт при объёме:
руками только триггер, всё остальное — конвейер с гейтом качества.

  1. pull   — забрать built-данные ВСЕХ гео с VPS (out_facet/, out_questions/);
  2. pages  — pages.py --all → portal-data (оба контура: факт-темы, вопрос-контур, хабы);
  3. check  — readycheck: рендер через октагон-шаблон + детерм-валидация;
  4. gate   — гео с проблемами (битые/пустые/кодировка) НЕ едут; чистые — едут;
  5. push   — чистые гео-блоки → pages-репо → git push → CF авто-деплой.

  6. mirror — автоматически после push: пере-рендер под info.multyspeak.ru → ветка `ru`
     (Dokploy на РФ-серваке) → обратный рендер под .online. Оба сайта = одна команда.

Запуск: python builder/ship.py [--dry] [--geo br,vn] [--no-mirror]   (из pseo/)
--dry = всё до push (посмотреть, что поедет). --no-mirror = только .online.
--mirror = только зеркало (без pull/pages/push). Дизайн шаблоном, гейт держит структуру.
"""

import json
import os
import re
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pseo
BUILT = f"{BASE}/builder"
sys.path.insert(0, BASE)
from render import OUT as _OUT  # noqa: E402  один источник каталога вывода — render.py

# ⛔ Не заводить свой путь: ship, readycheck и render обязаны читать ОДНО определение,
# иначе при заданном PSEO_OUT они разъедутся молча — рендер напишет в одно место, гейт
# проверит другое, а push уедет с пустым третьим.
OUT = str(_OUT)
PAGES_REPO = os.path.abspath(f"{BASE}/../../multyspeak-pages")
VPS = "root@199.195.252.114"
VPS_DIR = "/root/pseo_builder"
# Контейнер комбайна: в нём живут АКТУАЛЬНЫЕ рты (/app/builder), а на хосте .py протухли
# 20.07. Имя ищем префиксом — Dokploy добавляет свой хеш задачи и меняет его при редеплое.
PULT_NAME = "bots-luky-rodzkl"


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _buildable_langs():
    """Языки, которые сборщик РЕАЛЬНО умеет собрать: нужны и тексты портала (pages.COPY),
    и словарь рендера (i18n/<lang>.json). Один источник правды для того, что тянуть.

    ⛔ НЕ тянем всё подряд по маске `out_facet_*`: комбайн переводит в 13 языков, а собрать
    можно 4 — на остальные нет ни COPY, ни i18n, и `pages.langs_for` их молча пропустит.
    Разница в объёме: 291 МБ против 53 МБ, и 238 из них — данные, которые нечем отрендерить.
    """
    import sys as _sys

    if BUILT not in _sys.path:
        _sys.path.insert(0, BUILT)
    import pages as _pages

    have_i18n = {f[:-5] for f in os.listdir(f"{BASE}/i18n") if f.endswith(".json")}
    return sorted(set(_pages.COPY) & have_i18n)


def step_pull(only=None):
    """Забрать built-данные + runner_stamps (метка «гео дозрел») с VPS одним tar.

    ⭐ ТЯНЕМ И ПЕРЕВОДЫ (2026-08-07). Раньше в таре были только `out_facet` (ru) и
    `out_questions` — каталоги `out_facet_<lang>` не приезжали ВООБЩЕ. Итог: на десктопе
    лежали копии en/es/pt от 10-12 июля (34 гео вместо 90, без полок), а свежие переводы
    всех 90 гео с полками стояли на VPS и до сайта не доходили. Трёхдневный прогон
    переводов физически не мог попасть в публикацию.
    """
    langs = [x for x in _buildable_langs() if x != "ru"]  # ru лежит в out_facet
    dirs = " ".join(
        ["out_facet"] + [f"out_facet_{x}" for x in langs] + ["out_questions"]
    )
    # -z обязателен: без сжатия это 291 МБ, и base64 в память лёг бы четырьмястами.
    r = sh(
        [
            "ssh",
            "-o",
            "ConnectTimeout=25",
            VPS,
            f"cd {VPS_DIR} && tar czf - {dirs} runner_stamps.json 2>/dev/null | base64 -w0",
        ]
    )
    if r.returncode != 0 or not r.stdout.strip():
        print("pull: ssh/tar не отдал данных")
        return False
    import base64
    import io
    import tarfile

    buf = io.BytesIO(base64.b64decode(r.stdout.strip()))
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        tf.extractall(BUILT, filter="data")
    n_f = len([f for f in os.listdir(f"{BUILT}/out_facet") if f.endswith(".json")])
    n_q = len([f for f in os.listdir(f"{BUILT}/out_questions") if f.endswith(".json")])
    per_lang = ", ".join(
        "%s %d"
        % (
            x,
            len(
                [f for f in os.listdir(f"{BUILT}/out_facet_{x}") if f.endswith(".json")]
            ),
        )
        for x in langs
        if os.path.isdir(f"{BUILT}/out_facet_{x}")
    )
    print(f"pull: факт-гео {n_f}, вопрос-гео {n_q}, переводы: {per_lang}")
    _pull_mature()
    return True


def _pull_mature():
    """Забрать ВЫЧИСЛЕННУЮ зрелость гео из комбайна (facet.py --mature).

    ⭐ ЗАМЕНА МЁРТВЫМ ШТАМПАМ (2026-08-07). Гейт ниже пускал гео только по
    `runner_stamps.json`, а писал его `pseo-runner` — снесённый 20.07. Файл замёрз на
    36 гео из 90, и всё собранное позже не поехало бы никогда. Считать зрелость на
    десктопе нельзя: база мух и tags/ живут на VPS. Поэтому спрашиваем у того, кто по
    этому правилу берёт работу, — у facet в контейнере комбайна.
    ⛔ Не дублировать правило здесь: ровно на таких копиях мы горели трижды за сутки.
    Не ответил — молчим и падаем на старые штампы (гейт станет строже, не слабее).
    """
    r = sh(
        [
            "ssh",
            "-o",
            "ConnectTimeout=25",
            VPS,
            f"id=$(docker ps -q -f name={PULT_NAME} | head -1); "
            f"docker exec -w {VPS_DIR} $id python /app/builder/facet.py --mature",
        ]
    )
    txt = (r.stdout or "").strip().splitlines()
    try:
        data = json.loads(txt[-1]) if txt else None
        assert isinstance(data, dict) and data
    except Exception:
        print("pull: зрелость не получена — гейт пойдёт по старым штампам")
        return
    json.dump(
        data,
        open(f"{BUILT}/mature_geos.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    n = sum(1 for v in data.values() if v)
    print(f"pull: зрелость посчитана комбайном — зрелых {n} из {len(data)}")


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


def _repo_path_for(loc):
    """URL из sitemap → путь файла в pages-репо. '/ru/ar/' → 'ru/ar/index.html'."""
    rel = re.sub(r"^https?://[^/]+", "", loc).strip("/")
    return os.path.join(PAGES_REPO, *(rel.split("/") if rel else []), "index.html")


def _write_filtered_sitemap():
    """Sitemap = только то, что РЕАЛЬНО лежит в pages-репо, а не всё, что собралось.

    ⭐ ЗАЧЕМ (2026-08-07). Карта копировалась из `out/` целиком, а гейт отсекал часть гео
    ПОЗЖЕ. Итог: 188 русских адресов задержанных гео стояли в sitemap, а страниц по ним на
    сайте не было. Мы своей же картой звали Google на несуществующее — и теперь, когда
    появился настоящий 404, он бы честно их получил.

    ⛔ Фильтруем ПО ФАЙЛАМ, а не по списку уехавших гео: гейт русский, языковые деревья
    едут целиком, и правило «что уехало» у них разное. Проверка наличия файла верна для
    любого языка — плюс десять языков тут ничего не меняют.
    Даты (`lastmod`) сохраняются как есть: их ставит pages.py постранично.
    """
    src = f"{OUT}/sitemap.xml"
    if not os.path.exists(src):
        print("sitemap: нет out/sitemap.xml — пропускаю")
        return
    xml = open(src, encoding="utf-8").read()
    blocks = re.findall(r"  <url>.*?</url>\n", xml, re.S)
    kept, dropped = [], []
    for b in blocks:
        m = re.search(r"<loc>([^<]+)</loc>", b)
        if m and os.path.exists(_repo_path_for(m.group(1))):
            kept.append(b)
        elif m:
            dropped.append(m.group(1))
    out = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(kept)
        + "</urlset>\n"
    )
    open(f"{PAGES_REPO}/sitemap.xml", "w", encoding="utf-8").write(out)
    print(f"sitemap: {len(kept)} адресов; выкинуто без файла {len(dropped)}")
    if dropped:
        print("         примеры:", ", ".join(dropped[:3]))


def step_push(dry, only=None):
    bad = bad_geos()
    # completeness-гейт: НОВАЯ модель гео едет только когда блок ДОЗРЕЛ (runner stamps —
    # гео исчерпан при текущих данных). Частично-тегнутое гео = тонкая замена богатого старого
    # → держим. Гео БЕЗ built-данных (старые страницы, не тронуты pages.py) — едут как были.
    built_geos = {
        f[:-5] for f in os.listdir(f"{BUILT}/out_facet") if f.endswith(".json")
    }
    # ЗРЕЛОСТЬ: считает комбайн (facet.py --mature), привозит _pull_mature. Старые штампы —
    # только запасной путь: их писал pseo-runner, снесённый 20.07, и файл замёрз на 36 гео
    # из 90. Пока он был единственным источником, всё собранное после 19.07 не ехало вовсе.
    try:
        mature = json.load(open(f"{BUILT}/mature_geos.json", encoding="utf-8"))
        immature = {g for g in built_geos if not mature.get(g)}
        src = "вычислено комбайном"
    except Exception:
        try:
            stamps = set(
                json.load(open(f"{BUILT}/runner_stamps.json", encoding="utf-8"))
            )
        except Exception:
            stamps = set()
        immature = built_geos - stamps
        src = "СТАРЫЕ ШТАМПЫ (замёрзли 19.07) — зрелость не приехала"
    print(f"gate: зрелость — {src}")
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
    if os.path.exists(f"{OUT}/robots.txt"):
        shutil.copy2(f"{OUT}/robots.txt", f"{PAGES_REPO}/robots.txt")
    _write_filtered_sitemap()
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
    step_push(dry, only) or sys.exit(1)
    # ЕДИНЫЙ ship: оба сайта одной командой — .online и .ru-зеркало не разъезжаются
    if not dry and "--no-mirror" not in sys.argv:
        print("== 6. mirror ==")
        step_mirror(False)


if __name__ == "__main__":
    main()
