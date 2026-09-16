# -*- coding: utf-8 -*-
"""publish.py — звено 8: выкладка снимка песочницы на живой сайт.

PLAN.md §3.2. Живой сайт отдаёт nginx-контейнер `bots-pseosite` из bind-mount
`/root/pseo_builder/site/online` → `/usr/share/nginx/html`.

⛔ 02.09, проверено на одноразовых контейнерах ДВАЖДЫ:
  1. Голый `rename` папки, на которую bind-mount смотрит НАПРЯМУЮ, контейнер БЕЗ
     рестарта не видит (bind-mount держит inode, не путь). `docker restart` пульту
     недоступен — docker.sock у него нет.
  2. Если bind-mount смотрит на СТАБИЛЬНУЮ папку, а живая версия выбирается симлинком
     ВНУТРИ неё — подхватывается МГНОВЕННО: nginx открывает файлы по симлинку на
     каждый запрос, Docker не участвует.

⭐ 12.09: стабильная папка — САМ `site/online` (он и так примонтирован), маунт в Dokploy
НЕ трогается. Внутри него: версии `online_v<ts>/` и симлинк `current` → одна из них;
nginx `root /usr/share/nginx/html/current;`. Окно на VPS сделано 15.09 (`docs/vps_window.sh`).

Один способ подмены — симлинк. `swap_in` (rename папок) снесён: два механизма на одну
задачу, и следующая кнопка написалась бы на rename и наступила бы на п.1.

Порядок `publish_tree()` — ничего живого не трогается, пока новое не лежит целиком:
  копия `out` → `<site>/online_v<ts>.staging` → rename в `online_v<ts>` →
  `current` → на него → чистка версий старше `keep` (текущую не трогаем никогда).
Откат = переставить `current` на предыдущую версию, она лежит рядом.
"""

import argparse
import json
import os
import shutil
import sys
import time

PREFIX = "online_v"
CURRENT = "current"


def point_current(link_path: str, target_name: str) -> None:
    """Атомарно переставить симлинк `link_path` на `target_name` (имя-сосед В ТОЙ ЖЕ
    папке, не абсолютный путь — переносимо, не тащит путь хоста в конфиг).

    ⛔ `ln -sfn` (и любой `unlink()` + `symlink()` по отдельности) НЕ атомарно: между
    двумя сисколлами путь `link_path` какое-то время не существует вовсе. Здесь —
    новый симлинк под временным именем, потом `os.replace` (ОДИН атомарный сисколл,
    подменяет цель даже если `link_path` уже существует как файл/симлинк).
    """
    if not os.path.isdir(os.path.join(os.path.dirname(link_path), target_name)):
        raise FileNotFoundError(f"нет такой версии рядом с симлинком: {target_name}")
    tmp = f"{link_path}.tmp{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target_name, tmp)
    os.replace(tmp, link_path)


def prune_versions(site_dir: str, prefix: str, keep: int, current_target: str) -> list:
    """Убрать версии старше `keep` (по имени — имена растут по времени публикации,
    `online_v<unix_ts>`, сортировка строкой = сортировка по времени). Версию, на
    которую сейчас смотрит `current` (`current_target`), не трогаем НИКОГДА, даже
    если она вне последних `keep` — живое важнее лимита на диске.

    Возвращает список убранных имён (для отчёта в чат, не для повторной чистки).
    """
    versions = sorted(
        d
        for d in os.listdir(site_dir)
        if d.startswith(prefix)
        and not d.endswith(".staging")
        and os.path.isdir(os.path.join(site_dir, d))
    )
    doomed = [v for v in versions[:-keep] if v != current_target] if keep > 0 else []
    for v in doomed:
        shutil.rmtree(os.path.join(site_dir, v))
    return doomed


def current_target(site_dir: str) -> str | None:
    """Имя версии, на которую смотрит `current`, или None (симлинка ещё нет)."""
    link = os.path.join(site_dir, CURRENT)
    if not os.path.islink(link):
        return None
    return os.path.basename(os.readlink(link))


def publish_tree(out_dir: str, site_dir: str, keep: int = 3, now=None) -> dict:
    """Выложить дерево `out_dir` новой версией в `site_dir` и переключить `current`.

    ⛔ Готовность содержимого здесь не решается — это гейт звена 7 (`readiness.py`),
    кнопка публикации в пульте не включается, пока он не ✅. Здесь только перенос.
    """
    if not os.path.isdir(out_dir) or not os.listdir(out_dir):
        raise FileNotFoundError(f"нечего публиковать: пусто или нет {out_dir}")
    os.makedirs(site_dir, exist_ok=True)
    ts = int(now if now is not None else time.time())
    version = f"{PREFIX}{ts}"
    final = os.path.join(site_dir, version)
    staging = f"{final}.staging"
    if os.path.exists(final):
        raise FileExistsError(f"версия уже есть: {version}")
    if os.path.exists(staging):  # обрывок прошлой неудачной попытки
        shutil.rmtree(staging)
    t0 = time.time()
    # ⛔ Сначала ЦЕЛИКОМ в `.staging`, потом rename: `current` никогда не увидит
    # полусобранную версию, а обрывок копии виден по имени и убирается на входе.
    shutil.copytree(out_dir, staging)
    os.rename(staging, final)
    prev = current_target(site_dir)
    point_current(os.path.join(site_dir, CURRENT), version)
    pruned = prune_versions(site_dir, PREFIX, keep, version)
    files = sum(len(fs) for _, _, fs in os.walk(final))
    return {
        "version": version,
        "prev": prev,
        "files": files,
        "pruned": pruned,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="выложить снимок песочницы на живой сайт")
    p.add_argument("--out", required=True, help="дерево рендера (PSEO_OUT)")
    p.add_argument("--site", required=True, help="папка раздачи: версии + current")
    p.add_argument("--keep", type=int, default=3, help="сколько версий хранить")
    p.add_argument("--stamp", help="куда записать отчёт публикации (JSON)")
    a = p.parse_args(argv)
    rep = publish_tree(a.out, a.site, keep=a.keep)
    rep["ts"] = time.time()
    rep["out"] = os.path.abspath(a.out)
    if a.stamp:
        with open(a.stamp, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
