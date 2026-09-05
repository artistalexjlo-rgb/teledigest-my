# -*- coding: utf-8 -*-
"""publish.py — звено 8, шаг 3: атомарная подмена дерева на живом сайте.

PLAN.md §3.2, шаг 3. Живой `site/online` — то, что nginx-контейнер (`bots-pseosite`)
отдаёт прямо сейчас реальным посетителям.

⛔ 02.09, проверено на одноразовых контейнерах (не на живом сайте) ДВАЖДЫ, вторая
проверка исправила первую:
  1. Голый `rename` папки, на которую bind-mount смотрит НАПРЯМУЮ, — контейнер БЕЗ
     рестарта продолжает отдавать старое содержимое (bind-mount держит inode, не
     путь). Нужен `docker restart` — но у пульта нет доступа к docker.sock, самому
     его вызвать нечем.
  2. Если вместо этого bind-mount смотрит на СТАБИЛЬНУЮ родительскую папку (`site/`,
     которая никогда не переименовывается), а живая версия выбирается симлинком
     `site/current` ВНУТРИ неё — подхватывается МГНОВЕННО, без рестарта: nginx сам
     открывает файлы по симлинку на каждый запрос, Docker тут вообще не участвует.
     `point_current()` ниже — эта схема.

Требует ОДНОРАЗОВОЙ ручной правки: bind-mount `bots-pseosite` в Dokploy — Host Path
с `.../site/online` на `.../site` (родитель), плюс `root` в nginx-конфиге —
`.../html` на `.../html/current`. До этой правки `point_current()` не имеет смысла
звать на проде — переезд не сделан молча, ждёт факта от юзера.

`swap_in()` — более старая, независимая часть: атомарная замена ОДНОЙ папки другой
(две `os.rename`, старое не теряется). Годится сама по себе (например, переносит
свежесобранное `staging` в его версионированное имя `online_v<ts>` первой публикацией
— тогда `live` попросту не существовал, `prev=None`), но НЕ решает вопрос
«увидит ли живой nginx новое» — это теперь работа `point_current()`.
"""

import os
import shutil
import time


def swap_in(staging: str, live: str, keep_prev: bool = True) -> str | None:
    """Подменить `live` на `staging`. Возвращает путь к отведённой старой версии
    (или `None`, если `live` не существовал — самая первая публикация) — чистить её
    или нет, решает вызывающий, не эта функция.

    ⛔ Готовность `staging` (скопировалось ли ПОЛНОСТЬЮ) — забота вызывающего. Здесь
    только механика подмены, она не гадает, что считать «достаточно скопировано».
    """
    if not os.path.isdir(staging):
        raise FileNotFoundError(f"нет свежего дерева для подмены: {staging}")
    prev = None
    if os.path.exists(live):
        prev = f"{live}_prev_{int(time.time())}"
        os.rename(live, prev)
    try:
        os.rename(staging, live)
    except Exception:
        # подмена не удалась — вернуть старое НА МЕСТО: живой путь не должен
        # остаться вовсе без каталога из-за нашей же попытки его обновить
        if prev:
            os.rename(prev, live)
        raise
    if prev and not keep_prev:
        shutil.rmtree(prev)
        prev = None
    return prev


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
        if d.startswith(prefix) and os.path.isdir(os.path.join(site_dir, d))
    )
    doomed = [v for v in versions[:-keep] if v != current_target] if keep > 0 else []
    for v in doomed:
        shutil.rmtree(os.path.join(site_dir, v))
    return doomed
