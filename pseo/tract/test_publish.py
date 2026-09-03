# -*- coding: utf-8 -*-
"""Сторож звена 8, шаг 3 (PLAN.md §3.2): подмена живого дерева атомарна и обратима.

Не про rsync (это забота вызывающего) — только про саму подмену: старое дерево не
исчезает без следа, ошибка на подмене не оставляет живой путь пустым.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import publish  # noqa: E402

# `os.symlink` на Windows требует прав администратора/Developer Mode — реального
# ограничения кода тут нет (звено 8 живёт на Linux-VPS и в Linux-CI), но локальный
# Windows-прогон падал бы не на логике, а на самой ОС. Симлинк-логику проверил на
# настоящем Linux (одноразовый контейнер на VPS, 02.09) ДО того, как писать эти
# тесты — здесь она под CI, где Linux и есть.
needs_symlinks = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.symlink на Windows требует admin/Developer Mode — не ограничение кода",
)


def _tree(path, marker):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "marker.txt"), "w", encoding="utf-8") as fh:
        fh.write(marker)


def test_swap_puts_the_staging_content_live(tmp_path):
    live = str(tmp_path / "online")
    staging = str(tmp_path / "online_new")
    _tree(live, "старое")
    _tree(staging, "новое")

    publish.swap_in(staging, live)

    assert os.path.isdir(live)
    assert open(os.path.join(live, "marker.txt"), encoding="utf-8").read() == "новое"
    assert not os.path.isdir(staging), "staging должен был переехать, не скопироваться"


def test_swap_keeps_the_old_tree_for_rollback(tmp_path):
    live = str(tmp_path / "online")
    staging = str(tmp_path / "online_new")
    _tree(live, "старое")
    _tree(staging, "новое")

    prev = publish.swap_in(staging, live)

    assert prev is not None
    assert os.path.isdir(prev)
    assert open(os.path.join(prev, "marker.txt"), encoding="utf-8").read() == "старое"


def test_swap_with_keep_prev_false_removes_the_old_tree(tmp_path):
    live = str(tmp_path / "online")
    staging = str(tmp_path / "online_new")
    _tree(live, "старое")
    _tree(staging, "новое")

    prev = publish.swap_in(staging, live, keep_prev=False)

    assert prev is None


def test_first_ever_publish_with_no_prior_live_tree(tmp_path):
    """Живого дерева ещё нет вовсе — не первая публикация вообще, а первая
    В ЭТОМ каталоге (например .ru-зеркало до своего первого прогона)."""
    live = str(tmp_path / "online")
    staging = str(tmp_path / "online_new")
    _tree(staging, "новое")

    prev = publish.swap_in(staging, live)

    assert prev is None
    assert open(os.path.join(live, "marker.txt"), encoding="utf-8").read() == "новое"


def test_swap_refuses_a_missing_staging_tree(tmp_path):
    live = str(tmp_path / "online")
    _tree(live, "старое")

    try:
        publish.swap_in(str(tmp_path / "no_such_dir"), live)
        assert False, "должно было упасть — staging не существует"
    except FileNotFoundError:
        pass

    # живое дерево не тронуто отказавшейся подменой
    assert open(os.path.join(live, "marker.txt"), encoding="utf-8").read() == "старое"


def test_a_failed_second_rename_restores_the_live_tree(tmp_path, monkeypatch):
    """Если вторая подмена (staging -> live) обломилась — живой путь не должен
    остаться пустым: старое возвращается на место, а не теряется."""
    live = str(tmp_path / "online")
    staging = str(tmp_path / "online_new")
    _tree(live, "старое")
    _tree(staging, "новое")

    real_rename = os.rename
    calls = []

    def _flaky_rename(src, dst):
        calls.append((src, dst))
        if src == staging:
            raise OSError("диск кончился на середине")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _flaky_rename)

    try:
        publish.swap_in(staging, live)
        assert False, "должно было упасть"
    except OSError:
        pass

    assert os.path.isdir(live), "живой путь не должен остаться без каталога"
    assert open(os.path.join(live, "marker.txt"), encoding="utf-8").read() == "старое"


@needs_symlinks
def test_point_current_switches_the_symlink_atomically(tmp_path):
    site = tmp_path / "site"
    _tree(site / "online_v1", "версия 1")
    _tree(site / "online_v2", "версия 2")
    link = str(site / "current")
    os.symlink("online_v1", link)

    publish.point_current(link, "online_v2")

    assert os.path.islink(link)
    assert os.readlink(link) == "online_v2"
    resolved = os.path.join(str(site), os.readlink(link), "marker.txt")
    assert open(resolved, encoding="utf-8").read() == "версия 2"


@needs_symlinks
def test_point_current_works_when_symlink_does_not_exist_yet(tmp_path):
    """Самая первая публикация — `current` ещё не заведён."""
    site = tmp_path / "site"
    _tree(site / "online_v1", "версия 1")
    link = str(site / "current")

    publish.point_current(link, "online_v1")

    assert os.readlink(link) == "online_v1"


@needs_symlinks
def test_point_current_refuses_a_version_that_does_not_exist(tmp_path):
    site = tmp_path / "site"
    _tree(site / "online_v1", "версия 1")
    link = str(site / "current")
    os.symlink("online_v1", link)

    try:
        publish.point_current(link, "online_v999")
        assert False, "должно было упасть — такой версии нет"
    except FileNotFoundError:
        pass

    # симлинк не тронут отказавшейся подменой
    assert os.readlink(link) == "online_v1"


def test_prune_versions_keeps_the_last_n_and_never_touches_current(tmp_path):
    site = tmp_path / "site"
    for n in (1, 2, 3, 4, 5):
        _tree(site / f"online_v{n}", f"версия {n}")
    # v1 старше всех, но именно на неё сейчас смотрит current — не трогаем
    doomed = publish.prune_versions(
        str(site), "online_v", keep=2, current_target="online_v1"
    )

    assert set(doomed) == {"online_v2", "online_v3"}, doomed
    remaining = {d for d in os.listdir(site) if d.startswith("online_v")}
    assert remaining == {"online_v1", "online_v4", "online_v5"}, remaining


def test_prune_versions_ignores_files_that_are_not_versions(tmp_path):
    """Соседи без нужного префикса (nginxconf — своя папка конфига, `current` —
    вообще не директория, а указатель) не участвуют в подсчёте `keep` и не
    удаляются — `prune_versions` фильтрует строго по префиксу имени."""
    site = tmp_path / "site"
    _tree(site / "online_v1", "версия 1")
    _tree(site / "online_v2", "версия 2")
    os.makedirs(site / "nginxconf", exist_ok=True)

    doomed = publish.prune_versions(
        str(site), "online_v", keep=1, current_target="online_v2"
    )

    assert doomed == ["online_v1"]
    assert os.path.isdir(site / "nginxconf"), "чужая папка не должна была пострадать"
