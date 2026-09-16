# -*- coding: utf-8 -*-
"""Сторож звена 8 (PLAN.md §3.2): выкладка версией + симлинк, старое не исчезает.

⛔ 12.09: тесты `swap_in` сняты вместе с ним — один способ подмены, симлинк.
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


@needs_symlinks
def test_publish_tree_copies_a_version_and_points_current_at_it(tmp_path):
    out = tmp_path / "out"
    _tree(str(out / "ru"), "новое")
    site = str(tmp_path / "online")

    rep = publish.publish_tree(str(out), site, keep=3, now=1000)

    assert rep["version"] == "online_v1000"
    assert rep["prev"] is None, "первая публикация — предыдущей версии нет"
    assert os.readlink(os.path.join(site, "current")) == "online_v1000"
    live = os.path.join(site, "current", "ru", "marker.txt")
    assert open(live, encoding="utf-8").read() == "новое"
    assert os.listdir(str(out)) == ["ru"], "песочница не тронута — копия, не переезд"
    assert not any(d.endswith(".staging") for d in os.listdir(site)), "обрывков нет"


@needs_symlinks
def test_second_publish_keeps_previous_version_for_rollback(tmp_path):
    out = tmp_path / "out"
    site = str(tmp_path / "online")
    _tree(str(out / "ru"), "v1")
    publish.publish_tree(str(out), site, keep=3, now=1000)
    _tree(str(out / "ru"), "v2")

    rep = publish.publish_tree(str(out), site, keep=3, now=2000)

    assert rep["prev"] == "online_v1000"
    assert os.readlink(os.path.join(site, "current")) == "online_v2000"
    old = os.path.join(site, "online_v1000", "ru", "marker.txt")
    assert open(old, encoding="utf-8").read() == "v1", "путь отката лежит рядом"


@needs_symlinks
def test_publish_prunes_beyond_keep_but_never_current(tmp_path):
    out = tmp_path / "out"
    site = str(tmp_path / "online")
    _tree(str(out / "ru"), "x")
    for ts in (1000, 2000, 3000):
        publish.publish_tree(str(out), site, keep=2, now=ts)

    names = sorted(d for d in os.listdir(site) if d.startswith("online_v"))
    assert names == ["online_v2000", "online_v3000"], names
    assert os.readlink(os.path.join(site, "current")) == "online_v3000"


@needs_symlinks
def test_leftover_staging_from_a_failed_run_is_cleared(tmp_path):
    out = tmp_path / "out"
    site = tmp_path / "online"
    _tree(str(out / "ru"), "ok")
    _tree(str(site / "online_v1000.staging"), "обрывок")

    publish.publish_tree(str(out), str(site), keep=3, now=1000)

    assert not (site / "online_v1000.staging").exists()
    live = site / "current" / "ru" / "marker.txt"
    assert open(str(live), encoding="utf-8").read() == "ok"


def test_publish_refuses_an_empty_snapshot(tmp_path):
    """Пустая песочница — не «опубликовать пустой сайт», а ошибка до любого касания."""
    (tmp_path / "out").mkdir()
    with pytest.raises(FileNotFoundError):
        publish.publish_tree(str(tmp_path / "out"), str(tmp_path / "online"))
    assert not (tmp_path / "online" / "current").exists()


def test_swap_in_is_gone():
    """Один способ подмены — симлинк. Rename-подмена не должна вернуться (12.09)."""
    assert not hasattr(publish, "swap_in")
