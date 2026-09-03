# -*- coding: utf-8 -*-
"""Сторож звена 8, шаг 3 (PLAN.md §3.2): подмена живого дерева атомарна и обратима.

Не про rsync (это забота вызывающего) — только про саму подмену: старое дерево не
исчезает без следа, ошибка на подмене не оставляет живой путь пустым.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import publish  # noqa: E402


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
