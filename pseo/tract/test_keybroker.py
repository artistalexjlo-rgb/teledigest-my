# -*- coding: utf-8 -*-
"""Сторож `any_alive()` (29.08): пульту после паузы нужно ЧЕСТНО знать, отпустило ли пул
ключей, не трогая сам механизм выдачи (`acquire()` двигает круг/такт — вызывать его из
пульта, который ничего не считает, значило бы забрать ход у настоящего рта).
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import keybroker  # noqa: E402

MODEL = "gemini-3.1-flash-lite"


def _setup(tmp_path, monkeypatch, keys=("k1", "k2")):
    monkeypatch.setattr(keybroker, "DB", str(tmp_path / "kb.db"))
    monkeypatch.setattr(keybroker, "get_keys", lambda: list(keys))
    keybroker.init()


def test_any_alive_true_on_a_fresh_pool(tmp_path, monkeypatch):
    """Никто ничего не трогал — ключи годны, ждать нечего."""
    _setup(tmp_path, monkeypatch)
    assert keybroker.any_alive(MODEL) is True


def test_any_alive_false_when_every_key_is_day_banned(tmp_path, monkeypatch):
    """Все ключи в дневном бане — ровно тот случай, что печатает `call()` при сдаче."""
    _setup(tmp_path, monkeypatch)
    day = keybroker._pt_day()
    c = keybroker._conn()
    for k in ("k1", "k2"):
        c.execute(
            "INSERT INTO usage(key_hash, model, pt_day, count, banned) VALUES(?,?,?,?,1)",
            (keybroker._kh(k), MODEL, day, 999),
        )
    c.commit()
    c.close()
    assert keybroker.any_alive(MODEL) is False


def test_any_alive_false_while_every_key_is_in_cooldown(tmp_path, monkeypatch):
    """Все ключи временно в 429-кулдауне — тоже «нет годных сейчас», как и день-бан."""
    _setup(tmp_path, monkeypatch)
    c = keybroker._conn()
    future = time.time() + 3600
    for k in ("k1", "k2"):
        c.execute(
            "INSERT INTO key_clock(key_hash, cooldown_until) VALUES(?,?)",
            (keybroker._kh(k), future),
        )
    c.commit()
    c.close()
    assert keybroker.any_alive(MODEL) is False


def test_any_alive_true_when_one_of_two_keys_is_free(tmp_path, monkeypatch):
    """Один живой сосед среди забаненных — уже достаточно, ждать не надо."""
    _setup(tmp_path, monkeypatch)
    day = keybroker._pt_day()
    c = keybroker._conn()
    c.execute(
        "INSERT INTO usage(key_hash, model, pt_day, count, banned) VALUES(?,?,?,?,1)",
        (keybroker._kh("k1"), MODEL, day, 999),
    )
    c.commit()
    c.close()
    assert keybroker.any_alive(MODEL) is True


def test_any_alive_true_once_cooldown_has_passed(tmp_path, monkeypatch):
    """Кулдаун в прошлом — ключ снова годен, `any_alive` не застревает навсегда."""
    _setup(tmp_path, monkeypatch)
    c = keybroker._conn()
    c.execute(
        "INSERT INTO key_clock(key_hash, cooldown_until) VALUES(?,?)",
        (keybroker._kh("k1"), time.time() - 10),
    )
    c.commit()
    c.close()
    assert keybroker.any_alive(MODEL) is True
