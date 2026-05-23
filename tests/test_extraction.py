"""Тесты rotator-логики extraction.iter_model_key_pairs.

Покрытие:
- порядок выдачи (model A × все ключи → next model);
- sleep между моделями вызывается ровно столько раз, сколько использованных моделей;
- забаненные/закапанные пары пропускаются;
- StopIteration при пустом обороте.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from teledigest import extraction, extraction_db


@pytest.fixture
def temp_quota_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_quota.db"

    def _connect():
        return sqlite3.connect(str(db_path))

    monkeypatch.setattr(extraction_db, "get_db_connection", _connect)
    extraction_db.init_extraction_tables()
    yield db_path


def _collect(gen, max_items: int) -> list[tuple[str, str]]:
    """Pull up to max_items from generator, stopping early if StopIteration."""
    out: list[tuple[str, str]] = []
    for _ in range(max_items):
        try:
            out.append(next(gen))
        except StopIteration:
            break
    return out


def test_rotator_order_model_then_keys(temp_quota_db):
    sleeps: list[float] = []
    models = [("m1", 10), ("m2", 10)]
    gen = extraction.iter_model_key_pairs(
        ["k1", "k2", "k3"], sleep_fn=sleeps.append, models=models
    )
    # Тянем 7 — чтобы спровоцировать sleep после m2 (генератор выполняет
    # код после yield только на следующем next()).
    pairs = _collect(gen, 7)
    assert pairs[:6] == [
        ("m1", "k1"),
        ("m1", "k2"),
        ("m1", "k3"),
        ("m2", "k1"),
        ("m2", "k2"),
        ("m2", "k3"),
    ]
    # 7-й next() начинает новый круг и выдаёт первую пару следующего оборота.
    assert pairs[6] == ("m1", "k1")
    # Sleep вызвался дважды (после m1 и после m2).
    assert sleeps == [extraction._INTER_MODEL_SLEEP_S] * 2


def test_rotator_skips_banned_pairs(temp_quota_db):
    sleeps: list[float] = []
    # Забанить (k2, m1).
    extraction_db.quota_ban_today(extraction._key_hash("k2"), "m1")
    models = [("m1", 10), ("m2", 10)]
    gen = extraction.iter_model_key_pairs(
        ["k1", "k2", "k3"], sleep_fn=sleeps.append, models=models
    )
    pairs = _collect(gen, 5)
    # На m1 — только k1 и k3, k2 забанен.
    assert pairs[:2] == [("m1", "k1"), ("m1", "k3")]
    # Дальше — все три ключа на m2.
    assert pairs[2:5] == [("m2", "k1"), ("m2", "k2"), ("m2", "k3")]


def test_rotator_skips_capped_pairs(temp_quota_db):
    sleeps: list[float] = []
    # k1 на m1 уже сделал 20 запросов при cap=20 — должен скипаться.
    kh = extraction._key_hash("k1")
    for _ in range(20):
        extraction_db.quota_increment(kh, "m1")
    models = [("m1", 20), ("m2", 20)]
    gen = extraction.iter_model_key_pairs(
        ["k1", "k2"], sleep_fn=sleeps.append, models=models
    )
    pairs = _collect(gen, 3)
    # На m1 — только k2 (k1 на капе).
    assert pairs[0] == ("m1", "k2")
    # Затем m2 с обоими ключами.
    assert pairs[1:3] == [("m2", "k1"), ("m2", "k2")]


def test_rotator_stops_when_all_exhausted(temp_quota_db):
    sleeps: list[float] = []
    # Забанить все пары.
    for k in ("k1", "k2"):
        for m in ("m1", "m2"):
            extraction_db.quota_ban_today(extraction._key_hash(k), m)
    models = [("m1", 10), ("m2", 10)]
    gen = extraction.iter_model_key_pairs(
        ["k1", "k2"], sleep_fn=sleeps.append, models=models
    )
    pairs = _collect(gen, 10)
    assert pairs == []
    # И sleep вообще не звался — пустой оборот.
    assert sleeps == []


def test_rotator_empty_keys_returns_nothing(temp_quota_db):
    sleeps: list[float] = []
    gen = extraction.iter_model_key_pairs(
        [], sleep_fn=sleeps.append, models=[("m1", 10)]
    )
    assert list(gen) == []
    assert sleeps == []


def test_rotator_continues_to_next_round(temp_quota_db):
    """Второй оборот — те же пары снова доступны (RPD не достигнут)."""
    sleeps: list[float] = []
    models = [("m1", 10)]
    gen = extraction.iter_model_key_pairs(
        ["k1", "k2"], sleep_fn=sleeps.append, models=models
    )
    # Тянем 7 чтобы спровоцировать 3-й sleep после третьего прохода m1.
    pairs = _collect(gen, 7)
    assert pairs[:6] == [("m1", "k1"), ("m1", "k2")] * 3
    assert sleeps == [extraction._INTER_MODEL_SLEEP_S] * 3
