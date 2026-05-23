"""Тесты pacing/cooldown логики _embed_v2 для bulk-path.

Покрытие:
- min_interval_s sleep между текстами (success path);
- use_persistent_quota пишет в gemini_quota SQLite;
- первый 429 → in-memory cooldown, ключ скипается;
- второй 429 после cooldown → quota_ban_today (persistent);
- 3 разных ключа 429 в окне → global abuse pause.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from teledigest import extraction_db, gemini_brain


@pytest.fixture
def temp_quota_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_embed.db"

    def _connect():
        return sqlite3.connect(str(db_path))

    monkeypatch.setattr(extraction_db, "get_db_connection", _connect)
    extraction_db.init_extraction_tables()
    # Сбросить module-level state между тестами.
    gemini_brain._embed_cooldown_until.clear()
    gemini_brain._embed_was_cooldowned.clear()
    gemini_brain._embed_recent_429.clear()
    gemini_brain._embed_abuse_pause_until = 0.0
    gemini_brain._key_rpd_count.clear()
    gemini_brain._key_rr_idx = 0
    yield db_path


def _make_ok_client(vec: list[float] | None = None):
    """Mock genai.Client возвращает успешный embedding."""
    if vec is None:
        vec = [0.1] * 1536
    client = MagicMock()
    emb = MagicMock()
    emb.values = vec
    result = MagicMock()
    result.embeddings = [emb]
    client.models.embed_content.return_value = result
    return client


def _make_429_client():
    """Mock клиент кидает 429."""
    client = MagicMock()
    client.models.embed_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: free tier limit hit"
    )
    return client


def _patch_clients(monkeypatch, keys: list[str], clients_map: dict[str, MagicMock]):
    """Подменить genai.Client + _get_embedding_api_keys для теста.

    monkeypatch.setitem — автоматический cleanup, не засираем sys.modules
    глобально между тестами.
    """
    import sys

    fake_genai = MagicMock()
    fake_genai.Client.side_effect = lambda api_key: clients_map[api_key]
    fake_types = MagicMock()
    fake_types.EmbedContentConfig = MagicMock(return_value=MagicMock())
    fake_google = MagicMock()
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setattr(gemini_brain, "_get_embedding_api_keys", lambda: keys)


def test_min_interval_sleeps_between_texts(temp_quota_db, monkeypatch):
    keys = ["k1", "k2"]
    _patch_clients(monkeypatch, keys, {k: _make_ok_client() for k in keys})
    sleeps: list[float] = []
    monkeypatch.setattr(
        gemini_brain, "_time", MagicMock(time=time.time, sleep=sleeps.append)
    )

    out = gemini_brain._embed_v2(
        ["a", "b", "c"], "RETRIEVAL_DOCUMENT", min_interval_s=10.0
    )
    assert all(v is not None for v in out)
    # 10s sleep после каждого из 3 текстов.
    assert sleeps.count(10.0) == 3


def test_persistent_quota_increments_on_success(temp_quota_db, monkeypatch):
    keys = ["kALPHA", "kBETA"]
    _patch_clients(monkeypatch, keys, {k: _make_ok_client() for k in keys})
    monkeypatch.setattr(
        gemini_brain, "_time", MagicMock(time=time.time, sleep=lambda *_: None)
    )

    out = gemini_brain._embed_v2(
        ["text1", "text2"],
        "RETRIEVAL_DOCUMENT",
        min_interval_s=0.0,
        use_persistent_quota=True,
    )
    assert all(v is not None for v in out)
    # Каждый текст ушёл на один ключ — у двух ключей по 1 запросу.
    cnt_a, _ = (
        extraction_db.quota_state(
            gemini_brain._kh_for_test("kALPHA"), gemini_brain._EMBEDDING_MODEL_V2
        )
        if hasattr(gemini_brain, "_kh_for_test")
        else (None, None)
    )
    # Используем прямой импорт хэша.
    from teledigest.extraction_db import _key_hash

    a, _ = extraction_db.quota_state(
        _key_hash("kALPHA"), gemini_brain._EMBEDDING_MODEL_V2
    )
    b, _ = extraction_db.quota_state(
        _key_hash("kBETA"), gemini_brain._EMBEDDING_MODEL_V2
    )
    assert a + b == 2


def test_first_429_cooldowns_key(temp_quota_db, monkeypatch):
    keys = ["kBAD", "kGOOD"]
    clients = {"kBAD": _make_429_client(), "kGOOD": _make_ok_client()}
    _patch_clients(monkeypatch, keys, clients)
    monkeypatch.setattr(
        gemini_brain, "_time", MagicMock(time=time.time, sleep=lambda *_: None)
    )

    out = gemini_brain._embed_v2(
        ["x"], "RETRIEVAL_DOCUMENT", min_interval_s=0.0, use_persistent_quota=True
    )
    assert out[0] is not None  # успех через kGOOD
    # kBAD должен быть в cooldown.
    assert 0 in gemini_brain._embed_cooldown_until
    assert 0 in gemini_brain._embed_was_cooldowned
    # Persistent ban пока НЕ выставлен (первый 429).
    from teledigest.extraction_db import _key_hash, quota_state

    _, banned = quota_state(_key_hash("kBAD"), gemini_brain._EMBEDDING_MODEL_V2)
    assert banned is False


def test_second_429_after_cooldown_persists_ban(temp_quota_db, monkeypatch):
    keys = ["kBAD", "kGOOD"]
    clients = {"kBAD": _make_429_client(), "kGOOD": _make_ok_client()}
    _patch_clients(monkeypatch, keys, clients)
    # Имитируем что kBAD уже был в cooldown в прошлой сессии _embed_v2.
    gemini_brain._embed_was_cooldowned.add(0)
    monkeypatch.setattr(
        gemini_brain, "_time", MagicMock(time=time.time, sleep=lambda *_: None)
    )

    gemini_brain._embed_v2(
        ["y"], "RETRIEVAL_DOCUMENT", min_interval_s=0.0, use_persistent_quota=True
    )
    # Теперь persistent ban должен быть выставлен.
    from teledigest.extraction_db import _key_hash, quota_state

    _, banned = quota_state(_key_hash("kBAD"), gemini_brain._EMBEDDING_MODEL_V2)
    assert banned is True


def test_abuse_threshold_triggers_global_pause(temp_quota_db, monkeypatch):
    # 3 ключа все 429ят → должен сработать abuse flag.
    keys = ["k1", "k2", "k3"]
    clients = {k: _make_429_client() for k in keys}
    _patch_clients(monkeypatch, keys, clients)
    monkeypatch.setattr(
        gemini_brain, "_time", MagicMock(time=time.time, sleep=lambda *_: None)
    )

    gemini_brain._embed_v2(
        ["z"], "RETRIEVAL_DOCUMENT", min_interval_s=0.0, use_persistent_quota=True
    )
    # abuse_pause_until должен быть выставлен в будущее.
    assert gemini_brain._embed_abuse_pause_until > time.time()
