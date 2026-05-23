"""Unit tests for qdrant_db wrapper. Mocks qdrant_client to avoid real network."""

from __future__ import annotations

import datetime as dt
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_qdrant_client():
    """Reset module-level _CLIENT singleton between tests."""
    from teledigest import qdrant_db

    qdrant_db.reset_client()
    yield
    qdrant_db.reset_client()


def _stub_qdrant_modules():
    """Stub out `qdrant_client` so tests don't require the real package
    installed (which is a heavy network-only dep)."""
    qclient_pkg = types.ModuleType("qdrant_client")
    qhttp_pkg = types.ModuleType("qdrant_client.http")
    qhttp_models = types.ModuleType("qdrant_client.http.models")

    class _Distance:
        COSINE = "cosine"

    class _VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class _PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    qhttp_models.Distance = _Distance
    qhttp_models.VectorParams = _VectorParams
    qhttp_models.PointStruct = _PointStruct
    qhttp_pkg.models = qhttp_models
    qclient_pkg.http = qhttp_pkg

    # MagicMock() сам по себе callable + tracking — даёт нам и factory,
    # и assert_called_once() на нём же.
    qclient_pkg.QdrantClient = MagicMock(name="QdrantClient")

    sys.modules["qdrant_client"] = qclient_pkg
    sys.modules["qdrant_client.http"] = qhttp_pkg
    sys.modules["qdrant_client.http.models"] = qhttp_models
    return qclient_pkg


def _stub_config(host="localhost", port=6333, api_key=""):
    """Stub get_config() to return a tiny object with .qdrant fields."""
    qd = MagicMock()
    qd.host = host
    qd.port = port
    qd.api_key = api_key
    qd.vector_dim = 1536
    qd.wisdom_collection = "wisdom_base"
    qd.wiki_collection = "wikivoyage_base"
    cfg = MagicMock()
    cfg.qdrant = qd
    return cfg


def test_is_configured_true_when_host_set():
    from teledigest import qdrant_db

    with patch.object(
        qdrant_db, "get_config", return_value=_stub_config(host="localhost")
    ):
        assert qdrant_db.is_configured() is True


def test_is_configured_false_when_host_empty():
    from teledigest import qdrant_db

    with patch.object(qdrant_db, "get_config", return_value=_stub_config(host="")):
        assert qdrant_db.is_configured() is False


def test_get_client_raises_when_not_configured():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    with patch.object(qdrant_db, "get_config", return_value=_stub_config(host="")):
        with pytest.raises(RuntimeError, match="host is empty"):
            qdrant_db.get_client()


def test_get_client_singleton():
    qclient = _stub_qdrant_modules()
    from teledigest import qdrant_db

    with patch.object(qdrant_db, "get_config", return_value=_stub_config()):
        c1 = qdrant_db.get_client()
        c2 = qdrant_db.get_client()
        assert c1 is c2
        qclient.QdrantClient.assert_called_once()


def test_ensure_collection_skips_existing():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    existing = MagicMock()
    existing.collections = [MagicMock(name="wisdom_base")]
    existing.collections[0].name = "wisdom_base"
    fake_client.get_collections.return_value = existing

    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        qdrant_db.ensure_collection("wisdom_base")
    fake_client.create_collection.assert_not_called()


def test_ensure_collection_creates_when_missing():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    existing = MagicMock()
    existing.collections = []
    fake_client.get_collections.return_value = existing

    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        qdrant_db.ensure_collection("wisdom_base")
    fake_client.create_collection.assert_called_once()
    call = fake_client.create_collection.call_args
    assert call.kwargs["collection_name"] == "wisdom_base"
    assert call.kwargs["vectors_config"].size == 1536


def test_serialize_payload_drops_embedding_and_serializes_datetime():
    from teledigest.qdrant_db import _serialize_payload

    data = {
        "country": "br",
        "title": "CPF guide",
        "embedding": [0.1, 0.2, 0.3],  # must be dropped
        "createdAt": dt.datetime(2026, 5, 1, 12, 0, tzinfo=dt.timezone.utc),
    }
    p = _serialize_payload(data)
    assert "embedding" not in p
    assert p["country"] == "br"
    assert p["title"] == "CPF guide"
    assert p["createdAt"].startswith("2026-05-01")


def test_upsert_point_uses_doc_id_as_qdrant_id():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        qdrant_db.upsert_point(
            "wisdom_base",
            "abc123",
            [0.1] * 1536,
            {"country": "br"},
        )
    fake_client.upsert.assert_called_once()
    call = fake_client.upsert.call_args
    pts = call.kwargs["points"]
    assert len(pts) == 1
    # _point_id оборачивает doc_id в детерминированный UUID5 (Qdrant
    # требует UUID или uint, sha1-hex-24 не принимается).
    import uuid as _uuid

    expected = str(_uuid.uuid5(qdrant_db._QDRANT_POINT_UUID_NAMESPACE, "abc123"))
    assert pts[0].id == expected
    assert pts[0].vector == [0.1] * 1536
    assert pts[0].payload == {"country": "br"}


def test_upsert_points_batch_empty_noop():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        qdrant_db.upsert_points_batch("wisdom_base", [])
    fake_client.upsert.assert_not_called()


def test_count_returns_int():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    result = MagicMock()
    result.count = 42
    fake_client.count.return_value = result
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        assert qdrant_db.count("wisdom_base") == 42


def test_count_returns_minus1_on_error():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    fake_client.count.side_effect = RuntimeError("network")
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        assert qdrant_db.count("wisdom_base") == -1


def test_find_nearest_adds_source_label():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_hit = MagicMock()
    fake_hit.payload = {"country": "br", "title": "CPF"}
    response = MagicMock()
    response.points = [fake_hit]

    fake_client = MagicMock()
    fake_client.query_points.return_value = response
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        out = qdrant_db.find_nearest(
            "wisdom_base", [0.1] * 1536, limit=10, source_label="База данных"
        )
    assert out == [{"country": "br", "title": "CPF", "_source": "База данных"}]


def test_find_nearest_returns_empty_on_error():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    fake_client.query_points.side_effect = RuntimeError("conn refused")
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        out = qdrant_db.find_nearest("wisdom_base", [0.1] * 1536, limit=10)
    assert out == []


def test_point_exists_true_when_retrieve_returns_data():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    fake_client.retrieve.return_value = [MagicMock()]
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        assert qdrant_db.point_exists("wisdom_base", "abc") is True


def test_point_exists_false_when_retrieve_empty():
    _stub_qdrant_modules()
    from teledigest import qdrant_db

    fake_client = MagicMock()
    fake_client.retrieve.return_value = []
    with (
        patch.object(qdrant_db, "get_config", return_value=_stub_config()),
        patch.object(qdrant_db, "get_client", return_value=fake_client),
    ):
        assert qdrant_db.point_exists("wisdom_base", "abc") is False
