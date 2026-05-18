"""Unit tests for the v2 (cipher-fix) embedding functions.

The single-text path goes through google-genai SDK
(`client.models.embed_content`). The batch path goes through direct REST on
`:batchEmbedContents`. Tests mock both.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from teledigest import gemini_brain
from teledigest.gemini_brain import (
    _EMBEDDING_DIM_V2,
    _EMBEDDING_MODEL_V2,
    compute_document_embeddings_v2,
    compute_query_embedding_v2,
)


@pytest.fixture(autouse=True)
def _reset_pool_state(monkeypatch):
    """Reset module-level key-pool state between tests for determinism."""
    gemini_brain._key_rr_idx = 0
    gemini_brain._key_rpd_count = {}
    # Silence the 5s intra-sweep gap so tests stay fast.
    yield
    gemini_brain._key_rr_idx = 0
    gemini_brain._key_rpd_count = {}


def _fake_client_returning(vectors: list[list[float]]):
    """genai.Client mock whose embed_content returns vectors one-by-one."""
    fake_models = MagicMock()
    fake_results = [MagicMock(embeddings=[MagicMock(values=v)]) for v in vectors]
    fake_models.embed_content = MagicMock(side_effect=fake_results)
    fake_client = MagicMock()
    fake_client.models = fake_models
    return fake_client


def test_query_embed_empty_returns_none():
    assert compute_query_embedding_v2("") is None
    assert compute_query_embedding_v2("   ") is None


def test_document_embed_empty_returns_empty():
    assert compute_document_embeddings_v2([]) == []


def test_invalid_task_type_raises():
    from teledigest.gemini_brain import _embed_v2

    with pytest.raises(ValueError, match="task_type must be"):
        _embed_v2(["text"], "INVALID_TYPE")


def test_query_uses_retrieval_query_task_type():
    fake = _fake_client_returning([[0.1] * _EMBEDDING_DIM_V2])
    with (
        patch("google.genai.Client", return_value=fake),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_query_embedding_v2("rent a car")

    assert result is not None
    assert len(result) == _EMBEDDING_DIM_V2
    call_kwargs = fake.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == _EMBEDDING_MODEL_V2
    cfg = call_kwargs["config"]
    assert cfg.task_type == "RETRIEVAL_QUERY"
    assert cfg.output_dimensionality == _EMBEDDING_DIM_V2


def test_document_uses_retrieval_document_task_type():
    fake = _fake_client_returning(
        [[0.2] * _EMBEDDING_DIM_V2, [0.3] * _EMBEDDING_DIM_V2]
    )
    with (
        patch("google.genai.Client", return_value=fake),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_document_embeddings_v2(["doc one", "doc two"])

    assert len(result) == 2
    cfg = fake.models.embed_content.call_args.kwargs["config"]
    assert cfg.task_type == "RETRIEVAL_DOCUMENT"


def test_no_api_key_returns_none_for_query():
    with patch("teledigest.gemini_brain._get_embedding_api_keys", return_value=[]):
        assert compute_query_embedding_v2("hello") is None


def test_no_api_key_returns_nones_for_documents():
    with patch("teledigest.gemini_brain._get_embedding_api_keys", return_value=[]):
        result = compute_document_embeddings_v2(["a", "b", "c"])
    assert result == [None, None, None]


def test_api_failure_returns_nones(monkeypatch):
    """If every per-text call raises, every slot in the result is None."""
    # Squash the 5s intra-sweep gap so the test doesn't actually sleep.
    monkeypatch.setattr(gemini_brain, "_RPD_SOFT_CAP", 950)
    fake_models = MagicMock()
    fake_models.embed_content = MagicMock(side_effect=RuntimeError("api down"))
    fake_client = MagicMock()
    fake_client.models = fake_models
    with (
        patch("google.genai.Client", return_value=fake_client),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_document_embeddings_v2(["a", "b"])
    assert result == [None, None]


def test_partial_failure_other_texts_succeed():
    """One text raising doesn't poison the rest of the batch."""
    fake_models = MagicMock()
    good = MagicMock(embeddings=[MagicMock(values=[0.1] * _EMBEDDING_DIM_V2)])
    fake_models.embed_content = MagicMock(
        side_effect=[good, RuntimeError("transient"), good]
    )
    fake_client = MagicMock()
    fake_client.models = fake_models
    with (
        patch("google.genai.Client", return_value=fake_client),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_document_embeddings_v2(["a", "b", "c"])
    assert result[0] is not None
    assert result[1] is None
    assert result[2] is not None


def test_custom_dim_passed_through():
    fake = _fake_client_returning([[0.1] * 1536])
    with (
        patch("google.genai.Client", return_value=fake),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        compute_query_embedding_v2("x", dim=1536)
    cfg = fake.models.embed_content.call_args.kwargs["config"]
    assert cfg.output_dimensionality == 1536
