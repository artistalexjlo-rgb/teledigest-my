"""Unit tests for the v2 (cipher-fix) embedding functions.

Verifies the surface contract without hitting the live API:
- task_type validation
- empty input handling
- passes task_type and dim to the REST endpoint correctly
- 429 → falls over to next key; non-quota error → abandons text
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
def _reset_pool_state():
    """Reset the round-robin pointer between tests for determinism."""
    gemini_brain._key_rr_idx = 0
    yield
    gemini_brain._key_rr_idx = 0


def _fake_resp(status: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=json_body or {})
    r.text = text
    return r


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
    vec = [0.1] * _EMBEDDING_DIM_V2
    fake_post = MagicMock(return_value=_fake_resp(200, {"embedding": {"values": vec}}))
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_query_embedding_v2("rent a car")

    assert result is not None
    assert len(result) == _EMBEDDING_DIM_V2
    call = fake_post.call_args
    assert _EMBEDDING_MODEL_V2 in call.args[0]
    assert ":embedContent" in call.args[0]
    body = call.kwargs["json"]
    assert body["taskType"] == "RETRIEVAL_QUERY"
    assert body["outputDimensionality"] == _EMBEDDING_DIM_V2


def test_document_uses_retrieval_document_task_type():
    vec = [0.2] * _EMBEDDING_DIM_V2
    fake_post = MagicMock(return_value=_fake_resp(200, {"embedding": {"values": vec}}))
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        result = compute_document_embeddings_v2(["doc one", "doc two"])

    assert len(result) == 2
    body = fake_post.call_args.kwargs["json"]
    assert body["taskType"] == "RETRIEVAL_DOCUMENT"


def test_no_api_key_returns_none_for_query():
    with patch("teledigest.gemini_brain._get_embedding_api_keys", return_value=[]):
        assert compute_query_embedding_v2("hello") is None


def test_no_api_key_returns_nones_for_documents():
    with patch("teledigest.gemini_brain._get_embedding_api_keys", return_value=[]):
        result = compute_document_embeddings_v2(["a", "b", "c"])
    assert result == [None, None, None]


def test_429_falls_over_to_next_key():
    """First key returns 429, second key succeeds — text gets vector from #1."""
    good_vec = [0.3] * _EMBEDDING_DIM_V2
    fake_post = MagicMock(
        side_effect=[
            _fake_resp(429, text="RESOURCE_EXHAUSTED limit 1000"),
            _fake_resp(200, {"embedding": {"values": good_vec}}),
        ]
    )
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys",
            return_value=["k0", "k1"],
        ),
    ):
        result = compute_document_embeddings_v2(["a"])
    assert result == [good_vec]
    assert fake_post.call_count == 2


def test_all_keys_429_returns_none():
    fake_post = MagicMock(
        return_value=_fake_resp(429, text="RESOURCE_EXHAUSTED limit 1000")
    )
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys",
            return_value=["k0", "k1"],
        ),
    ):
        result = compute_document_embeddings_v2(["a"])
    assert result == [None]
    assert fake_post.call_count == 2


def test_non_quota_error_abandons_text_without_burning_other_keys():
    """500 on first key → don't try the second; just record None for that text."""
    fake_post = MagicMock(return_value=_fake_resp(500, text="oops"))
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys",
            return_value=["k0", "k1", "k2"],
        ),
    ):
        result = compute_document_embeddings_v2(["a", "b"])
    assert result == [None, None]
    # Two texts × one HTTP call each (first key only, then abandon).
    assert fake_post.call_count == 2


def test_custom_dim_passed_through():
    fake_post = MagicMock(
        return_value=_fake_resp(200, {"embedding": {"values": [0.1] * 1536}})
    )
    with (
        patch("requests.post", fake_post),
        patch(
            "teledigest.gemini_brain._get_embedding_api_keys", return_value=["fake-key"]
        ),
    ):
        compute_query_embedding_v2("x", dim=1536)
    body = fake_post.call_args.kwargs["json"]
    assert body["outputDimensionality"] == 1536
