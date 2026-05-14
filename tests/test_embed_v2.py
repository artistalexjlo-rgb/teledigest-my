"""Unit tests for the v2 (cipher-fix) embedding functions.

Verifies the surface contract without hitting the live API:
- task_type validation
- empty input handling
- passes task_type and dim to the SDK correctly
- separates query vs document calls

Live API verification (vectors actually differ for QUERY vs DOCUMENT on same
text) is in scripts/verify_embed_v2.py — kept out of CI because it costs quota.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from teledigest.gemini_brain import (
    _EMBEDDING_DIM_V2,
    _EMBEDDING_MODEL_V2,
    compute_document_embeddings_v2,
    compute_query_embedding_v2,
)


def _fake_client_returning(vectors: list[list[float]]):
    """Build a fake genai.Client whose embed_content returns given vectors."""
    fake_embeddings = [MagicMock(values=v) for v in vectors]
    fake_result = MagicMock(embeddings=fake_embeddings)
    fake_models = MagicMock()
    fake_models.embed_content = MagicMock(return_value=fake_result)
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
            "teledigest.gemini_brain._get_embedding_api_key", return_value="fake-key"
        ),
    ):
        result = compute_query_embedding_v2("rent a car")

    assert result is not None
    assert len(result) == _EMBEDDING_DIM_V2
    call_kwargs = fake.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == _EMBEDDING_MODEL_V2
    assert call_kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"
    assert call_kwargs["config"]["output_dimensionality"] == _EMBEDDING_DIM_V2


def test_document_uses_retrieval_document_task_type():
    fake = _fake_client_returning(
        [[0.2] * _EMBEDDING_DIM_V2, [0.3] * _EMBEDDING_DIM_V2]
    )
    with (
        patch("google.genai.Client", return_value=fake),
        patch(
            "teledigest.gemini_brain._get_embedding_api_key", return_value="fake-key"
        ),
    ):
        result = compute_document_embeddings_v2(["doc one", "doc two"])

    assert len(result) == 2
    call_kwargs = fake.models.embed_content.call_args.kwargs
    assert call_kwargs["config"]["task_type"] == "RETRIEVAL_DOCUMENT"


def test_no_api_key_returns_none_for_query():
    with patch("teledigest.gemini_brain._get_embedding_api_key", return_value=""):
        assert compute_query_embedding_v2("hello") is None


def test_no_api_key_returns_nones_for_documents():
    with patch("teledigest.gemini_brain._get_embedding_api_key", return_value=""):
        result = compute_document_embeddings_v2(["a", "b", "c"])
    assert result == [None, None, None]


def test_api_failure_returns_nones():
    fake_models = MagicMock()
    fake_models.embed_content = MagicMock(side_effect=RuntimeError("api down"))
    fake_client = MagicMock()
    fake_client.models = fake_models
    with (
        patch("google.genai.Client", return_value=fake_client),
        patch(
            "teledigest.gemini_brain._get_embedding_api_key", return_value="fake-key"
        ),
    ):
        result = compute_document_embeddings_v2(["a", "b"])
    assert result == [None, None]


def test_custom_dim_passed_through():
    fake = _fake_client_returning([[0.1] * 1536])
    with (
        patch("google.genai.Client", return_value=fake),
        patch(
            "teledigest.gemini_brain._get_embedding_api_key", return_value="fake-key"
        ),
    ):
        compute_query_embedding_v2("x", dim=1536)
    assert (
        fake.models.embed_content.call_args.kwargs["config"]["output_dimensionality"]
        == 1536
    )
