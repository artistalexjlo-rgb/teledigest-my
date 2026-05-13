"""
Tests for gemini_brain.py

Mini-pipeline scenario (process_minipipeline.md rule):
  User asks "МОЗГ где сделать SIM-карту в Бразилии"
  → Gemini enabled → wisdom_base has 2 docs for 'br' + 1 universal 'any'
  → Gemini synthesizes Russian answer
  → User sees 🧠 answer with source count

Fallback scenario:
  Gemini enabled but returns empty → falls back to DeepSeek path
  Gemini disabled → goes straight to DeepSeek path
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from teledigest.gemini_brain import _format_context, is_enabled

# --- Unit: _format_context ---------------------------------------------------


def test_format_context_basic():
    docs = [
        {
            "title": "SIM card Brazil",
            "tag": "Telecom",
            "country": "br",
            "instruction": "Vivo and Claro are the best carriers. SIM costs R$20.",
        },
        {
            "title": "Airport SIM",
            "tag": "Travel",
            "country": "any",
            "instruction": "Most airports have carrier kiosks past customs.",
        },
        {
            "title": "No instruction",
            "tag": "Other",
            "country": "br",
            "instruction": "",
        },  # should be skipped
    ]
    result = _format_context(docs)
    assert "[1." in result
    assert "[2." in result
    assert "[3." not in result  # empty instruction skipped
    assert "Vivo and Claro" in result
    assert "airport" in result.lower()


def test_format_context_empty():
    assert _format_context([]) == ""


def test_format_context_all_empty_instructions():
    docs = [{"title": "X", "tag": "Y", "country": "br", "instruction": ""}]
    assert _format_context(docs) == ""


# --- Unit: is_enabled --------------------------------------------------------


def test_is_enabled_with_key(monkeypatch):
    from teledigest import config as cfg_mod

    mock_cfg = MagicMock()
    mock_cfg.gemini.api_key = "fake-key"
    monkeypatch.setattr(cfg_mod, "_CONFIG", mock_cfg)
    assert is_enabled() is True


def test_is_enabled_no_key(monkeypatch):
    from teledigest import config as cfg_mod

    mock_cfg = MagicMock()
    mock_cfg.gemini.api_key = ""
    monkeypatch.setattr(cfg_mod, "_CONFIG", mock_cfg)
    assert is_enabled() is False


# --- Integration: search_and_format routes correctly -------------------------


def _make_mock_config(gemini_key: str = "fake-key", firestore_project: str = "proj"):
    cfg = MagicMock()
    cfg.gemini.api_key = gemini_key
    cfg.gemini.model = "gemini-2.0-flash"
    cfg.google.firestore_project_id = firestore_project
    cfg.google.firestore_database = "default"
    cfg.google.assistant_collection = "wisdom_base"
    cfg.google.token_path = MagicMock()
    cfg.google.token_path.exists.return_value = True
    return cfg


_SAMPLE_DOCS = [
    {
        "title": "SIM Brazil",
        "tag": "Telecom",
        "country": "br",
        "instruction": "Vivo is best, R$20 SIM, ID required.",
    },
    {
        "title": "Airport SIM",
        "tag": "Travel",
        "country": "any",
        "instruction": "Airport kiosks available past customs.",
    },
]


@pytest.mark.asyncio
@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom_and_wiki")
@patch("teledigest.gemini_brain._ask_live_api")
async def test_search_and_format_gemini_path(mock_live, mock_fetch, mock_get_cfg):
    """
    Mini-pipeline: _fetch_wisdom returns 2 docs → Live API synthesizes →
    answer returned.
    User scenario: asks 'где сделать SIM-карту', country=br.
    """
    cfg = _make_mock_config()
    cfg.gemini.live_model = "gemini-3.1-flash-live-preview"
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    async def _fake_live(prompt, model_name, api_key, history=None):
        assert "Vivo" in prompt  # context made it into the prompt
        assert model_name == "gemini-3.1-flash-live-preview"
        return "Лучший оператор — Vivo. SIM стоит R$20, нужен паспорт."

    mock_live.side_effect = _fake_live

    from teledigest.gemini_brain import search_and_format

    result = await search_and_format("br", "где сделать SIM-карту")

    assert "🧠" in result
    assert "Vivo" in result
    assert "записей из базы знаний" in result
    # _fetch_wisdom now takes no country argument — broad retrieval.
    mock_fetch.assert_called_once_with()


@pytest.mark.asyncio
@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom_and_wiki")
@patch("teledigest.gemini_brain._ask_live_api")
@patch("teledigest.gemini_brain._ask_sync_fallback")
async def test_search_and_format_falls_back_to_sync_when_live_fails(
    mock_sync,
    mock_live,
    mock_fetch,
    mock_get_cfg,
):
    """Live API raises → sync fallback runs → answer returned."""
    cfg = _make_mock_config()
    cfg.gemini.live_model = "gemini-3.1-flash-live-preview"
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    async def _fake_live_raises(*a, **kw):
        raise RuntimeError("simulated Live API outage")

    mock_live.side_effect = _fake_live_raises

    async def _fake_sync(prompt, model_name, api_key, history=None):
        return "Ответ из синхронного API."

    mock_sync.side_effect = _fake_sync

    from teledigest.gemini_brain import search_and_format

    result = await search_and_format("br", "anything")

    assert "🧠" in result
    assert "синхронного" in result
    mock_sync.assert_called_once()


@pytest.mark.asyncio
@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom_and_wiki")
@patch("teledigest.gemini_brain._ask_live_api")
async def test_search_and_format_passes_history_to_live(
    mock_live, mock_fetch, mock_get_cfg
):
    """history list reaches _ask_live_api as keyword arg."""
    cfg = _make_mock_config()
    cfg.gemini.live_model = "gemini-3.1-flash-live-preview"
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    received_history = {}

    async def _fake_live(prompt, model_name, api_key, history=None):
        received_history["value"] = history
        return "ответ с учётом истории"

    mock_live.side_effect = _fake_live

    history = [
        {"role": "user", "text": "что по такси шри-ланка"},
        {"role": "model", "text": "Автобусы, аренда, такси по приложению..."},
    ]
    from teledigest.gemini_brain import search_and_format

    result = await search_and_format("default", "Какое есть такси", history=history)

    assert "ответ с учётом истории" in result
    assert received_history["value"] == history


@pytest.mark.asyncio
@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom_and_wiki")
@patch("teledigest.gemini_brain._ask_live_api")
@patch("teledigest.gemini_brain._ask_sync_fallback")
async def test_search_and_format_both_paths_empty_returns_empty(
    mock_sync,
    mock_live,
    mock_fetch,
    mock_get_cfg,
):
    """Both Live and sync return empty → '' so caller hits DeepSeek."""
    cfg = _make_mock_config()
    cfg.gemini.live_model = "gemini-3.1-flash-live-preview"
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    async def _empty(*a, **kw):
        return ""

    mock_live.side_effect = _empty
    mock_sync.side_effect = _empty

    from teledigest.gemini_brain import search_and_format

    result = await search_and_format("br", "anything")
    assert result == ""


# --- Integration: knowledge_search falls back when Gemini disabled -----------


@pytest.mark.asyncio
async def test_knowledge_search_fallback_when_gemini_disabled(monkeypatch):
    """
    When Gemini is not enabled, search_and_format in knowledge_search.py
    should skip gemini_brain and go straight to DeepSeek+SQLite path.
    """
    import teledigest.gemini_brain as gb

    monkeypatch.setattr(gb, "is_enabled", lambda: False)

    import teledigest.knowledge_search as ks

    monkeypatch.setattr(ks, "search_knowledge", lambda country, query, limit=20: [])

    result = await ks.search_and_format("br", "anything")
    assert "Не нашёл ничего" in result
    assert "🧠" in result
