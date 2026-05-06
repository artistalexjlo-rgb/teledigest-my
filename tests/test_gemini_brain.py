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
        {"title": "SIM card Brazil", "tag": "Telecom", "country": "br",
         "instruction": "Vivo and Claro are the best carriers. SIM costs R$20."},
        {"title": "Airport SIM", "tag": "Travel", "country": "any",
         "instruction": "Most airports have carrier kiosks past customs."},
        {"title": "No instruction", "tag": "Other", "country": "br",
         "instruction": ""},  # should be skipped
    ]
    result = _format_context(docs)
    assert "[1." in result
    assert "[2." in result
    assert "[3." not in result          # empty instruction skipped
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


def _make_mock_genai_module(answer_text: str):
    """Inject a fake google.generativeai module into sys.modules."""
    import sys
    import types as _types

    mock_genai = MagicMock()
    mock_response = MagicMock()
    mock_response.text = answer_text
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

    # Preserve real google package if present; only override generativeai submodule
    if "google" not in sys.modules:
        sys.modules["google"] = _types.ModuleType("google")
    sys.modules["google.generativeai"] = mock_genai
    return mock_genai


_SAMPLE_DOCS = [
    {"title": "SIM Brazil", "tag": "Telecom", "country": "br",
     "instruction": "Vivo is best, R$20 SIM, ID required."},
    {"title": "Airport SIM", "tag": "Travel", "country": "any",
     "instruction": "Airport kiosks available past customs."},
]


@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom")     # mock Firestore fetch
def test_search_and_format_gemini_path(mock_fetch, mock_get_cfg):
    """
    Mini-pipeline: _fetch_wisdom returns 2 docs → Gemini synthesizes → answer returned.
    User scenario: asks 'где сделать SIM-карту', country=br.
    """
    cfg = _make_mock_config()
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    mock_genai = _make_mock_genai_module("Лучший оператор — Vivo. SIM стоит R$20, нужен паспорт.")

    from teledigest.gemini_brain import search_and_format
    result = search_and_format("br", "где сделать SIM-карту")

    assert "🧠" in result
    assert "Vivo" in result
    assert "записей из базы знаний" in result
    mock_genai.configure.assert_called_once_with(api_key="fake-key")
    mock_fetch.assert_called_once_with("br")


@patch("teledigest.gemini_brain.get_config")
@patch("teledigest.gemini_brain._fetch_wisdom")
def test_search_and_format_gemini_empty_returns_empty(mock_fetch, mock_get_cfg):
    """Gemini returns empty text → search_and_format returns '' so caller fallbacks to DeepSeek."""
    cfg = _make_mock_config()
    mock_get_cfg.return_value = cfg
    mock_fetch.return_value = _SAMPLE_DOCS

    _make_mock_genai_module("")  # empty answer from Gemini

    from teledigest.gemini_brain import search_and_format
    result = search_and_format("br", "some question")
    assert result == ""  # empty → caller should fallback


# --- Integration: knowledge_search falls back when Gemini disabled -----------

def test_knowledge_search_fallback_when_gemini_disabled(monkeypatch):
    """
    When Gemini is not enabled, search_and_format in knowledge_search.py
    should skip gemini_brain and go straight to DeepSeek+SQLite path.
    """
    import teledigest.gemini_brain as gb
    monkeypatch.setattr(gb, "is_enabled", lambda: False)

    # Patch the SQLite search to return empty (simulates no results)
    import teledigest.knowledge_search as ks
    monkeypatch.setattr(ks, "search_knowledge", lambda country, query, limit=20: [])

    result = ks.search_and_format("br", "anything")
    assert "Не нашёл ничего" in result
    assert "🧠" in result
