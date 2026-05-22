"""Smoke tests for extraction_db SQLite layer.

Создаёт реальную in-memory SQLite через стандартный db.get_db_connection,
проверяет CRUD циклы для extracted_patterns / wikivoyage_patterns.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from teledigest import extraction_db


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch):
    """Точная инициализация SQLite в tmp_path + patch get_db_connection
    чтобы каждый тест работал с чистой БД."""
    db_path = tmp_path / "test_extraction.db"

    def _connect():
        conn = sqlite3.connect(str(db_path))
        return conn

    monkeypatch.setattr(extraction_db, "get_db_connection", _connect)
    extraction_db.init_extraction_tables()
    yield db_path


def test_init_tables_idempotent(temp_db):
    # вызов второй раз не должен падать
    extraction_db.init_extraction_tables()
    extraction_db.init_extraction_tables()


def test_insert_and_fetch_extracted_pending(temp_db):
    extraction_db.insert_extracted_pattern(
        id_="abc123",
        collection_target=extraction_db.COLLECTION_WISDOM,
        country="br",
        title="CPF guide",
        tag="Bureaucracy",
        routing="both",
        ai_lesson="Apply at Receita Federal.",
        human_story=None,
        target_languages=None,
        source_country_file="2026-05-20_br_chat.txt",
        source_country_file_idx=3,
    )
    pending = extraction_db.fetch_pending_extracted(
        extraction_db.COLLECTION_WISDOM, limit=10
    )
    assert len(pending) == 1
    row = pending[0]
    assert row["id"] == "abc123"
    assert row["country"] == "br"
    assert row["title"] == "CPF guide"
    assert row["ai_lesson"] == "Apply at Receita Federal."
    assert row["target_languages"] == []


def test_insert_idempotent_on_duplicate_id(temp_db):
    args = dict(
        id_="abc123",
        collection_target=extraction_db.COLLECTION_WISDOM,
        country="br",
        title="CPF guide",
        tag="Bureaucracy",
        routing="both",
        ai_lesson="lesson v1",
        human_story=None,
        target_languages=None,
        source_country_file="f.txt",
        source_country_file_idx=0,
    )
    extraction_db.insert_extracted_pattern(**args)
    args["ai_lesson"] = "lesson v2 (would overwrite)"
    extraction_db.insert_extracted_pattern(**args)
    pending = extraction_db.fetch_pending_extracted(
        extraction_db.COLLECTION_WISDOM, limit=10
    )
    assert len(pending) == 1
    # OR IGNORE → первый insert остался
    assert pending[0]["ai_lesson"] == "lesson v1"


def test_target_languages_json_round_trip(temp_db):
    extraction_db.insert_extracted_pattern(
        id_="x1",
        collection_target=extraction_db.COLLECTION_STORIES,
        country="br",
        title="Story",
        tag="Culture",
        routing="channel_only",
        ai_lesson=None,
        human_story="Once upon a time in Brazil...",
        target_languages=["ru", "en", "pt"],
        source_country_file="f.txt",
        source_country_file_idx=0,
    )
    pending = extraction_db.fetch_pending_extracted(
        extraction_db.COLLECTION_STORIES, limit=10
    )
    assert len(pending) == 1
    assert pending[0]["target_languages"] == ["ru", "en", "pt"]


def test_mark_embedded_removes_from_pending(temp_db):
    extraction_db.insert_extracted_pattern(
        id_="x1",
        collection_target=extraction_db.COLLECTION_WISDOM,
        country="br",
        title="T",
        tag="G",
        routing="both",
        ai_lesson="lesson",
        human_story=None,
        target_languages=None,
        source_country_file="f.txt",
        source_country_file_idx=0,
    )
    assert (
        len(
            extraction_db.fetch_pending_extracted(
                extraction_db.COLLECTION_WISDOM, limit=10
            )
        )
        == 1
    )
    extraction_db.mark_embedded("extracted_patterns", ["x1"])
    assert (
        len(
            extraction_db.fetch_pending_extracted(
                extraction_db.COLLECTION_WISDOM, limit=10
            )
        )
        == 0
    )


def test_mark_embed_failed_increments_count(temp_db):
    extraction_db.insert_extracted_pattern(
        id_="x1",
        collection_target=extraction_db.COLLECTION_WISDOM,
        country="br",
        title="T",
        tag="G",
        routing="both",
        ai_lesson="lesson",
        human_story=None,
        target_languages=None,
        source_country_file="f.txt",
        source_country_file_idx=0,
    )
    extraction_db.mark_embed_failed("extracted_patterns", "x1", "Quota exhausted")
    extraction_db.mark_embed_failed("extracted_patterns", "x1", "Quota exhausted")
    # 2 failures — всё ещё в pending (cap = 5)
    pending = extraction_db.fetch_pending_extracted(
        extraction_db.COLLECTION_WISDOM, limit=10
    )
    assert len(pending) == 1
    # После 5 failures выпадает из pending выборки
    for _ in range(3):
        extraction_db.mark_embed_failed("extracted_patterns", "x1", "Q")
    pending = extraction_db.fetch_pending_extracted(
        extraction_db.COLLECTION_WISDOM, limit=10
    )
    assert len(pending) == 0


def test_insert_wiki_pattern_idempotent_returns_bool(temp_db):
    inserted_first = extraction_db.insert_wiki_pattern(
        id_="w1",
        country="th",
        title="Bangkok",
        tag="Travel",
        instruction="Visit grand palace.",
        source_title="Bangkok",
        source_url="https://en.wikivoyage.org/wiki/Bangkok",
    )
    assert inserted_first is True
    inserted_second = extraction_db.insert_wiki_pattern(
        id_="w1",
        country="th",
        title="Bangkok",
        tag="Travel",
        instruction="Different content",
        source_title="Bangkok",
        source_url="https://en.wikivoyage.org/wiki/Bangkok",
    )
    assert inserted_second is False
    pending = extraction_db.fetch_pending_wiki(limit=10)
    assert len(pending) == 1
    # OR IGNORE → первый instruction остался
    assert pending[0]["instruction"] == "Visit grand palace."


def test_stats_returns_counts(temp_db):
    extraction_db.insert_extracted_pattern(
        id_="x1",
        collection_target=extraction_db.COLLECTION_WISDOM,
        country="br",
        title="T",
        tag="G",
        routing="both",
        ai_lesson="lesson",
        human_story=None,
        target_languages=None,
        source_country_file="f.txt",
        source_country_file_idx=0,
    )
    extraction_db.insert_extracted_pattern(
        id_="x2",
        collection_target=extraction_db.COLLECTION_STORIES,
        country="th",
        title="T2",
        tag="G",
        routing="channel_only",
        ai_lesson=None,
        human_story="story",
        target_languages=["ru"],
        source_country_file="f.txt",
        source_country_file_idx=1,
    )
    extraction_db.insert_wiki_pattern(
        id_="w1",
        country="th",
        title="Bangkok",
        tag="Travel",
        instruction="Visit X",
        source_title="Bangkok",
        source_url="https://w.org/Bangkok",
    )
    st = extraction_db.stats()
    assert "extracted (wisdom+stories)" in st
    assert st["extracted (wisdom+stories)"]["total"] == 2
    assert st["extracted (wisdom+stories)"]["pending"] == 2
    assert "wiki" in st
    assert st["wiki"]["total"] == 1
    assert st["wiki"]["pending"] == 1
