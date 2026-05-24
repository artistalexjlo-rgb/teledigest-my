"""Тесты для channel_poster — pure-функции выбора и форматирования.

Покрытие:
- select_next_candidate: alphabetical round-robin по странам,
  пропуск recent_countries, пропуск excluded, пустая очередь
- format_message: контент + хештеги (страна + tag)
- _channel_field_safe: санитизация @luky_channel → luky_channel
- pattern_posts CRUD: mark, unposted-фильтр, recent_posted_countries
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from teledigest import channel_poster, extraction_db


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_poster.db"

    def _connect():
        return sqlite3.connect(str(db_path))

    monkeypatch.setattr(extraction_db, "get_db_connection", _connect)
    extraction_db.init_extraction_tables()
    yield db_path


def _insert_story(country: str, idx: int, story: str = "test story") -> str:
    """Helper — добавить story в extracted_patterns."""
    pattern_id = f"{country}_{idx}".ljust(24, "x")[:24]
    extraction_db.insert_extracted_pattern(
        id_=pattern_id,
        collection_target="telegram_queue",
        country=country,
        title=f"Title {idx}",
        tag="Travel",
        routing="channel_only",
        ai_lesson=None,
        human_story=story,
        target_languages=["ru"],
        source_country_file="t.txt",
        source_country_file_idx=idx,
    )
    return pattern_id


def test_channel_field_safe():
    assert channel_poster._channel_field_safe("@luky_channel") == "luky_channel"
    assert channel_poster._channel_field_safe("-1001234567890") == "1001234567890"
    assert channel_poster._channel_field_safe("") == "default"
    assert channel_poster._channel_field_safe("Some@Chat!") == "somechat"


def test_country_hashtag_known():
    # br → Бразилия (из COUNTRIES в country_codes.py)
    tag = channel_poster._country_hashtag("br")
    assert tag.startswith("#")
    assert "Бразили" in tag  # допускаем разные склонения "Бразилия"


def test_country_hashtag_any_special():
    assert channel_poster._country_hashtag("any") == "#ЛайфхакиВПути"


def test_country_hashtag_unknown_falls_back_to_iso():
    # 'zz' не в COUNTRIES → fallback на uppercase
    assert channel_poster._country_hashtag("zz") == "#ZZ"


def test_tag_hashtag():
    assert channel_poster._tag_hashtag("Finance") == "#Finance"
    assert channel_poster._tag_hashtag("") == ""
    assert channel_poster._tag_hashtag("Travel Tips") == "#TravelTips"


def test_format_message_combines_content_and_hashtags():
    cand = channel_poster.PostCandidate(
        pattern_id="x",
        country="br",
        title="t",
        content="История про CPF",
        tag="Bureaucracy",
    )
    msg = channel_poster.format_message(cand)
    assert "История про CPF" in msg
    assert "#Бразили" in msg  # допускаем разные склонения
    assert "#Bureaucracy" in msg


def test_select_returns_none_on_empty_queue(temp_db):
    cand = channel_poster.select_next_candidate("luky")
    assert cand is None


def test_select_round_robin_alphabetical(temp_db):
    # br и id, каждая по 2 истории. Без recent_countries → возьмёт br (alpha-first).
    _insert_story("br", 1)
    _insert_story("br", 2)
    _insert_story("id", 1)
    _insert_story("id", 2)

    c = channel_poster.select_next_candidate("luky", recent_countries=[])
    assert c is not None
    assert c.country == "br"

    # Если только что постили br → следующая страна id
    c = channel_poster.select_next_candidate("luky", recent_countries=["br"])
    assert c is not None
    assert c.country == "id"


def test_select_skips_recent_countries(temp_db):
    _insert_story("br", 1)
    _insert_story("id", 1)
    _insert_story("vn", 1)

    # Recent = br, id → должен взять vn
    c = channel_poster.select_next_candidate("luky", recent_countries=["br", "id"])
    assert c is not None
    assert c.country == "vn"


def test_select_falls_back_when_all_in_recent(temp_db):
    # В очереди только br → даже если в recent, должны взять br чтоб не голодать
    _insert_story("br", 1)
    c = channel_poster.select_next_candidate("luky", recent_countries=["br"])
    assert c is not None
    assert c.country == "br"


def test_select_excludes_country(temp_db):
    _insert_story("br", 1)
    _insert_story("ru", 1)
    c = channel_poster.select_next_candidate(
        "luky", recent_countries=[], excluded_countries={"ru"}
    )
    assert c is not None
    assert c.country == "br"


def test_select_skips_already_posted(temp_db):
    pid_br = _insert_story("br", 1)
    _insert_story("id", 1)
    # Помечаем br как уже posted в luky → должен выдать id
    extraction_db.mark_pattern_posted(pid_br, "luky", "posted text")

    c = channel_poster.select_next_candidate("luky", recent_countries=[])
    assert c is not None
    assert c.country == "id"


def test_mark_posted_idempotent(temp_db):
    pid = _insert_story("br", 1)
    extraction_db.mark_pattern_posted(pid, "luky", "first")
    # Повторный INSERT через PRIMARY KEY (pattern_id, channel) — silent noop
    extraction_db.mark_pattern_posted(pid, "luky", "second")

    with extraction_db.get_db_connection() as c:
        cur = c.cursor()
        cur.execute(
            "SELECT COUNT(*), posted_text FROM pattern_posts WHERE pattern_id=?",
            (pid,),
        )
        cnt, text = cur.fetchone()
    assert cnt == 1
    # INSERT OR IGNORE — первая запись сохраняется
    assert text == "first"


def test_recent_posted_countries_oldest_first(temp_db):
    pid_br = _insert_story("br", 1)
    pid_id = _insert_story("id", 1)
    pid_vn = _insert_story("vn", 1)

    import time

    extraction_db.mark_pattern_posted(pid_br, "luky", "")
    time.sleep(0.01)
    extraction_db.mark_pattern_posted(pid_id, "luky", "")
    time.sleep(0.01)
    extraction_db.mark_pattern_posted(pid_vn, "luky", "")

    recent = extraction_db.recent_posted_countries("luky", n=3)
    # Возвращает oldest-first, т.е. recent[-1] = самая свежая
    assert recent == ["br", "id", "vn"]


def test_recent_posted_countries_per_channel(temp_db):
    pid = _insert_story("br", 1)
    extraction_db.mark_pattern_posted(pid, "luky", "")
    # vk не получал ничего
    assert extraction_db.recent_posted_countries("vk_main") == []
    assert extraction_db.recent_posted_countries("luky") == ["br"]


def test_todays_slots_5_per_day():
    class _Cfg:
        class channel:
            posts_per_day = 5
            window_start_hour = 8
            window_end_hour = 24

    slots = channel_poster._todays_slots(_Cfg())
    assert len(slots) == 5
    # Первый — 08:00, последний — 20:48 (8 + 4*3.2h)
    assert slots[0].hour == 8 and slots[0].minute == 0
    assert slots[-1].hour == 20 and slots[-1].minute == 48
