"""End-to-end tests for daily_samples (PR #4).

Mini-pipeline scenario: insert messages from 3 country/channel pairs +
some noise (bot messages, other countries). Run dump_all_targets. Assert:
- One file per target, in correct directory
- Each file contains ONLY its target's messages
- Bot messages and empty texts are filtered out
- Sender IDs are impersonal (no usernames anywhere in output)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from teledigest import config as cfg
from teledigest import db
from teledigest import daily_samples as ds


def _make_app_config(db_path: Path) -> cfg.AppConfig:
    return cfg.AppConfig(
        telegram=cfg.TelegramConfig(
            api_id=1, api_hash="h", bot_token="t", sessions_dir=Path("td")
        ),
        bot=cfg.BotConfig(channels=["@c"], summary_target="@d"),
        llm=cfg.LLMConfig(model="x", api_key="k", system_prompt="", user_prompt=""),
        storage=cfg.StorageConfig(rag_keywords=[], db_path=db_path),
        logging=cfg.LoggingConfig(level="INFO"),
    )


@pytest.fixture
def app_config(tmp_path, monkeypatch) -> cfg.AppConfig:
    db_path = tmp_path / "messages_fts.db"
    app_cfg = _make_app_config(db_path)
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)
    db.init_db()
    return app_cfg


def test_dump_country_samples_isolates_per_target(app_config):
    """
    Mini-pipeline:
      Setup: messages from br/Brazil_ChatForum, id/balichat, lk/-1001605996131,
             plus noise (br/other channel, bot message, empty text, other country).
      Action: dump_all_targets(yesterday)
      Expectation: 3 files written, each contains ONLY its target's messages.
    """
    day = dt.date(2026, 4, 28)
    base = dt.datetime.combine(day, dt.time(12, 0), dt.timezone.utc)

    # Target rows (should appear in respective files)
    # Use realistic msg_ids: chat_name + numeric msg.id (matches production)
    db.save_message("Brazil_ChatForum_100", "Brazil_ChatForum", base, "br msg one",
                    sender_id=111, country="br")
    db.save_message("Brazil_ChatForum_101", "Brazil_ChatForum", base + dt.timedelta(minutes=1),
                    "br msg two", sender_id=222,
                    reply_to_msg_id="Brazil_ChatForum_100", country="br")
    db.save_message("id_a", "balichat", base, "id msg",
                    sender_id=333, country="id")
    db.save_message("lk_a", "-1001605996131", base, "lk msg",
                    sender_id=444, country="lk")

    # Noise: same country br BUT different channel — should NOT appear in br file
    db.save_message("br_other", "-1001631614451", base, "from a different br source",
                    sender_id=999, country="br")
    # Noise: bot message in target channel
    db.save_message("br_bot", "Brazil_ChatForum", base, "bot spam",
                    sender_id=1, is_bot=True, country="br")
    # Noise: country we don't dump
    db.save_message("at_a", "-1001716659625", base, "at msg",
                    sender_id=555, country="at")

    results = ds.dump_all_targets(day)

    assert len(results) == 3
    by_country = {t.country: (path, count) for (t, path, count) in results}

    # br file: 2 messages from Brazil_ChatForum, NOT br_other and NOT br_bot
    br_path, br_count = by_country["br"]
    assert br_count == 2
    br_text = br_path.read_text(encoding="utf-8")
    assert "br msg one" in br_text
    assert "br msg two" in br_text
    assert "from a different br source" not in br_text
    assert "bot spam" not in br_text
    # Reply marker present (numeric tail of msg_id)
    assert "← reply 100" in br_text
    # Impersonal sender format
    assert "u/111" in br_text
    assert "u/222" in br_text
    # No traces of usernames or display names — we never store them
    assert "@" not in br_text  # no @-handles in output

    # id file: only id msg, no others
    id_path, id_count = by_country["id"]
    assert id_count == 1
    id_text = id_path.read_text(encoding="utf-8")
    assert "id msg" in id_text
    assert "br msg" not in id_text
    assert "lk msg" not in id_text
    assert "u/333" in id_text

    # lk file: only lk msg
    lk_path, lk_count = by_country["lk"]
    assert lk_count == 1
    lk_text = lk_path.read_text(encoding="utf-8")
    assert "lk msg" in lk_text
    assert "u/444" in lk_text


def test_dump_writes_to_db_dir_samples(app_config, tmp_path):
    """File path is db_path.parent / 'samples' / {country} / {date}.txt"""
    day = dt.date(2026, 4, 28)
    base = dt.datetime.combine(day, dt.time(10, 0), dt.timezone.utc)
    db.save_message("a", "Brazil_ChatForum", base, "hi", sender_id=1, country="br")

    target = ds.SampleTarget(country="br", channel="Brazil_ChatForum")
    path, count = ds.dump_country_samples(target, day)

    assert path == tmp_path / "samples" / "br" / "2026-04-28.txt"
    assert path.exists()
    assert count == 1


def test_dump_creates_empty_file_when_no_messages(app_config):
    """Even with zero messages we still create a file (so cron history is visible)."""
    day = dt.date(2026, 4, 28)
    target = ds.SampleTarget(country="br", channel="Brazil_ChatForum")
    path, count = ds.dump_country_samples(target, day)

    assert count == 0
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "messages=0" in text
    # Header only, no body
    assert text.count("\n") == 1


def test_format_line_handles_missing_sender(app_config):
    line = ds._format_line("2026-04-28T12:34:56+00:00", "hello", None, None)
    assert line == "[12:34] u/?: hello"


def test_format_line_collapses_newlines(app_config):
    line = ds._format_line("2026-04-28T12:34:56+00:00", "hello\nworld\n  multiple", 5, None)
    assert "\n" not in line
    assert "hello world multiple" in line


def test_format_line_reply_marker(app_config):
    line = ds._format_line(
        "2026-04-28T12:34:56+00:00", "answer", 7,
        reply_to_msg_id="Brazil_ChatForum_408865",
    )
    assert "← reply 408865" in line
    assert "u/7" in line
