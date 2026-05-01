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

import sqlite3

from teledigest import config as cfg
from teledigest import db
from teledigest import daily_samples as ds
from teledigest import sources_db


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
    sources_db.init_sources_table()
    # Seed sources matching real prod shapes:
    #   - br has TWO sources: public (Brazil_ChatForum) and invite-link
    #     (TravelAsk Brazil — channel value will be the chat_id)
    #   - id one public source (balichat)
    #   - lk one invite-link source (chat_id channel value)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO sources (country, url, name, language, added_at, chat_id) "
        "VALUES (?, ?, ?, 'ru', '2026-04-24T00:00:00', ?)",
        [
            ("br", "@Brazil_ChatForum", "Brazil Chat Forum", -1001221994108),
            ("br", "https://t.me/+invite_br", "TravelAsk Brazil", -1001631614451),
            ("id", "https://t.me/balichat", "balichat", -1001032422089),
            ("lk", "https://t.me/+invite_lk", "Sri Lanka Chat", -1001605996131),
        ],
    )
    conn.commit()
    conn.close()
    return app_cfg


def test_dump_all_targets_uses_sources_db_and_isolates(app_config):
    """
    Mini-pipeline:
      Setup: 4 sources in DB (2 br: public + invite-link, 1 id, 1 lk).
             6 target messages + noise (bot message, other country).
      Action: dump_all_targets(yesterday) — reads targets from DB.
      Expectation: 4 files written (one per source), filenames include
                   country and channel slug, each file isolated to its
                   source's messages only.
    """
    day = dt.date(2026, 4, 28)
    base = dt.datetime.combine(day, dt.time(12, 0), dt.timezone.utc)

    # Public br source (Brazil_ChatForum) — channel value = handle
    db.save_message("Brazil_ChatForum_100", "Brazil_ChatForum", base, "br public msg",
                    sender_id=111, country="br")
    db.save_message("Brazil_ChatForum_101", "Brazil_ChatForum", base + dt.timedelta(minutes=1),
                    "br public reply", sender_id=222,
                    reply_to_msg_id="Brazil_ChatForum_100", country="br")
    # Invite-link br source (TravelAsk Brazil) — channel value = numeric chat_id
    db.save_message("invite_br_50", "-1001631614451", base, "br invite msg",
                    sender_id=333, country="br")
    # id source (balichat handle)
    db.save_message("balichat_a", "balichat", base, "id msg",
                    sender_id=444, country="id")
    # lk source (invite-link numeric)
    db.save_message("lk_a", "-1001605996131", base, "lk msg",
                    sender_id=555, country="lk")

    # Noise: bot message in a target channel
    db.save_message("bot_a", "Brazil_ChatForum", base, "bot spam",
                    sender_id=1, is_bot=True, country="br")
    # Noise: country with no source row in our seeded DB
    db.save_message("at_a", "-1001716659625", base, "at msg",
                    sender_id=999, country="at")

    results = ds.dump_all_targets(day)

    # 4 sources in DB → 4 files
    assert len(results) == 4
    by_channel = {t.channel: (t, path, count) for (t, path, count) in results}

    # Brazil_ChatForum (public br) file
    _, br_pub_path, br_pub_count = by_channel["Brazil_ChatForum"]
    assert br_pub_count == 2  # public reply chain, NOT invite-link msg, NOT bot
    br_pub_text = br_pub_path.read_text(encoding="utf-8")
    assert "br public msg" in br_pub_text
    assert "br public reply" in br_pub_text
    assert "br invite msg" not in br_pub_text  # other br source isolated
    assert "bot spam" not in br_pub_text
    assert "← reply 100" in br_pub_text
    assert "u/111" in br_pub_text
    assert "@" not in br_pub_text

    # Filename contains country and channel slug — no two files share name
    assert "br_Brazil_ChatForum.txt" in br_pub_path.name

    # TravelAsk Brazil (invite-link br) file — numeric chat_id slug strips leading dash
    _, br_inv_path, br_inv_count = by_channel["-1001631614451"]
    assert br_inv_count == 1
    assert "1001631614451.txt" in br_inv_path.name  # leading dash stripped
    br_inv_text = br_inv_path.read_text(encoding="utf-8")
    assert "br invite msg" in br_inv_text
    assert "br public msg" not in br_inv_text  # other br source isolated

    # id file
    _, id_path, id_count = by_channel["balichat"]
    assert id_count == 1
    assert "id_balichat.txt" in id_path.name
    assert "id msg" in id_path.read_text(encoding="utf-8")

    # lk file
    _, lk_path, lk_count = by_channel["-1001605996131"]
    assert lk_count == 1
    assert "lk_1001605996131.txt" in lk_path.name


def test_filename_pattern(app_config, tmp_path):
    """Path is samples/{country}/{date}_{country}_{channel-slug}.txt"""
    day = dt.date(2026, 4, 28)
    target = ds.SampleTarget(country="br", channel="Brazil_ChatForum")
    path, _ = ds.dump_country_samples(target, day)
    assert path == tmp_path / "samples" / "br" / "2026-04-28_br_Brazil_ChatForum.txt"


def test_filename_strips_leading_dash_from_numeric_chat_id(app_config, tmp_path):
    day = dt.date(2026, 4, 28)
    target = ds.SampleTarget(country="lk", channel="-1001605996131")
    path, _ = ds.dump_country_samples(target, day)
    assert path == tmp_path / "samples" / "lk" / "2026-04-28_lk_1001605996131.txt"


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


def test_get_sample_targets_picks_chat_id_for_invite_links(app_config):
    """Invite-link sources should produce SampleTarget with numeric chat_id."""
    targets = ds.get_sample_targets()
    by_country = {(t.country, t.channel) for t in targets}
    # Public br source picks the URL handle as channel
    assert ("br", "Brazil_ChatForum") in by_country
    # Invite-link br source picks the numeric chat_id
    assert ("br", "-1001631614451") in by_country
    # Public id source — handle
    assert ("id", "balichat") in by_country
    # Invite-link lk source — numeric
    assert ("lk", "-1001605996131") in by_country
    assert len(targets) == 4


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
