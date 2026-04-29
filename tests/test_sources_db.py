from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from teledigest import config as cfg
from teledigest import sources_db


def _make_app_config(db_path: Path) -> cfg.AppConfig:
    telegram = cfg.TelegramConfig(
        api_id=1, api_hash="h", bot_token="t", sessions_dir=Path("testdata")
    )
    bot = cfg.BotConfig(channels=["@c1"], summary_target="@digest")
    llm = cfg.LLMConfig(model="x", api_key="k", system_prompt="", user_prompt="")
    storage = cfg.StorageConfig(rag_keywords=[], db_path=db_path)
    logging_cfg = cfg.LoggingConfig(level="INFO")
    return cfg.AppConfig(
        telegram=telegram, bot=bot, llm=llm, storage=storage, logging=logging_cfg
    )


@pytest.fixture
def app_config(tmp_path, monkeypatch) -> cfg.AppConfig:
    db_path = tmp_path / "messages_fts.db"
    app_cfg = _make_app_config(db_path)
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)
    sources_db.init_sources_table()
    # Seed three real-world style sources
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT OR IGNORE INTO sources (country, url, name, language, added_at) "
        "VALUES (?, ?, ?, ?, '2026-04-24T00:00:00')",
        [
            ("br", "@Brazil_ChatForum", "Brazil Chat Forum", "ru"),
            ("id", "https://t.me/balichat", "balichat", "ru"),
            ("mu", "https://t.me/+iVaiJserqqNkNTc6", "-1001646838441", "ru"),
        ],
    )
    conn.commit()
    conn.close()
    return app_cfg


def test_normalize_url_handle_strips_at_prefix() -> None:
    assert sources_db._normalize_url_handle("@Brazil_ChatForum") == "Brazil_ChatForum"


def test_normalize_url_handle_extracts_t_me_handle() -> None:
    assert sources_db._normalize_url_handle("https://t.me/balichat") == "balichat"
    assert sources_db._normalize_url_handle("http://t.me/foo/123") == "foo"


def test_normalize_url_handle_keeps_invite_link_payload() -> None:
    # Invite-link hashes (start with +) get returned as-is; they will not
    # match a public channel handle, but the function shouldn't crash.
    assert sources_db._normalize_url_handle("https://t.me/+abc") == "+abc"


def test_normalize_url_handle_returns_none_for_empty() -> None:
    assert sources_db._normalize_url_handle("") is None
    assert sources_db._normalize_url_handle("   ") is None


def test_resolve_country_matches_url_handle(app_config: cfg.AppConfig) -> None:
    assert sources_db.resolve_country_for_channel("Brazil_ChatForum") == "br"
    assert sources_db.resolve_country_for_channel("balichat") == "id"


def test_resolve_country_matches_name_field(app_config: cfg.AppConfig) -> None:
    # Numeric chat_id stored in sources.name (invite-link channels)
    assert sources_db.resolve_country_for_channel("-1001646838441") == "mu"


def test_resolve_country_returns_none_for_unknown(app_config: cfg.AppConfig) -> None:
    assert sources_db.resolve_country_for_channel("unknown_channel") is None
    assert sources_db.resolve_country_for_channel("") is None


def test_build_channel_country_map_includes_all_keys(app_config: cfg.AppConfig) -> None:
    mapping = sources_db.build_channel_country_map()
    # name + url-handle both indexed
    assert mapping["Brazil_ChatForum"] == "br"
    assert mapping["Brazil Chat Forum"] == "br"      # name with spaces
    assert mapping["balichat"] == "id"               # via url + via name
    assert mapping["-1001646838441"] == "mu"
