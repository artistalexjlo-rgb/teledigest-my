"""End-to-end tests for channel_poster (PR #7).

Mocks Firestore client and bot client. Verifies:
- Selection skips already-posted docs
- Country rotation prefers different country than last
- Excluded countries are filtered
- format_message includes content + country/tag hashtags
- Channel field name is sanitized for Firestore (no @ etc.)
- Schedule slots are evenly spaced in window
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from teledigest import config as cfg
from teledigest import channel_poster as cp


def _make_app_config(
    db_path: Path,
    *,
    channel_enabled: bool = True,
    target: str = "@luky_channel",
) -> cfg.AppConfig:
    return cfg.AppConfig(
        telegram=cfg.TelegramConfig(
            api_id=1, api_hash="h", bot_token="t", sessions_dir=Path("td")
        ),
        bot=cfg.BotConfig(channels=["@c"], summary_target="@d"),
        llm=cfg.LLMConfig(model="x", api_key="k", system_prompt="", user_prompt=""),
        storage=cfg.StorageConfig(rag_keywords=[], db_path=db_path),
        logging=cfg.LoggingConfig(level="INFO"),
        google=cfg.GoogleConfig(
            drive_folder_id="F", credentials_path=Path("c"), token_path=Path("t"),
            enabled=True,
            firestore_project_id="proj-1", firestore_database="default",
            firestore_collection="telegram_queue",
        ),
        channel=cfg.ChannelConfig(
            target=target, posts_per_day=5, window_start_hour=8, window_end_hour=24,
            jitter_minutes=5, enabled=channel_enabled,
        ),
    )


@pytest.fixture
def app_config(tmp_path, monkeypatch) -> cfg.AppConfig:
    app_cfg = _make_app_config(db_path=tmp_path / "messages_fts.db")
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)
    return app_cfg


# --- Hashtag helpers ----------------------------------------------------------

def test_country_hashtag_known_codes():
    assert cp._country_hashtag("br") == "#Бразилия"
    assert cp._country_hashtag("lk") == "#ШриЛанка"
    # 'any' falls back to the curated label
    assert cp._country_hashtag("any") == "#ЛайфхакиВПути"


def test_country_hashtag_unknown_falls_back_to_uppercase():
    assert cp._country_hashtag("zz") == "#ZZ"


def test_tag_hashtag():
    assert cp._tag_hashtag("Finance") == "#Finance"
    assert cp._tag_hashtag("") == ""


def test_channel_field_safe_strips_special_chars():
    assert cp._channel_field_safe("@luky_channel") == "luky_channel"
    assert cp._channel_field_safe("-1001234567890") == "1001234567890"
    assert cp._channel_field_safe("MixedCase123") == "mixedcase123"


# --- format_message -----------------------------------------------------------

def test_format_message_combines_content_and_hashtags():
    cand = cp.PostCandidate(
        doc_id="x", country="br", title="t",
        content="Это пример истории про Бразилию.",
        tag="Finance", created_at=None,
    )
    msg = cp.format_message(cand)
    assert "Это пример истории про Бразилию." in msg
    assert "#Бразилия" in msg
    assert "#Finance" in msg
    # Tags on a separate line for readability
    assert msg.endswith("#Бразилия #Finance")


# --- selection ----------------------------------------------------------------

def _fake_doc(doc_id, country, content, posted_to_channel=None, tag="General",
              created_at=None):
    """Build a minimal stand-in for Firestore DocumentSnapshot."""
    posted = {}
    if posted_to_channel:
        posted = {posted_to_channel: {"posted": True}}
    data = {
        "country": country, "content": content, "tag": tag, "title": doc_id,
        "createdAt": created_at, "postedTo": posted,
    }
    snap = MagicMock()
    snap.id = doc_id
    snap.to_dict.return_value = data
    return snap


def _fake_db(docs):
    """Build a fake Firestore client where stream() returns the given docs."""
    db = MagicMock()
    coll = MagicMock()
    query = MagicMock()
    query.stream.return_value = iter(docs)
    coll.order_by.return_value.limit.return_value = query
    db.collection.return_value = coll
    return db


def test_select_skips_already_posted():
    """
    Mini-pipeline: 3 docs in queue. Doc 1 already posted to luky_channel.
    Selection must skip it and return doc 2.
    """
    docs = [
        _fake_doc("d1", "br", "br msg", posted_to_channel="luky_channel"),
        _fake_doc("d2", "id", "id msg"),
        _fake_doc("d3", "lk", "lk msg"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(db, "telegram_queue", "@luky_channel")
    assert pick is not None
    assert pick.doc_id == "d2"
    assert pick.country == "id"


def test_select_country_rotation_prefers_different():
    """If last_country='br', selection prefers a non-br doc even if br is older."""
    docs = [
        _fake_doc("d1", "br", "old br"),
        _fake_doc("d2", "br", "newer br"),
        _fake_doc("d3", "id", "id msg"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(db, "telegram_queue", "@luky_channel",
                                    last_country="br")
    assert pick is not None
    assert pick.doc_id == "d3"
    assert pick.country == "id"


def test_select_country_rotation_falls_back_when_only_repeat_left():
    """If queue has only 'br' docs and last_country='br' — accept br anyway."""
    docs = [
        _fake_doc("d1", "br", "br1"),
        _fake_doc("d2", "br", "br2"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(db, "telegram_queue", "@luky_channel",
                                    last_country="br")
    assert pick is not None
    assert pick.doc_id == "d1"


def test_select_excludes_countries():
    docs = [
        _fake_doc("d1", "vn", "vn msg"),
        _fake_doc("d2", "br", "br msg"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(
        db, "telegram_queue", "@luky_channel",
        excluded_countries={"vn"},
    )
    assert pick is not None
    assert pick.doc_id == "d2"


def test_select_returns_none_when_all_posted():
    docs = [
        _fake_doc("d1", "br", "x", posted_to_channel="luky_channel"),
        _fake_doc("d2", "id", "y", posted_to_channel="luky_channel"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(db, "telegram_queue", "@luky_channel")
    assert pick is None


def test_select_skips_empty_content():
    docs = [
        _fake_doc("d1", "br", ""),       # empty content — skip
        _fake_doc("d2", "br", "   "),    # whitespace only — skip
        _fake_doc("d3", "id", "real content"),
    ]
    db = _fake_db(docs)
    pick = cp.select_next_candidate(db, "telegram_queue", "@luky_channel")
    assert pick is not None
    assert pick.doc_id == "d3"


# --- mark_posted --------------------------------------------------------------

def test_mark_posted_writes_correct_path(app_config):
    """Verify that postedTo.<channel-key>.* fields are written on update()."""
    db = MagicMock()
    coll = MagicMock()
    doc_ref = MagicMock()
    coll.document.return_value = doc_ref
    db.collection.return_value = coll

    cp.mark_posted(db, "telegram_queue", "doc-abc", "@luky_channel", "posted text")

    db.collection.assert_called_once_with("telegram_queue")
    coll.document.assert_called_once_with("doc-abc")
    doc_ref.update.assert_called_once()
    update_arg = doc_ref.update.call_args[0][0]
    assert update_arg["postedTo.luky_channel.posted"] is True
    assert update_arg["postedTo.luky_channel.text"] == "posted text"
    assert update_arg["postedTo.luky_channel.target"] == "@luky_channel"
    assert "posted_at" in update_arg["postedTo.luky_channel.posted_at"].__class__.__name__.lower() or \
           isinstance(update_arg["postedTo.luky_channel.posted_at"], dt.datetime)


# --- post_one end-to-end (with mocks) ----------------------------------------

@pytest.mark.asyncio
async def test_post_one_end_to_end(app_config):
    """
    Mini-pipeline:
      Setup: Firestore queue has 1 unposted doc.
      Action: post_one() with a mock bot_client.
      Expect:
        - bot_client.send_message called with formatted text and target
        - mark_posted called for the doc
        - returned country == doc's country
    """
    docs = [_fake_doc("d1", "br", "Бразильская история", tag="Travel")]
    fake_db = _fake_db(docs)
    bot_client = MagicMock()
    bot_client.send_message = AsyncMock()

    with patch.object(cp, "_build_firestore_client", return_value=fake_db), \
         patch.object(cp, "mark_posted") as mark_mock:
        result = await cp.post_one(bot_client, last_country=None)

    assert result == "br"
    bot_client.send_message.assert_awaited_once()
    args, kwargs = bot_client.send_message.call_args
    assert args[0] == "@luky_channel"
    sent_text = args[1]
    assert "Бразильская история" in sent_text
    assert "#Бразилия" in sent_text
    assert "#Travel" in sent_text
    assert kwargs.get("link_preview") is False
    mark_mock.assert_called_once()


@pytest.mark.asyncio
async def test_post_one_skip_when_disabled(tmp_path, monkeypatch):
    cfg_disabled = _make_app_config(db_path=tmp_path / "x.db", channel_enabled=False)
    monkeypatch.setattr(cfg, "_CONFIG", cfg_disabled, raising=False)
    bot_client = MagicMock()
    result = await cp.post_one(bot_client)
    assert result is None


@pytest.mark.asyncio
async def test_post_one_does_not_mark_when_send_fails(app_config):
    """If Telegram send fails, do NOT mark posted (we'll retry next slot)."""
    docs = [_fake_doc("d1", "br", "story", tag="Travel")]
    fake_db = _fake_db(docs)
    bot_client = MagicMock()
    bot_client.send_message = AsyncMock(side_effect=RuntimeError("network"))

    with patch.object(cp, "_build_firestore_client", return_value=fake_db), \
         patch.object(cp, "mark_posted") as mark_mock:
        result = await cp.post_one(bot_client)

    assert result is None
    mark_mock.assert_not_called()


# --- schedule slots -----------------------------------------------------------

def test_todays_slots_5_per_day_8_to_24(app_config):
    slots = cp._todays_slots(app_config)
    assert [t.strftime("%H:%M") for t in slots] == [
        "08:00", "11:12", "14:24", "17:36", "20:48"
    ]


def test_todays_slots_3_per_day_custom_window(app_config):
    app_config.channel.posts_per_day = 3
    app_config.channel.window_start_hour = 9
    app_config.channel.window_end_hour = 21
    slots = cp._todays_slots(app_config)
    assert [t.strftime("%H:%M") for t in slots] == ["09:00", "13:00", "17:00"]


def test_next_slot_at_picks_today_when_future_slot_exists(app_config):
    slots = [dt.time(8, 0), dt.time(14, 0), dt.time(20, 0)]
    now = dt.datetime(2026, 5, 5, 10, 0, 0)
    nxt = cp._next_slot_at(now, slots)
    assert nxt == dt.datetime(2026, 5, 5, 14, 0, 0)


def test_next_slot_at_rolls_to_tomorrow_after_last(app_config):
    slots = [dt.time(8, 0), dt.time(14, 0), dt.time(20, 0)]
    now = dt.datetime(2026, 5, 5, 22, 0, 0)
    nxt = cp._next_slot_at(now, slots)
    assert nxt == dt.datetime(2026, 5, 6, 8, 0, 0)
