"""Tests for teledigest.scheduler — covers the digest scheduling loop.

The scheduler is an infinite ``while True`` loop, so every test drives it
by replacing ``asyncio.sleep`` with an async stub that raises ``_BreakLoop``
after a configurable number of calls, then catches that sentinel exception.

``datetime.datetime`` in the scheduler's own namespace is patched via
``patch.object(sched, "dt")`` so that ``dt.datetime.now(tz)`` returns a
fixed ``datetime`` object without touching the global ``datetime`` module.
All dependent I/O functions (LLM, DB, Telegram, Telegraph) are patched to
avoid real network calls.
"""

from __future__ import annotations

import datetime as real_dt
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from teledigest import config as cfg
from teledigest import scheduler as sched
from teledigest.db import Message

# ---------------------------------------------------------------------------
# Sentinel used to break the infinite loop
# ---------------------------------------------------------------------------


class _BreakLoop(Exception):
    """Raised by the fake asyncio.sleep to exit the scheduler loop."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_config(
    *,
    summary_hour: int = 21,
    summary_minute: int = 0,
    summary_brief: bool = False,
    time_zone: str = "UTC",
    max_messages: int = 100,
) -> cfg.AppConfig:
    return cfg.AppConfig(
        telegram=cfg.TelegramConfig(
            api_id=1, api_hash="hash", bot_token="token", sessions_dir=Path("data")
        ),
        bot=cfg.BotConfig(
            channels=["@c1"],
            summary_target="@digest",
            time_zone=time_zone,
            summary_hour=summary_hour,
            summary_minute=summary_minute,
            summary_brief=summary_brief,
        ),
        llm=cfg.LLMConfig(
            model="gpt-4.1",
            api_key="sk-test",
            system_prompt="",
            user_prompt="",
            max_messages=max_messages,
        ),
        storage=cfg.StorageConfig(rag_keywords=[], db_path=Path(":memory:")),
        logging=cfg.LoggingConfig(level="WARNING"),
    )


@pytest.fixture
def app_config(monkeypatch) -> cfg.AppConfig:
    app_cfg = _make_app_config()
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)
    return app_cfg


def _fixed_now(hour: int = 21, minute: int = 0) -> real_dt.datetime:
    """Return a UTC datetime that matches the default summary_hour:summary_minute."""
    return real_dt.datetime(2024, 6, 1, hour, minute, 0, tzinfo=real_dt.timezone.utc)


async def _drive(max_sleeps: int = 1) -> list[float]:
    """Run the scheduler until *max_sleeps* asyncio.sleep calls have occurred.

    Returns the list of sleep durations so callers can assert on them.
    """
    sleep_durations: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_durations.append(seconds)
        if len(sleep_durations) >= max_sleeps:
            raise _BreakLoop

    with patch("asyncio.sleep", new=fake_sleep):
        with pytest.raises(_BreakLoop):
            await sched.summary_scheduler()

    return sleep_durations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_scheduler_sends_digest_at_scheduled_time(app_config):
    """When the clock matches summary_hour:summary_minute the scheduler must
    call llm_summarize, then send_message_long with the result."""
    messages = [Message(channel="@c1", text="latest war news")]
    send_mock = AsyncMock()

    with (
        patch.object(sched, "dt") as mock_dt,
        patch.object(sched, "get_relevant_messages_last_24h", return_value=messages),
        patch.object(sched, "llm_summarize", return_value="AI Summary"),
        patch.object(sched, "send_message_long", send_mock),
        patch.object(sched, "get_bot_client", return_value=MagicMock()),
    ):
        mock_dt.datetime.now.return_value = _fixed_now(21, 0)
        await _drive(max_sleeps=1)

    send_mock.assert_awaited_once()
    _bot_client, target, outgoing, *_ = send_mock.call_args[0]
    assert target == "@digest"
    assert "AI Summary" in outgoing


@pytest.mark.asyncio
async def test_summary_scheduler_deduplicates_same_day(app_config):
    """A second trigger within the same calendar day must not call
    send_message_long again; the deduplication sleep(60) fires instead."""
    messages = [Message(channel="@c1", text="war update")]
    send_mock = AsyncMock()

    with (
        patch.object(sched, "dt") as mock_dt,
        patch.object(sched, "get_relevant_messages_last_24h", return_value=messages),
        patch.object(sched, "llm_summarize", return_value="Summary"),
        patch.object(sched, "send_message_long", send_mock),
        patch.object(sched, "get_bot_client", return_value=MagicMock()),
    ):
        # Both iterations return the same time so today == last_run_for on the second pass.
        mock_dt.datetime.now.return_value = _fixed_now(21, 0)
        # First sleep(65) after sending + second sleep(60) from dedup guard = 2 sleeps.
        sleep_durations = await _drive(max_sleeps=2)

    # The digest is sent exactly once despite two trigger matches.
    assert send_mock.await_count == 1
    # The second sleep must be the short dedup pause.
    assert sleep_durations == [65, 60]


@pytest.mark.asyncio
async def test_summary_scheduler_sends_notice_when_no_messages(app_config):
    """When no messages are available the scheduler must send a human-readable
    notice without calling llm_summarize."""
    send_mock = AsyncMock()

    with (
        patch.object(sched, "dt") as mock_dt,
        patch.object(sched, "get_relevant_messages_last_24h", return_value=[]),
        patch.object(sched, "llm_summarize") as mock_llm,
        patch.object(sched, "send_message_long", send_mock),
        patch.object(sched, "get_bot_client", return_value=MagicMock()),
    ):
        mock_dt.datetime.now.return_value = _fixed_now(21, 0)
        await _drive(max_sleeps=1)

    mock_llm.assert_not_called()
    send_mock.assert_awaited_once()
    _bot, _target, outgoing, *_ = send_mock.call_args[0]
    assert "no messages" in outgoing.lower()


@pytest.mark.asyncio
async def test_summary_scheduler_uses_brief_path_when_configured(monkeypatch):
    """When summary_brief=True and messages exist, the scheduler must post to
    Telegraph, call llm_summarize_brief, and send the combined output."""
    app_cfg = _make_app_config(summary_brief=True)
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)

    messages = [Message(channel="@c1", text="war news")]
    send_mock = AsyncMock()

    with (
        patch.object(sched, "dt") as mock_dt,
        patch.object(sched, "get_relevant_messages_last_24h", return_value=messages),
        patch.object(sched, "llm_summarize", return_value="Full digest"),
        patch.object(
            sched,
            "post_to_telegraph",
            return_value="https://telegra.ph/test-01-01",
        ),
        patch.object(sched, "llm_summarize_brief", return_value="Brief!"),
        patch.object(sched, "send_message_long", send_mock),
        patch.object(sched, "get_bot_client", return_value=MagicMock()),
    ):
        mock_dt.datetime.now.return_value = _fixed_now(21, 0)
        await _drive(max_sleeps=1)

    send_mock.assert_awaited_once()
    _bot, _target, outgoing, *_ = send_mock.call_args[0]
    assert "Brief!" in outgoing
    assert "telegra.ph" in outgoing


@pytest.mark.asyncio
async def test_summary_scheduler_handles_rpc_error_without_propagating(app_config):
    """An RPCError raised by send_message_long must be caught and logged; the
    scheduler must still update last_run_for and sleep normally afterwards."""

    # Replace the module-level RPCError name with a plain exception so we can
    # raise it without constructing a real Telethon RPC object.
    class _FakeRPCError(Exception):
        pass

    messages = [Message(channel="@c1", text="news")]
    send_mock = AsyncMock(side_effect=_FakeRPCError("timed out"))

    with (
        patch.object(sched, "dt") as mock_dt,
        patch.object(sched, "RPCError", _FakeRPCError),
        patch.object(sched, "get_relevant_messages_last_24h", return_value=messages),
        patch.object(sched, "llm_summarize", return_value="Summary"),
        patch.object(sched, "send_message_long", send_mock),
        patch.object(sched, "get_bot_client", return_value=MagicMock()),
    ):
        mock_dt.datetime.now.return_value = _fixed_now(21, 0)
        # sleep(65) still fires after the error — that is what we break on.
        sleep_durations = await _drive(max_sleeps=1)

    # The error was swallowed; the post-error sleep(65) still ran.
    assert sleep_durations == [65]


# ---------------------------------------------------------------------------
# End-to-end mini-pipeline: country-scoped daily understanding
# (rule: per-country knowledge must be built ONLY from per-country messages)
# ---------------------------------------------------------------------------


@pytest.fixture
def realdb_app_config(tmp_path, monkeypatch) -> cfg.AppConfig:
    """A config wired to a real (file) SQLite DB so we can exercise db queries."""
    import sqlite3 as _sqlite
    db_path = tmp_path / "messages_fts.db"
    app_cfg = _make_app_config()
    app_cfg.storage.db_path = db_path
    monkeypatch.setattr(cfg, "_CONFIG", app_cfg, raising=False)

    # Init schema like production does
    from teledigest import db as _db
    from teledigest import sources_db as _sdb
    _db.init_db()
    _sdb.init_sources_table()

    # Seed sources: br (with digest_target) + lk (without)
    conn = _sqlite.connect(db_path)
    conn.executemany(
        "INSERT INTO sources (country, url, name, language, added_at, chat_id, digest_target) "
        "VALUES (?, ?, ?, 'ru', '2026-04-24T00:00:00', ?, ?)",
        [
            ("br", "@Brazil_ChatForum", "Brazil Chat Forum", -1001221994108, "@Digest_Br"),
            ("lk", "https://t.me/+sri_lanka", "https://t.me/+sri_lanka", -1001605996131, None),
        ],
    )
    conn.commit()
    conn.close()
    return app_cfg


def test_daily_understanding_only_uses_messages_from_its_country(realdb_app_config):
    """
    End-to-end mini-pipeline scenario:
      Setup: 2 countries — br (with digest_target) and lk (without).
      Insert 3 messages from a br-channel and 3 from an lk-channel.
      Run _run_daily_understanding(day, 'br') and (day, 'lk').
      Expectation:
        - The artifact built for 'br' sees ONLY br-channel messages.
        - The artifact built for 'lk' sees ONLY lk-channel messages.

    Catches BUG #2 (claims from all countries mixed into each country's
    knowledge) which was present before this PR.
    """
    from teledigest import db as _db
    from teledigest import scheduler as _sched

    day = real_dt.date(2026, 4, 28)
    base = real_dt.datetime.combine(day, real_dt.time(12, 0), real_dt.timezone.utc)

    # br-channel messages — channel name matches the URL handle of the br source
    for i in range(3):
        _db.save_message(
            f"br_{i}", "Brazil_ChatForum", base + real_dt.timedelta(minutes=i),
            f"br message about price reais {i}", country="br",
        )
    # lk-channel messages — channel value is the numeric chat_id (invite link)
    for i in range(3):
        _db.save_message(
            f"lk_{i}", "-1001605996131", base + real_dt.timedelta(minutes=i),
            f"lk message about visa rupees {i}", country="lk",
        )

    captured: dict[str, list[str] | None] = {}

    def fake_build(day_arg, channels):
        # Capture which channels were requested + manually filter messages
        # exactly the way real _fetch_day_messages would.
        captured.setdefault("calls", []).append(list(channels) if channels else None)
        # Pull from the real DB, filtered by channels just like prod.
        import sqlite3
        conn = sqlite3.connect(realdb_app_config.storage.db_path)
        rows = conn.execute(
            "SELECT channel, text FROM messages WHERE substr(date,1,10) = ?",
            (day_arg.isoformat(),),
        ).fetchall()
        conn.close()
        if channels is not None:
            rows = [r for r in rows if r[0] in channels]
        return {
            "day": day_arg.isoformat(),
            "messages_count": len(rows),
            "chains_count": 0,
            "spans_count": 0,
            "claims_count": 0,
            "claims": [],
            "_observed_channels": [r[0] for r in rows],
        }

    with patch.object(_sched, "build_daily_artifact", side_effect=fake_build):
        _sched._run_daily_understanding(day, "br")
        _sched._run_daily_understanding(day, "lk")

    # 1. Every call to build_daily_artifact MUST receive a non-None channels list.
    #    A None here means we regressed — daily understanding fell back to
    #    "all channels" (the BUG #2 behaviour).
    assert all(c is not None for c in captured["calls"]), (
        "build_daily_artifact called without channel scope — BUG #2 regressed"
    )

    # 2. The br call must include the br URL handle and NOT the lk chat_id.
    br_channels = captured["calls"][0]
    assert "Brazil_ChatForum" in br_channels
    assert "-1001605996131" not in br_channels

    # 3. The lk call must include the lk chat_id and NOT the br URL handle.
    lk_channels = captured["calls"][1]
    assert "-1001605996131" in lk_channels
    assert "Brazil_ChatForum" not in lk_channels


def test_daily_understanding_skips_country_with_no_sources(realdb_app_config):
    """If a country has no rows in `sources`, daily understanding logs a warning
    and returns 0 — no fallback to all-messages."""
    from teledigest import scheduler as _sched
    day = real_dt.date(2026, 4, 28)

    with patch.object(_sched, "build_daily_artifact") as build_mock:
        loaded = _sched._run_daily_understanding(day, "xx")  # not in sources

    assert loaded == 0
    build_mock.assert_not_called()
