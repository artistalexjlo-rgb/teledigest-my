"""Tests for teledigest.scheduler.embed_scheduler — the independent,
Pacific-quota-aligned embed pass.

The loop is infinite and wraps its body in `except Exception` (resilience), so
we break out with a BaseException sentinel that the broad handler won't swallow.
`dt` and `asyncio` are patched in the scheduler's namespace to drive a fixed
Pacific time and to intercept the embed call without real I/O.
"""

from __future__ import annotations

import datetime as real_dt
from unittest.mock import AsyncMock, MagicMock

import pytest

from teledigest import embed_pump as ep
from teledigest import scheduler as sched


class _Break(BaseException):
    """Sentinel to exit the infinite loop (not caught by `except Exception`)."""


def _patch_common(monkeypatch, pt_dt: real_dt.datetime):
    fake_dt = MagicMock()
    fake_dt.datetime.now.return_value = pt_dt
    monkeypatch.setattr(sched, "dt", fake_dt)

    pass_mock = MagicMock(return_value={"wiki": {"embedded": 10}})
    monkeypatch.setattr(ep, "run_embed_pass", pass_mock)

    to_thread = AsyncMock(return_value={"wiki": {"embedded": 10}})
    monkeypatch.setattr(sched.asyncio, "to_thread", to_thread)

    async def fake_sleep(_):
        raise _Break

    monkeypatch.setattr(sched.asyncio, "sleep", fake_sleep)
    return pass_mock, to_thread


@pytest.mark.asyncio
async def test_embed_scheduler_fires_on_start_any_time(monkeypatch):
    # Catch-up: fires immediately on start at ANY time of the Pacific day
    # (mid-day after a redeploy), not only at 00:30.
    mid_day = real_dt.datetime(2026, 6, 8, 13, 0, tzinfo=sched._EMBED_TZ)
    pass_mock, to_thread = _patch_common(monkeypatch, mid_day)

    with pytest.raises(_Break):
        await sched.embed_scheduler()

    to_thread.assert_awaited_once()
    assert to_thread.await_args.args[0] is pass_mock


@pytest.mark.asyncio
async def test_embed_scheduler_fires_once_per_day(monkeypatch):
    # After the day's pass runs, it must NOT re-fire the same Pacific day.
    same_day = real_dt.datetime(2026, 6, 8, 13, 0, tzinfo=sched._EMBED_TZ)
    fake_dt = MagicMock()
    fake_dt.datetime.now.return_value = same_day
    monkeypatch.setattr(sched, "dt", fake_dt)

    pass_mock = MagicMock(return_value={"wiki": {"embedded": 10}})
    monkeypatch.setattr(ep, "run_embed_pass", pass_mock)
    to_thread = AsyncMock(return_value={"wiki": {"embedded": 10}})
    monkeypatch.setattr(sched.asyncio, "to_thread", to_thread)

    calls = {"n": 0}

    async def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] >= 2:  # let iteration 1 fire+sleep, break on iteration 2
            raise _Break

    monkeypatch.setattr(sched.asyncio, "sleep", fake_sleep)

    with pytest.raises(_Break):
        await sched.embed_scheduler()

    # Fired exactly once despite two loop iterations on the same PT-day.
    to_thread.assert_awaited_once()
