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
async def test_embed_scheduler_fires_at_pt_reset(monkeypatch):
    at_trigger = real_dt.datetime(
        2026, 5, 30, sched._EMBED_HOUR, sched._EMBED_MINUTE, tzinfo=sched._EMBED_TZ
    )
    pass_mock, to_thread = _patch_common(monkeypatch, at_trigger)

    with pytest.raises(_Break):
        await sched.embed_scheduler()

    to_thread.assert_awaited_once()
    # run_embed_pass is what gets handed to to_thread.
    assert to_thread.await_args.args[0] is pass_mock


@pytest.mark.asyncio
async def test_embed_scheduler_skips_outside_window(monkeypatch):
    off_window = real_dt.datetime(2026, 5, 30, 5, 0, tzinfo=sched._EMBED_TZ)
    _pass_mock, to_thread = _patch_common(monkeypatch, off_window)

    with pytest.raises(_Break):
        await sched.embed_scheduler()

    to_thread.assert_not_awaited()
