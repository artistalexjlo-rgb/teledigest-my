"""Сторож политики «только IPv4» и честности счётчика квоты.

⛔ ПОВОД (19.08.2026, замер на боевом VPS, ключи не тронуты — curl без ключа):
    IPv4 172.217.113.4        → ответ за 0.18 с
    IPv6 2001:4860:4842:400:: → 8 с молчания, код 000
IPv6 к `generativelanguage.googleapis.com` с этого сервера — чёрная дыра. Резолвер отдаёт AAAA
первым, `requests` идёт по нему и висит до таймаута: в логе 15 повисаний по 60 с за два часа.
В pseo та же болезнь вылечена в `builder/keybroker.py` ещё в июле — во второе дерево починку
не перенесли, поэтому здесь правило живёт ОДНИМ местом и проверяется машиной.
"""

from __future__ import annotations

import socket
import sqlite3
from pathlib import Path

import pytest

from teledigest import ipv4_only

_V6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:4860:4842:400::", 443, 0, 0))
_V4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("172.217.113.4", 443))


@pytest.fixture
def fresh_filter(monkeypatch):
    """Фильтр уже стоит (его ставят вызывающие на импорте) — для проверки ставим заново."""
    saved = socket.getaddrinfo
    monkeypatch.setattr(ipv4_only, "_applied", False)
    yield
    socket.getaddrinfo = saved


def test_ipv6_records_are_dropped(fresh_filter, monkeypatch):
    """Главное: после фильтра до сокета доходят только IPv4-адреса."""
    monkeypatch.setattr(ipv4_only, "_orig_getaddrinfo", lambda *a, **k: [_V6, _V4])
    assert ipv4_only.force_ipv4() is True
    got = socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    assert [r[0] for r in got] == [socket.AF_INET], got


def test_no_ipv4_means_no_breakage(fresh_filter, monkeypatch):
    """⛔ Отличие от эталона в pseo: если A-записей НЕТ, отдаём как было.

    Иначе на IPv6-only хосте фильтр не «снял предпочтение», а сломал разрешение имён целиком —
    то есть починка одного сервера уронила бы любой другой.
    """
    monkeypatch.setattr(ipv4_only, "_orig_getaddrinfo", lambda *a, **k: [_V6])
    ipv4_only.force_ipv4()
    assert socket.getaddrinfo("example.invalid", 443) == [_V6]


def test_applying_twice_does_not_stack(fresh_filter, monkeypatch):
    """Идемпотентность: два вызывающих зовут владельца, обёртка ставится один раз."""
    monkeypatch.setattr(ipv4_only, "_orig_getaddrinfo", lambda *a, **k: [_V6, _V4])
    assert ipv4_only.force_ipv4() is True
    first = socket.getaddrinfo
    assert ipv4_only.force_ipv4() is False
    assert socket.getaddrinfo is first


@pytest.mark.parametrize("name", ["teledigest.extraction", "teledigest.gemini_brain"])
def test_gemini_caller_applies_the_policy(name, monkeypatch):
    """⛔ Сторож ПРОВОДКИ: модуль-владелец сам по себе ничего не лечит.

    Оба вызывающих Gemini обязаны позвать владельца на импорте, иначе фильтр — мёртвый код, а
    запросы по-прежнему уходят в IPv6.

    ⛔ ПЕРВАЯ ВЕРСИЯ ЭТОГО СТОРОЖА БЫЛА ФАЛЬШИВОЙ: она искала подстроку `force_ipv4()` в файле,
    а закомментированный `# force_ipv4()` её содержит — мутация проходила зелёной. И проверка
    `_applied is True` тоже ничего не значила: второй вызывающий ставил флаг за первого.
    Поэтому проверяем ИСПОЛНЕНИЕМ: сбрасываем флаг и перезагружаем ИМЕННО ЭТОТ модуль.
    """
    import importlib

    mod = importlib.import_module(name)
    lines = Path(mod.__file__).read_text(encoding="utf-8").splitlines()
    assert any(x.strip() == "force_ipv4()" for x in lines), f"{name} не зовёт владельца"

    saved = socket.getaddrinfo
    monkeypatch.setattr(ipv4_only, "_applied", False)
    try:
        importlib.reload(mod)
        assert ipv4_only._applied is True, f"{name} не поставил фильтр при импорте"
    finally:
        socket.getaddrinfo = saved


# ── СЧЁТЧИК КВОТЫ ────────────────────────────────────────────────────────────────────
# 15.09: учёт ключей — в мозге (`keybroker.report`), свой счётчик экстрактора снесён.
# Правило то же: запрос, который НЕ УШЁЛ (транспорт, status -1), квоту Google не трогает и
# в счётчик не попадает; 19.08 резолверы легли на десять минут, и RPD списывался за каждую
# непосланную попытку — счётчик врал там, где по нему решают «ключи кончились».


@pytest.fixture
def broker_db(tmp_path: Path, monkeypatch):
    from teledigest import keybroker

    monkeypatch.setattr(keybroker, "DB", str(tmp_path / "kb.db"))
    monkeypatch.setattr(keybroker, "_SCHEMA_OK", False)
    return keybroker


def _usage(kb, key: str, model: str) -> tuple[int, int]:
    c = sqlite3.connect(kb.DB)
    row = c.execute(
        "SELECT count, banned FROM usage WHERE key_hash=? AND model=?",
        (kb._kh(key), model),
    ).fetchone()
    c.close()
    return (row[0], row[1]) if row else (0, 0)


def test_transport_failure_does_not_spend_quota(broker_db):
    broker_db.report("extract", "AIza-test-key", "m1", -1)
    assert _usage(broker_db, "AIza-test-key", "m1") == (0, 0)


def test_real_answer_spends_quota(broker_db):
    broker_db.report("extract", "AIza-test-key", "m1", 200)
    assert _usage(broker_db, "AIza-test-key", "m1")[0] == 1


def test_first_429_is_a_strike_not_a_day_ban(broker_db):
    """⛔ Старый экстрактор банил пару на сутки с ПЕРВОГО 429 (15.09: 7 ключей из 12 за
    полтора часа). Мозг: первый 429 — метка, ключ остаётся в очереди; бан — только после
    всей лестницы отдыха."""
    broker_db.report("extract", "AIza-test-key", "m1", 429)
    count, banned = _usage(broker_db, "AIza-test-key", "m1")
    assert banned == 0 and count == 0
    c = sqlite3.connect(broker_db.DB)
    struck = c.execute(
        "SELECT struck FROM key_clock WHERE key_hash=?",
        (broker_db._kh("AIza-test-key"),),
    ).fetchone()[0]
    c.close()
    assert struck == 1
