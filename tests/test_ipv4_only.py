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

from teledigest import extraction, extraction_db, ipv4_only

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


@pytest.fixture
def quota_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "quota.db"
    monkeypatch.setattr(
        extraction_db, "get_db_connection", lambda: sqlite3.connect(str(db_path))
    )
    extraction_db.init_extraction_tables()
    yield db_path


def _run_one(monkeypatch, tmp_path, status, resp=None):
    """Прогнать process_file с подменённым HTTP-вызовом. Рты не зовём."""
    f = tmp_path / "2026-08-19_br_test.txt"
    f.write_text("живой текст лога про Бразилию", encoding="utf-8")
    monkeypatch.setattr(extraction, "_RETRY_DELAYS_S", [])  # без пауз в тесте
    monkeypatch.setattr(
        extraction, "_gemini_generate_json", lambda *a, **k: (resp, status)
    )
    rotator = iter([("m1", "AIza-test-key")] * 4)
    extraction.process_file(f, rotator)
    kh = extraction_db._key_hash("AIza-test-key")
    return extraction_db.quota_state(kh, "m1")


def test_transport_failure_does_not_spend_quota(quota_db, tmp_path, monkeypatch):
    """⛔ Запрос НЕ УШЁЛ (status 0) → квота Google не тронута, счётчик обязан молчать.

    Так и легло 19.08: резолверы провайдера отвалились на десять минут, а RPD списывался за
    каждую непосланную попытку. Счётчик врал там, где по нему решают «ключи кончились».
    """
    count, banned = _run_one(monkeypatch, tmp_path, status=0)
    assert count == 0, f"списали {count} попыток, не отправив ни одного запроса"
    assert not banned


def test_real_http_answer_spends_quota(quota_db, tmp_path, monkeypatch):
    """Обратная сторона: реальный ответ Google (даже 500) попытками считается."""
    count, _banned = _run_one(monkeypatch, tmp_path, status=500)
    assert count == 1, count


def test_429_spends_quota_and_bans_the_pair(quota_db, tmp_path, monkeypatch):
    """429 — это превышение: и попытка, и бан пары до UTC-полуночи. Не сломать заодно."""
    count, banned = _run_one(monkeypatch, tmp_path, status=429)
    assert count >= 1 and banned, (count, banned)
