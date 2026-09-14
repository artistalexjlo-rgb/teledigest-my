"""Сторожа экстрактора (15.09 — экстрактор через мозг `keybroker`).

Покрытие:
- нарезка файла на куски по границам строк;
- порядок моделей: следующая — только когда у текущей нет живых ключей или мозг вернул None;
- сайдкары: `.processed` при извлечении, `.empty` при нуле, ничего при провале куска;
- индекс паттерна сквозной по файлу (окно на кусок), повтор не задваивает;
- проход останавливается, когда у мозга нет живых ключей;
- junk-guard ai_lesson (детерминированный, без LLM).

Мозг сюда не зовём (у него свои сторожа в pseo/tract/) — подменяем `keybroker.call` /
`keybroker.any_alive` и `_persist_patterns`, проверяем СКЛЕЙКУ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from teledigest import extraction

# ── нарезка ────────────────────────────────────────────────────────────────────────────


def test_split_keeps_small_text_whole():
    assert extraction.split_content("a\nb\n", limit=100) == ["a\nb\n"]
    assert extraction.split_content("   \n", limit=100) == []


def test_split_cuts_on_line_boundaries_under_limit():
    text = "".join(f"line{i:02d}\n" for i in range(10))  # 7 симв. на строку
    chunks = extraction.split_content(text, limit=20)
    assert "".join(chunks) == text, "ничего не потеряно и не задвоено"
    assert all(len(c) <= 20 for c in chunks), [len(c) for c in chunks]
    assert all(c.endswith("\n") for c in chunks), "рез только по границе строки"


def test_split_keeps_an_oversized_line_as_its_own_chunk():
    text = "short\n" + "x" * 50 + "\nshort\n"
    chunks = extraction.split_content(text, limit=20)
    assert "".join(chunks) == text
    assert "x" * 50 + "\n" in chunks, "длинная строка — своим куском, не резана пополам"


# ── вызов мозга: порядок моделей ────────────────────────────────────────────────────────


def test_ask_uses_first_model_with_alive_keys(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(extraction.keybroker, "any_alive", lambda m, role: m != "m1")

    def fake_call(user, sysprompt, consumer, model, role, timeout, salvage):
        calls.append(model)
        assert consumer == extraction.CONSUMER and role == extraction.ROLE
        assert salvage == ("patterns", "title")
        return {"patterns": [{"title": "t"}]}

    monkeypatch.setattr(extraction.keybroker, "call", fake_call)
    out = extraction.ask("лог", models=["m1", "m2", "m3"])
    assert out == [{"title": "t"}]
    assert calls == ["m2"], "m1 без живых ключей пропущена, m3 не понадобилась"


def test_ask_falls_through_when_brain_returns_none(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(extraction.keybroker, "any_alive", lambda m, role: True)

    def fake_call(user, sysprompt, consumer, model, **kw):
        calls.append(model)
        return None if model == "m1" else {"patterns": []}

    monkeypatch.setattr(extraction.keybroker, "call", fake_call)
    assert extraction.ask("лог", models=["m1", "m2", "m3"]) == []
    assert calls == ["m1", "m2"], "пустой список — это ОТВЕТ, к m3 не идём"


def test_ask_returns_none_when_no_model_answers(monkeypatch):
    monkeypatch.setattr(extraction.keybroker, "any_alive", lambda m, role: True)
    monkeypatch.setattr(extraction.keybroker, "call", lambda *a, **k: None)
    assert extraction.ask("лог", models=["m1", "m2"]) is None


# ── файл: куски, индексы, сайдкары ───────────────────────────────────────────────────────


@pytest.fixture
def persisted(monkeypatch):
    """Подмена записи в базу: копим (file, idx_offset, patterns), отвечаем «все сохранены»."""
    seen: list[tuple[str, int, list]] = []

    def fake_persist(file_name, patterns, idx_offset=0):
        seen.append((file_name, idx_offset, patterns))
        return len(patterns), len(patterns)

    monkeypatch.setattr(extraction, "_persist_patterns", fake_persist)
    return seen


def _sample(tmp_path: Path, name="2026-09-13_gr_1.txt", text="[10:00] u/1: привет\n"):
    d = tmp_path / "gr"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text(text, encoding="utf-8")
    return f


def test_chunks_get_disjoint_index_windows(tmp_path, monkeypatch, persisted):
    f = _sample(
        tmp_path, text="".join(f"[10:0{i}] u/1: строка {i}\n" for i in range(6))
    )
    monkeypatch.setattr(extraction, "CHUNK_CHARS", 40)  # → несколько кусков
    monkeypatch.setattr(extraction, "ask", lambda chunk: [{"title": chunk[:5]}])

    saved, attempted, ok = extraction.process_file(f)

    assert ok and saved == attempted == len(persisted) >= 2
    offsets = [o for _, o, _ in persisted]
    assert offsets == [i * extraction._IDX_STRIDE for i in range(len(persisted))]


def test_processed_marker_when_patterns_extracted(tmp_path, monkeypatch, persisted):
    f = _sample(tmp_path)
    monkeypatch.setattr(extraction, "ask", lambda chunk: [{"title": "t"}])
    monkeypatch.setattr(extraction, "_any_model_alive", lambda: True)
    monkeypatch.setattr(extraction, "init_extraction_tables", lambda: None)

    assert extraction.run_extraction_pass(tmp_path) == (1, 1, 1)
    assert Path(str(f) + ".processed").exists()
    assert not Path(str(f) + ".empty").exists()


def test_empty_marker_when_answer_has_no_patterns(tmp_path, monkeypatch, persisted):
    """⛔ 15.09: пустой ответ раньше не помечался — 241 файл «сообщений нет» крутился в
    очереди каждый проход, съедая вызов и ключ."""
    f = _sample(tmp_path)
    monkeypatch.setattr(extraction, "ask", lambda chunk: [])
    monkeypatch.setattr(extraction, "_any_model_alive", lambda: True)
    monkeypatch.setattr(extraction, "init_extraction_tables", lambda: None)

    assert extraction.run_extraction_pass(tmp_path) == (1, 0, 0)
    assert Path(str(f) + ".empty").exists()
    assert not Path(str(f) + ".processed").exists()
    # второй проход файл не берёт
    assert extraction.run_extraction_pass(tmp_path) == (0, 0, 0)


def test_no_marker_when_a_chunk_fails(tmp_path, monkeypatch, persisted):
    """Провал куска — файл вернётся; извлечённое из удачных кусков уже записано."""
    f = _sample(
        tmp_path, text="".join(f"[10:0{i}] u/1: строка {i}\n" for i in range(6))
    )
    monkeypatch.setattr(extraction, "CHUNK_CHARS", 40)
    answers = iter([[{"title": "a"}], None, [{"title": "c"}]])
    monkeypatch.setattr(extraction, "ask", lambda chunk: next(answers, None))
    monkeypatch.setattr(extraction, "_any_model_alive", lambda: True)
    monkeypatch.setattr(extraction, "init_extraction_tables", lambda: None)

    files, saved, attempted = extraction.run_extraction_pass(tmp_path)

    assert files == 1 and saved >= 1
    assert not Path(str(f) + ".processed").exists()
    assert not Path(str(f) + ".empty").exists()
    assert extraction._is_done(f) is False


def test_pass_stops_when_brain_has_no_alive_keys(tmp_path, monkeypatch, persisted):
    _sample(tmp_path, "a.txt")
    _sample(tmp_path, "b.txt")
    monkeypatch.setattr(extraction, "ask", lambda chunk: [{"title": "t"}])
    monkeypatch.setattr(extraction, "init_extraction_tables", lambda: None)
    alive = iter([True, False])
    monkeypatch.setattr(extraction, "_any_model_alive", lambda: next(alive))

    files, *_ = extraction.run_extraction_pass(tmp_path)
    assert files == 1, "второй файл не тронут — у мозга нет живых ключей"


def test_own_rotator_and_quota_are_gone():
    """Один учёт ключей — мозг. Свой ротатор/квота не должны вернуться (15.09)."""
    for name in ("iter_model_key_pairs", "_gemini_generate_json", "_MINING_MODELS"):
        assert not hasattr(extraction, name), name
    assert (
        extraction.CONSUMER in extraction.keybroker.CAPS
    ), "имя `extract` в реестре мозга"


# ── junk-guard ───────────────────────────────────────────────────────────────────────────


def test_is_junk_ai_lesson_catches_inquiry_narration():
    """Guard ловит пересказы вопросов/пустоты/листингов, но не трогает факты."""
    junk = [
        "Inquiry about the best pharmacy in Nha Trang for medication.",
        "User is asking how to find Russian-speaking guides in Vietnam.",
        "A user inquired if Visa cards can be used for payments in Vietnam.",
        "User is looking for someone traveling to Bali to deliver a package.",
        "Clarification needed on whether e-tickets require entry at a set time.",
        "Information is not explicitly provided in the log; should be researched.",
        "A room is available for rent in Vienna from July.",
        "For Thailand real estate, consult with Pavel via their Telegram channel.",
    ]
    for t in junk:
        assert extraction.is_junk_ai_lesson(t), t

    good = [
        "CPF can be issued online via Receita Federal; takes 10-30 minutes.",
        "ATMs in Bali (BNI, BCA) usually charge no commission; limit ~3M IDR.",
        "In France, recurring payments must be set up via automatic bank debit.",
        "Mauritius: USD is more advantageous to exchange than CNY at the airport.",
    ]
    for t in good:
        assert not extraction.is_junk_ai_lesson(t), t

    assert extraction.is_junk_ai_lesson(None) is False
    assert extraction.is_junk_ai_lesson("") is False
