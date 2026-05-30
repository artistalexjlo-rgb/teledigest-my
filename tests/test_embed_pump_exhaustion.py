"""Регрессия: embed_pump не должен грайндить pending в ложные фейлы и зависать,
когда дневная квота всех ключей исчерпана.

Инцидент 2026-05-26: in-memory RPD-счётчик не сбрасывался между UTC-днями,
все ключи застряли capped, _pump_wiki в while True гнал каждую pending-строку
в embedding=None → mark_embed_failed (к 5 фейлам → навсегда из очереди), не
выходил, и заморозил весь ночной scheduler. Тест фиксирует исправленное
поведение: исчерпаны ключи → пасс выходит, строки остаются pending, fail=0.
"""

from __future__ import annotations

from teledigest import embed_pump, gemini_brain


def _row(i: int) -> dict:
    return {
        "id": f"wikivoyage:xx:Page:{i}",
        "country": "xx",
        "title": f"Page {i}",
        "tag": "Travel",
        "instruction": f"do thing {i}",
    }


def test_pump_wiki_stops_clean_when_keys_exhausted(monkeypatch):
    rows = [_row(i) for i in range(3)]

    # fetch_pending_wiki всегда отдаёт те же строки (имитируем, что они НЕ
    # маркируются embedded и крутятся в очереди) — если фикс сломан, цикл
    # зациклится; с фиксом он выходит после первого батча.
    monkeypatch.setattr(embed_pump, "fetch_pending_wiki", lambda limit=50: list(rows))

    failed_calls: list = []
    monkeypatch.setattr(
        embed_pump,
        "mark_embed_failed",
        lambda *a, **k: failed_calls.append(a),
    )

    # Все ключи исчерпаны → embed возвращает None на каждый текст.
    monkeypatch.setattr(
        gemini_brain,
        "compute_document_embeddings_v2",
        lambda texts, **k: [None] * len(texts),
    )
    monkeypatch.setattr(gemini_brain, "keys_all_exhausted", lambda *a, **k: True)

    import teledigest.qdrant_db as qd

    monkeypatch.setattr(qd, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(qd, "upsert_points_batch", lambda *a, **k: None)

    embedded, failed = embed_pump._pump_wiki(batch_size=50)

    assert embedded == 0
    # Ключевое: НИ одной строки не помечено failed (capped != failure).
    assert failed == 0
    assert failed_calls == []
