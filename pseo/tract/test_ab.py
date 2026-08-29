# -*- coding: utf-8 -*-
"""Сторож звена 4: проход А называет, проход Б только выбирает из закрытого списка.

Три обещания, ради которых звено переписано с одного вызова на два (PLAN.md, 23-24.08):

1. **список закрыт** — имя, которого нет в списке А, совету не достаётся. Иначе корзины
   плодятся: 22.08 один вызов на тему дал 33 разные подтемы в одной странице;
2. **не подошло — в остаток**, а не «своя корзина из подтемы». Иначе каждая муха заводит
   страницу сама себе;
3. **рту идут номера, не id** — сквозное правило: настоящий 24-символьный хеш роту не
   показывается ни в метках, ни в советах, ни в именах.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tract  # noqa: E402


def _tags(tmp_path, rows):
    os.makedirs(tmp_path / "tests" / "tags", exist_ok=True)
    with open(tmp_path / "tests" / "tags" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)


def _row(i, podtema, tema="visa"):
    """Запись разметки: id, тема, подтема, счётчик. Текста в тегах НЕТ — он в базе."""
    return {"id": f"hash-{i}", "theme": tema, "subtheme": podtema, "n": 1}


def _texts(monkeypatch, rows):
    """Сторожу база не нужна — нужен текст по id, поэтому чтение корпуса подменяется."""
    # ⛔ Текст НЕ содержит id: иначе проверка «настоящий хеш роту не уходит» была бы
    # ложной — хеш приезжал бы в запрос внутри самого текста.
    pairs = [(r["id"], f"advice number {j}") for j, r in enumerate(rows)]
    monkeypatch.setattr(tract, "load_flies", lambda geo: list(pairs))


def test_second_run_does_not_touch_old_names_or_reopen_pass_a(tmp_path, monkeypatch):
    """Новые мухи в теме не трогают уже присвоенные имена и не зовут проход А заново.

    ⛔ 28.08, юзер поймал риск: рот не детерминирован, и повторный вызов проход А на
    теме с новыми мухами мог выдать ДРУГОЙ список имён — тогда проход Б переприсвоил бы
    имена ВСЕМ советам темы (не только новым), уводя адреса уже опубликованных страниц.
    Список имён темы теперь заморожен, как только у темы появился хоть один именованный
    совет; проход Б трогает только `not r.get("name")`.
    """
    monkeypatch.chdir(tmp_path)
    rows = [_row(i, f"label {i}") for i in range(3)]
    _tags(tmp_path, rows)
    _texts(monkeypatch, rows)

    def fake_call_1(user, sysprompt, **kw):
        if kw["consumer"] == "canon":
            return {"names": ["visa documents"]}
        adv = json.loads(user)["advices"]
        return {"map": {k: "1" for k in adv}}

    monkeypatch.setattr(tract, "call", fake_call_1)
    tract.summarize("gr")

    tagged = json.load(open(tmp_path / "tests" / "tags" / "gr.json", encoding="utf-8"))
    assert all(r["name"] == "visa documents" for r in tagged), tagged

    # ── новые мухи в ту же тему — дописываем к УЖЕ ИМЕНОВАННЫМ, не к исходным `rows`
    new_rows = [_row(i, f"label {i}") for i in range(3, 5)]
    rows2 = tagged + new_rows
    _tags(tmp_path, rows2)
    _texts(monkeypatch, rows2)

    seen2 = {}

    def fake_call_2(user, sysprompt, **kw):
        seen2[kw["consumer"]] = json.loads(user)
        if kw["consumer"] == "canon":
            # Если проход А реально позвался — рот "передумал" бы имя. Тест обязан
            # упасть на следующей проверке, если это случится, а не молча пройти.
            return {"names": ["completely different name"]}
        adv = json.loads(user)["advices"]
        return {"map": {k: "1" for k in adv}}

    monkeypatch.setattr(tract, "call", fake_call_2)
    tract.summarize("gr")

    assert "canon" not in seen2, "проход А позвался повторно — список имён не заморожен"
    assert "assign" in seen2, "проход Б не позвался для новых советов"

    tagged2 = json.load(open(tmp_path / "tests" / "tags" / "gr.json", encoding="utf-8"))
    by_id = {r["id"]: r for r in tagged2}
    for r in rows:  # старые — имя НЕ ИЗМЕНИЛОСЬ
        assert by_id[r["id"]]["name"] == "visa documents", by_id[r["id"]]
    for r in rows2[3:]:  # новые — получили имя из ЗАМОРОЖЕННОГО списка
        assert by_id[r["id"]]["name"] == "visa documents", by_id[r["id"]]


def test_names_come_from_pass_a_and_assignment_only_picks(tmp_path, monkeypatch):
    """А называет, Б выбирает: у советов оказываются ИМЕНА ИЗ СПИСКА, а не свои подтемы."""
    monkeypatch.chdir(tmp_path)
    rows = [_row(i, f"label {i}") for i in range(6)]
    _tags(tmp_path, rows)
    _texts(monkeypatch, rows)
    seen = {}

    def fake_call(user, sysprompt, **kw):
        seen[kw["consumer"]] = json.loads(user)
        if kw["consumer"] == "canon":
            return {"names": ["visa documents", "visa processing time"]}
        adv = json.loads(user)["advices"]
        return {"map": {k: ("0" if int(k) >= 3 else "1") for k in adv}}

    monkeypatch.setattr(tract, "call", fake_call)
    tract.summarize("gr")

    # проход А получил МЕТКИ с массами, проход Б — список имён и советы
    assert set(seen) == {"canon", "assign"}, seen
    assert all("(" in v for v in seen["canon"].values()), seen["canon"]
    assert seen["assign"]["names"] == {
        "1": "visa documents",
        "2": "visa processing time",
    }

    tagged = json.load(open(tmp_path / "tests" / "tags" / "gr.json", encoding="utf-8"))
    names = [r.get("name") for r in tagged]
    assert names.count("visa documents") == 3, names
    assert names.count(None) == 3, "«0» не должен давать имя"

    canon = json.load(open(tmp_path / "tests" / "canon.json", encoding="utf-8"))
    assert canon["visa documents"]["slug"] == "visa-documents", canon


def test_name_outside_the_list_is_dropped(tmp_path, monkeypatch):
    """Номер вне списка — промах, а не новое имя: список закрыт."""
    monkeypatch.chdir(tmp_path)
    rows = [_row(i, "label") for i in range(3)]
    _tags(tmp_path, rows)
    _texts(monkeypatch, rows)

    def fake_call(user, sysprompt, **kw):
        if kw["consumer"] == "canon":
            return {"names": ["visa documents"]}
        return {"map": {"0": "1", "1": "7", "2": "нет"}}  # 7 и «нет» — мимо списка

    monkeypatch.setattr(tract, "call", fake_call)
    tract.summarize("gr")
    tagged = json.load(open(tmp_path / "tests" / "tags" / "gr.json", encoding="utf-8"))
    assert [r.get("name") for r in tagged] == ["visa documents", None, None]


def test_real_ids_never_reach_the_mouth(tmp_path, monkeypatch):
    """Сквозное правило: 24-символьный хеш роту не показывается нигде."""
    monkeypatch.chdir(tmp_path)
    rows = [_row(i, f"label {i}") for i in range(3)]
    _tags(tmp_path, rows)
    _texts(monkeypatch, rows)
    payloads = []

    def fake_call(user, sysprompt, **kw):
        payloads.append(user)
        if kw["consumer"] == "canon":
            return {"names": ["visa documents"]}
        return {"map": {"0": "1", "1": "1", "2": "1"}}

    monkeypatch.setattr(tract, "call", fake_call)
    tract.summarize("gr")
    assert payloads, "рот не звался"
    for p in payloads:
        assert "hash-" not in p, p[:200]


def test_advice_without_a_name_goes_to_the_remainder(tmp_path, monkeypatch):
    """Совет без имени идёт в ОСТАТОК темы, а не заводит корзину из своей подтемы."""
    monkeypatch.chdir(tmp_path)
    rows = [_row(i, "своя метка") for i in range(5)]
    for r in rows[:4]:
        r["name"] = "visa documents"
    _tags(tmp_path, rows)
    _texts(monkeypatch, rows)
    tract.build_corpus("gr")
    out = json.load(
        open(tmp_path / "tests" / "out_facet_en" / "gr.json", encoding="utf-8")
    )
    assert [v["title"] for v in out["views_by_task"]] == ["visa documents"]
    assert sum(len(s["items"]) for s in out["shelves"]) == 1, "безымянный не в остатке"
