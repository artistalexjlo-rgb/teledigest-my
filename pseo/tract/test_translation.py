# -*- coding: utf-8 -*-
"""Сторож звена 6: переводится ВСЁ видимое, платим только за новое, адреса не двигаются.

Четыре обещания, ради которых переводчик переписан заново (PLAN.md, звено 6):

1. **русский — обычный язык перевода**. Старый модуль переводил С РУССКОГО, и русская
   версия была источником; теперь источник английский, а `ru` идёт как прочие тринадцать;
2. **платим за новое и за переписанное**, остальное берётся из готового файла — иначе
   каждый прогон покупал бы весь корпус заново;
3. **адрес, ветка и части НЕ трогаются переводом** — иначе переключатель языка уведёт на
   другую страницу, а хвост адреса перестанет быть общим;
4. **мелочь остатка тоже переводится**: она показывается абзацами на странице темы, и
   английская вставка посреди русской страницы — это брак.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import translation  # noqa: E402


def _korpus(tmp_path, monkeypatch, views, shelves=()):
    monkeypatch.setattr(translation, "BUILT", str(tmp_path))
    themes_file = tmp_path / "theme_names.json"
    if not themes_file.exists():
        # Как в бою: themes.json несёт только АНГЛИЙСКИЙ источник. Русский — обычный
        # язык перевода, как и советы, и покупается ротом наравне с прочими тринадцатью.
        themes_file.write_text(
            json.dumps({"en": {"visa": "Visa Procedures"}}), encoding="utf-8"
        )
    monkeypatch.setattr(translation, "THEMES_FILE", str(themes_file))
    os.makedirs(tmp_path / "out_facet_en", exist_ok=True)
    with open(tmp_path / "out_facet_en" / "gr.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"geo": "gr", "views_by_task": list(views), "shelves": list(shelves)},
            fh,
            ensure_ascii=False,
        )


def _view(zadacha, adres, texts, tema="visa", branch=None, part=1, parts=1):
    return {
        "title": zadacha,
        "theme": tema,
        "slug": adres,
        "branch": branch or adres,
        "part": part,
        "parts": parts,
        "items": [
            {"id": f"{adres}-{i}", "text": t, "n": 1} for i, t in enumerate(texts)
        ],
    }


def _rot(monkeypatch, schet=None):
    """Рот-заглушка: возвращает «<lang>:<текст>». Считает, сколько текстов купили."""

    def fake_call(user, sysprompt, **kw):
        payload = json.loads(user)
        if schet is not None:
            schet.setdefault(kw["consumer"], []).extend(payload.values())
        metka = "ru" if "Russian" in sysprompt else "xx"
        return {k: f"{metka}:{v}" for k, v in payload.items()}

    monkeypatch.setattr(translation, "call", fake_call)


def _out(tmp_path, lang="ru"):
    with open(tmp_path / f"out_facet_{lang}" / "gr.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_russian_is_an_ordinary_target_language(tmp_path, monkeypatch):
    """`ru` переводится ротом, как и прочие: отдельного пути к нему нет."""
    _korpus(
        tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a", "b"])]
    )
    schet = {}
    _rot(monkeypatch, schet)
    translation.translate_geo("gr", "ru")
    out = _out(tmp_path)
    assert [it["text"] for it in out["views_by_task"][0]["items"]] == ["ru:a", "ru:b"]
    assert out["views_by_task"][0]["title"] == "ru:visa documents"
    assert sorted(schet) == ["labels", "translate"]


def test_second_run_buys_only_what_changed(tmp_path, monkeypatch):
    """Второй прогон покупает ТОЛЬКО переписанный текст и новое имя, остальное готово."""
    _korpus(
        tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a", "b"])]
    )
    _rot(monkeypatch)
    translation.translate_geo("gr", "ru")

    # источник переписали в одном совете, имя ветки поменяли
    _korpus(
        tmp_path, monkeypatch, [_view("visa papers", "visa-documents", ["a", "B-2"])]
    )
    schet = {}
    _rot(monkeypatch, schet)
    translation.translate_geo("gr", "ru")
    assert schet["translate"] == ["B-2"], schet
    assert schet["labels"] == ["visa papers"], schet
    out = _out(tmp_path)
    assert [it["text"] for it in out["views_by_task"][0]["items"]] == ["ru:a", "ru:B-2"]


def test_branch_name_is_bought_once_for_all_its_parts(tmp_path, monkeypatch):
    """Имя одно на ветку: три части не значат три покупки заголовка.

    Тема на русском заранее в файле — тест смотрит на ветки, а не на темы, и имя темы
    покупается отдельным, уже проверенным путём (см. purchase-тест звена тем).
    """
    _korpus(
        tmp_path,
        monkeypatch,
        [
            _view(
                "visa documents",
                "visa-documents",
                ["a"],
                branch="visa-documents",
                part=1,
                parts=2,
            ),
            _view(
                "visa documents",
                "visa-documents-2",
                ["b"],
                branch="visa-documents",
                part=2,
                parts=2,
            ),
        ],
    )
    (tmp_path / "theme_names.json").write_text(
        json.dumps(
            {"en": {"visa": "Visa Procedures"}, "ru": {"visa": "Визовые процедуры"}}
        ),
        encoding="utf-8",
    )
    schet = {}
    _rot(monkeypatch, schet)
    translation.translate_geo("gr", "ru")
    assert schet["labels"] == ["visa documents"], schet


def test_addresses_and_branches_survive_translation(tmp_path, monkeypatch):
    """Перевод меняет ТЕКСТ, а не адрес: иначе переключатель языка уведёт не туда."""
    _korpus(
        tmp_path,
        monkeypatch,
        [
            _view(
                "visa documents",
                "visa-documents",
                ["a"],
                branch="visa-documents",
                part=1,
                parts=2,
            ),
            _view(
                "visa documents",
                "visa-documents-2",
                ["b"],
                branch="visa-documents",
                part=2,
                parts=2,
            ),
        ],
    )
    _rot(monkeypatch)
    translation.translate_geo("gr", "ru")
    out = _out(tmp_path)
    assert [v["slug"] for v in out["views_by_task"]] == [
        "visa-documents",
        "visa-documents-2",
    ]
    assert {v["branch"] for v in out["views_by_task"]} == {"visa-documents"}
    assert [v["part"] for v in out["views_by_task"]] == [1, 2]


def test_the_small_remainder_is_translated_too(tmp_path, monkeypatch):
    """Мелочь остатка видна читателю абзацами — значит переводится вместе со всем."""
    _korpus(
        tmp_path,
        monkeypatch,
        [_view("visa documents", "visa-documents", ["a"])],
        shelves=[
            {
                "items": [{"id": "x1", "text": "leftover", "n": 1}],
            }
        ],
    )
    _rot(monkeypatch)
    translation.translate_geo("gr", "ru")
    out = _out(tmp_path)
    assert out["shelves"][0]["items"][0]["text"] == "ru:leftover"


def test_theme_names_are_bought_from_english_and_cached(tmp_path, monkeypatch):
    """Имена тем: русский — обычный язык перевода, ни особого пути, ни готового текста.

    Юзер 27.08: «схема — все переводим с английского»; готовый русский в файле рядом с
    покупными двенадцатью языками — та же узкая классификация другими словами.
    """
    _korpus(tmp_path, monkeypatch, [_view("visa documents", "visa-documents", ["a"])])
    schet = {}
    _rot(monkeypatch, schet)
    translation.translate_geo("gr", "ru")
    assert "border" in schet["labels"], "источником для темы был не английский список"
    themes = json.loads((tmp_path / "theme_names.json").read_text(encoding="utf-8"))
    assert themes["ru"]["visa"] == "ru:Visa Procedures"

    schet2 = {}
    _rot(monkeypatch, schet2)
    translation.translate_geo("gr", "ru")
    assert "labels" not in schet2 or "border" not in schet2.get(
        "labels", []
    ), "второй прогон купил темы заново — кэш не сработал"


def test_themes_file_lives_on_the_mounted_volume_not_in_the_image():
    """themes.json РАСТЁТ (звено 6 дописывает языки) — обязан жить на BUILT_DIR, а не
    рядом с кодом: иначе редеплой контейнера стирал бы все покупки (28.08, юзер поймал).

    ⛔ `BUILT`/`THEMES_FILE` — константы, посчитанные при импорте: монкипатчить `BUILT` и
    ждать, что `THEMES_FILE` пересчитается, бессмысленно. Проверяем СВЯЗЬ на текущих
    значениях, как есть.
    """
    assert translation.THEMES_FILE == f"{translation.BUILT}/themes.json"
    assert (
        translation.SEED_THEMES_FILE != translation.THEMES_FILE
    ), "сид и рабочий файл — РАЗНЫЕ пути, иначе первый прогон затирал бы сид"


def test_first_touch_of_an_empty_volume_seeds_english_from_git(tmp_path, monkeypatch):
    """Свежий контейнер, смонтированный том ещё пуст: английский источник копируется из
    git-сида на том и там остаётся — второй прогон сид уже не трогает.
    """
    monkeypatch.setattr(translation, "BUILT", str(tmp_path))
    monkeypatch.setattr(translation, "THEMES_FILE", str(tmp_path / "themes.json"))
    assert not (tmp_path / "themes.json").exists(), "том должен стартовать пустым"

    schet = {}
    _rot(monkeypatch, schet)
    translation.theme_names("ru")
    on_disk = json.loads((tmp_path / "themes.json").read_text(encoding="utf-8"))
    assert on_disk["en"], "английский сид не скопировался на том при первом касании"
    assert (
        len(schet.get("labels") or []) == 13
    ), "русский всё равно куплен ротом, а не сид"
