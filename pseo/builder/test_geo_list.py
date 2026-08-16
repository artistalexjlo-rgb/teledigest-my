"""Сторож шага 8: ГЕО У МУХИ — СПИСОК, а не одно значение.

Корень, замеренный по базе: у мухи ОДНА колонка `country` и сравнение на равенство
(`WHERE country=?`), а промпт экстрактора давал бинарный выбор «одна страна или any».
Совет про две страны выразить было нечем — он падал либо в `any` (3 077 мух), либо в
мусорный ключ с запятой: таких 29 (`de, ru` 5, `ru, kg` 4, `kg, kz, ru` 2, `au, nz` 2 …),
и эти мухи не попадали НИ В ОДНУ страну. Один дефект с двумя исходами, оба видели живьём.

⛔ Схему НЕ меняем (решение записано в реестре): колонка остаётся одна, читатель берёт
мух по ВХОЖДЕНИЮ кода в список, нормализация — на чтении, по справочнику стран.

⛔ Что защищаем:
  1. `de, ru` попадает и в Германию, и в Россию;
  2. неизвестный код отбрасывается — гео-призрак страницей не станет;
  3. `an` не цепляет `any`, а `ru` — не цепляет всё, где есть эти буквы (грубый `LIKE`
     в SQL обязан уточняться точной проверкой);
  4. `any` остаётся отдельным псевдо-гео и в страны НЕ растекается: пере-тег 3 077 мух
     из `any` — отдельная работа по отмашке юзера, а не побочный эффект шага;
  5. промпт экстрактора разрешает перечисление, иначе новых списков просто не появится.

⛔ Правила проверены поломкой кода.

Сети и ключей не требует; SQLite — временная база на фикстуре.
"""

import pathlib
import sqlite3
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import facet  # noqa: E402

TEXT = "A long enough lesson about local rules, prices and paperwork. " * 4


def _db(tmp_path, rows):
    """Мини-база того же вида, что боевая: id, country, ai_lesson."""
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE extracted_patterns (id INTEGER PRIMARY KEY, country TEXT, "
        "ai_lesson TEXT)"
    )
    c.executemany(
        "INSERT INTO extracted_patterns (id, country, ai_lesson) VALUES (?,?,?)", rows
    )
    c.commit()
    c.close()
    return str(p)


def test_codes_are_parsed_and_normalised():
    """Разбор значения колонки: регистр, пробелы, разделители, мусор."""
    assert facet.geo_codes("de, ru") == {"de", "ru"}
    assert facet.geo_codes(" DE ;ru ") == {"de", "ru"}
    assert facet.geo_codes("kg, kz, ru") == {"kg", "kz", "ru"}
    assert facet.geo_codes("any") == {"any"}
    assert facet.geo_codes("") == set() and facet.geo_codes(None) == set()
    # ⛔ Неизвестное отбрасываем: «eu» и «uk» — не коды ISO, и именно из таких значений
    # рождались гео-призраки, у которых на сайте были пустые хабы.
    assert facet.geo_codes("eu") == set(), "гео-призрак прошёл"
    assert facet.geo_codes("de, eu") == {"de"}


def test_multi_geo_fly_lands_in_every_country(tmp_path, monkeypatch):
    """`de, ru` обязан прийти И в Германию, И в Россию. До шага 8 — ни в одну."""
    monkeypatch.setattr(
        facet,
        "DB",
        _db(
            tmp_path,
            [
                (1, "de, ru", TEXT),
                (2, "de", TEXT),
                (3, "ru", TEXT),
            ],
        ),
    )
    assert {r[0] for r in facet.load_flies("de")} == {1, 2}
    assert {r[0] for r in facet.load_flies("ru")} == {1, 3}


def test_unknown_code_gives_no_geo(tmp_path, monkeypatch):
    """Мусорное значение не даёт гео вовсе — ни своего, ни чужого."""
    monkeypatch.setattr(facet, "DB", _db(tmp_path, [(1, "eu", TEXT), (2, "de", TEXT)]))
    assert facet.load_flies("eu") == []
    assert {r[0] for r in facet.load_flies("de")} == {2}


def test_substring_does_not_leak(tmp_path, monkeypatch):
    """⛔ `LIKE` в запросе — грубый пред-отбор, и без точной проверки он течёт:
    `an` цеплял бы `any`, `ru` — `bru`. Пред-отбор нужен только чтобы не тащить все
    23 924 текста на каждый вызов, решение принимает разбор списка."""
    monkeypatch.setattr(
        facet,
        "DB",
        _db(
            tmp_path,
            [
                (1, "any", TEXT),
                (2, "ru", TEXT),
                (
                    3,
                    "bru",
                    TEXT,
                ),  # не код ISO: и как гео не существует, и в `ru` попасть не должен
            ],
        ),
    )
    assert facet.load_flies("an") == [], "код `an` подобрал мух из `any`"
    assert {r[0] for r in facet.load_flies("ru")} == {2}
    assert {r[0] for r in facet.load_flies("any")} == {1}


def test_any_does_not_spread_into_countries(tmp_path, monkeypatch):
    """`any` остаётся своим псевдо-гео. Растечься по странам он не должен: это 3 077 мух,
    и их пере-тег — отдельная работа по отмашке, а не побочный эффект чтения."""
    monkeypatch.setattr(facet, "DB", _db(tmp_path, [(1, "any", TEXT), (2, "de", TEXT)]))
    assert {r[0] for r in facet.load_flies("de")} == {2}
    assert {r[0] for r in facet.load_flies("any")} == {1}


def test_exclude_and_limit_still_work(tmp_path, monkeypatch):
    """Бережное потребление не сломано: `exclude` и `limit` работают как раньше."""
    monkeypatch.setattr(
        facet,
        "DB",
        _db(
            tmp_path,
            [
                (1, "de, ru", TEXT),
                (2, "de", TEXT),
                (3, "de", TEXT),
            ],
        ),
    )
    assert [r[0] for r in facet.load_flies("de", limit=2)] == [1, 2]
    assert [r[0] for r in facet.load_flies("de", exclude={1, 2})] == [3]


def test_short_and_junk_flies_still_filtered(tmp_path, monkeypatch):
    """Порог длины И junk-фильтр остались на месте — иначе шаг 8 протащил бы мусор.

    ⛔ Первая версия этого сторожа держала только КОРОТКУЮ муху, поэтому мутация «снять
    junk-фильтр» оставалась зелёной: короткую отсекал порог в SQL, а не фильтр. Нужна
    настоящая junk-муха — той формы, что и в базе: пересказ запроса вместо совета.
    """
    junk = "User is asking about visa requirements and needed documents. " * 4
    assert facet.is_junk(junk), "фикстура не junk — сторож ничего не проверит"
    monkeypatch.setattr(
        facet,
        "DB",
        _db(
            tmp_path,
            [
                (1, "de", "слишком коротко"),
                (2, "de, ru", TEXT),
                (3, "de", junk),
            ],
        ),
    )
    assert {r[0] for r in facet.load_flies("de")} == {2}


def test_extractor_prompt_allows_a_list():
    """Промпт обязан РАЗРЕШАТЬ перечисление и сузить `any`, иначе новых списков в базе
    не появится и правка читателя останется без данных."""
    src = (HERE.parent.parent / "src" / "teledigest" / "extraction.py").read_text(
        encoding="utf-8"
    )
    # ⛔ Смотрим ИМЕННО пункт про country, а не весь промпт: первая версия проверки искала
    # «через запятую» по всему тексту и оставалась зелёной, когда правило из пункта убрали.
    assert "- country:" in src, "пункт про страну исчез из промпта"
    bullet = src.split("- country:")[1].split("- routing:")[0]
    assert "через запятую" in bullet, "перечисление в пункте про страну не разрешено"
    assert "НЕСКОЛЬКИХ стран" in bullet, "не сказано, что стран может быть несколько"
    assert "универсального" in bullet, "`any` не сужен до действительно универсального"
