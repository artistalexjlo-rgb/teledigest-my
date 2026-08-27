"""Сторож спасения записей: битое тело стоит ОДНОЙ записи, а не всей пачки.

⛔ Повод (20.08): ответ рта на пачку — один большой JSON, и одна незакрытая кавычка внутри
перевода уносила все 25 мух. Замер по логу: 42 таких случая, 31 из них на переводе. Заказ юзера:
не экранировать кавычки, а резать ответ по фигурным скобкам и разбирать записи по одной. Скобки
годятся по замеру: в 26 241 тексте корпуса `{` и `}` не встречаются ни разу.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import keybroker  # noqa: E402

BITOE = (
    '{"rows": ['
    '{"i": "0", "perevod": "Аренда от 45 евро", "tema": "transport", "podtema": "аренда авто"},'
    '{"i": "1", "perevod": "Тут рот сломал "кавычку", запятая и мусор", "tema": "x"},'
    '{"i": "2", "perevod": "Парковка 1.3 евро в час", "tema": "transport", "podtema": "парковка"}'
    "]}"
)


def test_salvage_keeps_good_records():
    """Из битого тела достаются целые записи, битая теряется одна."""
    recs = keybroker.salvage_objects(BITOE, "podtema")
    assert len(recs) == 2, recs
    assert [r["i"] for r in recs] == ["0", "2"], recs
    assert recs[0]["podtema"] == "аренда авто"


def test_salvage_looks_inside_the_wrapper():
    """Записи лежат внутри обёртки `{"rows": [...]}` — спасатель обязан заглянуть внутрь."""
    celoe = '{"rows": [{"i": "7", "podtema": "паром"}]}'
    recs = keybroker.salvage_objects(celoe, "podtema")
    assert [r["i"] for r in recs] == ["7"], recs


def test_salvage_ignores_records_without_the_key():
    """Кусок без нужного ключа — не запись: обёртки и обрывки не должны считаться."""
    assert keybroker.salvage_objects('{"a": 1} {"b": 2}', "podtema") == []


def test_call_actually_uses_salvage():
    """⛔ Спасатель сам по себе бесполезен: его должен звать общий вызов на битом теле.

    Мутация «выключить спасение в вызове» проходила зелёной — сторожей на проводку не было.
    Поведением это не проверить без битого HTTP-ответа, поэтому проверяем, что ветка на месте.
    """
    src = (HERE / "keybroker.py").read_text(encoding="utf-8")
    body = src[src.index("def call(") :]
    assert "if salvage:" in body, "вызов не спасает записи"
    assert "salvage_objects(raw, salvage[1])" in body, "вызов не зовёт спасателя"
    assert "return {salvage[0]: recs}" in body, "спасённое не возвращается"


def test_tract_steps_ask_for_salvage():
    """Оба рта тракта обязаны просить спасение: ответ приходит на всю пачку сразу.

    Шаги тракта по канону §0.19: разметка (пачка 25) и обобщение — проход А (список имён
    на тему) плюс проход Б (присваивание пачками).
    """
    src = (HERE / "tract.py").read_text(encoding="utf-8")
    mk = src[src.index("def mark(") : src.index("def summarize(")]
    assert 'salvage=("rows", "subtheme")' in mk, "разметка не просит спасения"
    ob = src[src.index("def summarize(") :]
    # звено 4 = два прохода, и спасать обязаны ОБА: битое тело стоит одной пачки, а не темы
    assert 'salvage=("names", "names")' in ob, "проход А не просит спасения"
    assert 'salvage=("map", "map")' in ob, "проход Б не просит спасения"
