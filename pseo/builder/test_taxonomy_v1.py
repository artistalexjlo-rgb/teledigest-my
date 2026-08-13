"""Сторожа таксономии v1 и целевой пере-раскладки полки.

Повод — замер 13.08. Полка `life` («Работа, учёба, сообщества и быт») была склейкой остатка,
и в её собственной записи стояло «дробить при росте». Выросла: 344 вида, 2293 абзаца, внутри
досуг с пляжами (67 видов), жильё (43), покупки (40), здоровье (19), работа (13). На хабе это
дало бы плитку «Работа, учёба, быт» с Критом и Санторини внутри.

⛔ Главное, что защищаем: смена набора полок ОБЯЗАНА поднимать VERSION. Иначе пульт считает
шаг сделанным (полки есть!) и не предлагает пере-раскладку — ровно так 82 гео выглядели
готовыми, пока пляжи лежали в «работе». Счётчик мерил наличие вместо верности.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import facet  # noqa: E402
import tail_taxonomy as tax  # noqa: E402


def test_thirteen_shelves_unique_and_named():
    keys = [k for k, _, _ in tax.SHELVES]
    assert len(tax.SHELVES) == 13, f"полок {len(tax.SHELVES)}, ожидалось 13"
    assert len(set(keys)) == len(keys), "дубли ключей полок"
    for k, name, desc in tax.SHELVES:
        assert k and name and desc, f"полка {k!r} без имени или описания"
        assert len(desc) > 30, f"полка {name}: описание короче границы, чем это полезно"


def test_life_is_gone_and_replacements_present():
    """Разобранной полки больше нет, а пять новых на месте — иначе пере-раскладке некуда
    складывать, и модель снова положит пляжи в «работу»."""
    names = set(tax.SHELF_NAMES)
    assert "Работа, учёба, сообщества и быт" not in names, "склейка остатка вернулась"
    for need in (
        "Туризм и досуг",
        "Жильё и аренда",
        "Покупки и сервисы",
        "Работа, учёба и сообщества",
        "Здоровье и медицина",
    ):
        assert need in names, f"нет полки «{need}»"


def test_version_bumped_past_v0():
    """Версия — единственный сигнал, по которому пульт узнаёт об устаревшей раскладке."""
    assert (
        tax.VERSION != "v0-2026-07-19"
    ), "набор полок сменился, а версия осталась старой"
    assert tax.VERSION.startswith("v1-"), tax.VERSION


def test_boundaries_say_where_the_four_disputed_cases_go():
    """Решения юзера 13.08 записаны В ГРАНИЦАХ, а не в моей памяти: погода, выбор региона и
    прививки-на-въезд → туризм; местные идентификаторы (CPF) → документы."""
    by_key = {k: desc for k, _, desc in tax.SHELVES}
    t = by_key["tourism"].lower()
    for word in ("погод", "регион", "прививк"):
        assert word in t, f"в границе туризма нет «{word}»"
    assert (
        "cpf" in by_key["docs"].lower()
    ), "местные идентификаторы не попали в документы"
    assert "поездк" in by_key["housing"].lower(), "жильё не отделено от поездочного"


def test_reassign_shelf_moves_only_that_shelf(tmp_path, monkeypatch):
    """Целевой режим трогает ОДНУ полку, остальные оставляет как есть, и честно помечает,
    что пере-разложено, — иначе файл заявлял бы полное соответствие новой таксономии."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tags").mkdir()
    (tmp_path / "out_facet").mkdir()
    flies = [
        {
            "id": f"f{i}",
            "perevod": f"Text {i}",
            "sushnosti": [],
            "mesto": None,
            "uslovie": None,
        }
        for i in range(4)
    ]
    (tmp_path / "tags" / "gr.json").write_text(
        json.dumps(flies, ensure_ascii=False), encoding="utf-8"
    )
    page = {
        "geo": "gr",
        "views_by_task": [],
        "shelves": [
            {
                "shelf": "Визовые процедуры",
                "items": [{"id": "f0", "type": "Кейс-отзыв"}],
            },
            {
                "shelf": "Работа, учёба, сообщества и быт",
                "items": [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}],
            },
        ],
        "prochee": [],
        "taxonomy_version": "v0-2026-07-19",
    }
    (tmp_path / "out_facet" / "gr.json").write_text(
        json.dumps(page, ensure_ascii=False), encoding="utf-8"
    )

    def fake_assign_tail(fids, by_id, fails=None):
        # модель не зовём: проверяем СШИВКУ результата, а не саму раскладку
        return {"Туризм и досуг": [{"id": f, "type": "Кейс-отзыв"} for f in fids]}, []

    monkeypatch.setattr(facet, "assign_tail", fake_assign_tail)
    n = facet.run_reassign_shelf("gr", "Работа, учёба, сообщества и быт")
    assert n == 3, f"взято мух {n}, ожидалось 3"

    got = json.loads((tmp_path / "out_facet" / "gr.json").read_text(encoding="utf-8"))
    shelves = {s["shelf"]: s["items"] for s in got["shelves"]}
    assert "Работа, учёба, сообщества и быт" not in shelves, "старая полка осталась"
    assert len(shelves["Туризм и досуг"]) == 3, "мухи не переехали"
    assert [i["id"] for i in shelves["Визовые процедуры"]] == [
        "f0"
    ], "чужую полку тронули"
    assert got["taxonomy_version"] == tax.VERSION
    assert got["taxonomy_reassigned"] == [
        "Работа, учёба, сообщества и быт"
    ], "не записано, ЧТО именно пере-разложено — файл заявлял бы полное соответствие"


def test_reassign_missing_shelf_is_noop(tmp_path, monkeypatch):
    """Полки нет — не падаем и ничего не пишем: пульт может позвать по устаревшему списку."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tags").mkdir()
    (tmp_path / "out_facet").mkdir()
    (tmp_path / "tags" / "xx.json").write_text("[]", encoding="utf-8")
    p = tmp_path / "out_facet" / "xx.json"
    p.write_text(json.dumps({"geo": "xx", "shelves": []}), encoding="utf-8")
    before = p.read_bytes()
    assert facet.run_reassign_shelf("xx", "Нет такой полки") == 0
    assert p.read_bytes() == before, "файл тронут при пустой работе"
