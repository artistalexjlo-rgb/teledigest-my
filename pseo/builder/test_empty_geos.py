"""НЕЧЕГО ПОКАЗАТЬ — НЕ ПОКАЗЫВАЙ. Гео без страниц не получает ни хаба, ни чипа, ни адреса.

Повод — живой замер 2026-08-12. На зеркале 33 хаба гео отдавали 200 при НУЛЕ ссылок внутрь:
страница из одной обвязки (804 символа интро, CTA и подвала). У всех этих гео ноль видов и
1–8 мух на всё гео. На `.online` те же гео давали 404 прямо из навигации: главная их
перечисляла по корпусу, а пуш отсеивал по зрелости — рендер и отгрузка расходились.

⛔ Почему это не поймал `readycheck`: его правило «пустая страница» — текста меньше 400
символов, а обвязка весит 804. Порог ниже веса шаблона, значит по пустоте гейт не мог
сработать НИ РАЗУ. Правило должно жить там, где видно содержимое, — в сборке.

Проверяем поведением: собираем два гео в одном временном корпусе — с содержимым и без.
Сети, ключей и БД не требует.
"""

import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import pages as pg  # noqa: E402


def _view(zadacha, n, key):
    items = [{"id": f"{key}{i}", "text": f"Advice {i}. Detail."} for i in range(n)]
    return {
        "zadacha": zadacha,
        "key": key,
        "items": items,
        "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items],
    }


def _corpus(tmp, geo, views, shelves=()):
    os.makedirs(f"{tmp}/out_facet", exist_ok=True)
    json.dump(
        {"geo": geo, "views_by_task": list(views), "shelves": list(shelves)},
        open(f"{tmp}/out_facet/{geo}.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )


def _build(geo, views, shelves=()):
    """Собрать одно гео в изоляции. Возвращает (результат build_geo, {файл: данные})."""
    tmp, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    _corpus(tmp, geo, views, shelves)
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = tmp, out
    try:
        res = pg.build_geo(geo, "ru")
    finally:
        pg.BUILT, pg.DATA = built, data
    files = {
        fn: json.load(open(f"{out}/{fn}", encoding="utf-8")) for fn in os.listdir(out)
    }
    return res, files


def test_geo_with_content_gets_hub():
    """Контроль: годное гео собирается как раньше. Без этого тест ничего не значил бы —
    «ничего не собралось» проходило бы как успех."""
    res, files = _build("xx", [_view("Banking", 6, "banking")])
    assert res[0] > 0, "гео с содержимым должно собраться"
    hubs = [f for f in files if f.endswith("_hub.json")]
    assert hubs, f"хаб не написан, а должен: {sorted(files)}"
    hub = files[hubs[0]]
    assert hub["tiles"], "хаб без плиток — это и есть пустая страница"


def test_empty_geo_gets_no_hub():
    """Ни одного вида, дотянувшего до страницы (порог 4 пункта) → хаба нет вообще."""
    res, files = _build("zz", [_view("Thin", 2, "thin")])
    assert res == (0, 0, 0, 0), f"пустое гео должно вернуть нули, вернуло {res}"
    assert not [
        f for f in files if f.endswith("_hub.json")
    ], f"хаб написан для гео без страниц: {sorted(files)}"


def test_no_hub_without_internal_links():
    """Инвариант, а не частный случай: КАЖДЫЙ написанный хаб имеет хотя бы одну ссылку
    внутрь гео. Именно этого правила не хватало — 33 хаба на зеркале его нарушали."""
    for geo, views in (
        ("aa", [_view("Docs", 5, "docs")]),
        ("bb", [_view("T", 1, "t")]),
    ):
        _res, files = _build(geo, views)
        for fn, d in files.items():
            if not fn.endswith("_hub.json"):
                continue
            urls = [t.get("url", "") for t in d.get("tiles") or []]
            inner = [u for u in urls if u.startswith(f"/ru/{geo}/")]
            assert inner, f"{fn}: хаб без ссылок внутрь гео"


def test_thin_geo_prints_reason():
    """Отсев обязан быть ВИДЕН в логе. Молчаливый пропуск — то, из-за чего 33 пустышки
    год жили незамеченными: ни в логе, ни в тесте отдельной страницы их не видно."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _build("zz", [_view("Thin", 2, "thin")])
    log = buf.getvalue()
    assert (
        "пропущено" in log and "хаб не пишем" in log
    ), f"причина не напечатана: {log!r}"
