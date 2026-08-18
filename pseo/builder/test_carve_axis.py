"""Сторож канона §0.15: ОСЬ НАРЕЗКИ = РАЗДЕЛ, и нарезка идёт ДВУМЯ проходами.

Что было не так. Семьи собирались по ПЕРВОМУ СЛОВУ метки задачи: «Сроки оформления
шенгенской визы» и «Требования к документам» уезжали в РАЗНЫЕ семьи, и рот `carve`
физически не мог их сравнить — он видит только свою семью. Замер по визам Греции: 41 разбор
при девяти делах. Раздробленность рождалась в этой оси, а не в модели.

Как стало: раздел ставится каждой мухе (рот `assign`, закрытый список 13) → семья =
«страна × раздел» → проход А составляет ЗАКРЫТЫЙ список дел раздела, проход Б присваивает
мухам дела ИЗ ЭТОГО списка. Закрытый список и есть защита от размножения формулировок.

⛔ Ртов тут не зовём: подменяем их ответы и проверяем СШИВКУ — то, что ломается молча.
Каждое правило проверено поломкой кода.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import facet  # noqa: E402
import tail_taxonomy as tax  # noqa: E402


def _fly(fid, shelf_key, zadachi, text=None):
    return {
        "id": fid,
        "perevod": text or f"Живой совет {fid} про важное дело.",
        "zadachi": zadachi,
        "sushnosti": [],
        "mesto": None,
        "uslovie": None,
        **({"shelf_key": shelf_key} if shelf_key else {}),
    }


def _corpus(monkeypatch, tmp_path, flies, deals=None, mapping=None, tail=None):
    """Прогнать сборку видов, подменив рты. deals — ответ прохода А, mapping — прохода Б."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_deals(mass, fails=None, family=None):
        seen.setdefault("A", []).append((family, dict(mass)))
        return list(deals or [])

    def fake_assign(fids, by_id, dl, fails=None, family=None):
        seen.setdefault("B", []).append((family, list(fids), list(dl)))
        out = {}
        for fid in fids:
            name = (mapping or {}).get(fid)
            if name:
                out.setdefault(name, []).append(fid)
        return out

    def fake_tail(fids, by_id, fails=None):
        seen["tail"] = list(fids)
        return (tail or {}), []

    monkeypatch.setattr(facet, "carve_deals", fake_deals)
    monkeypatch.setattr(facet, "assign_to_deals", fake_assign)
    monkeypatch.setattr(facet, "assign_tail", fake_tail)
    views, shelves, prochee = facet.build_views_by_carve(flies, fails=[])
    return views, shelves, prochee, seen


VISA = [
    _fly("v1", "visa", ["Сроки оформления визы"]),
    _fly("v2", "visa", ["Требования к документам"]),
    _fly("v3", "visa", ["Сроки рассмотрения"]),
    _fly("v4", "visa", ["Пакет документов"]),
    _fly("v5", "visa", ["Финансовые требования"]),
    _fly("v6", "visa", ["Спонсорство поездки"]),
]


def test_family_is_a_section_not_a_first_word(tmp_path, monkeypatch):
    """Мухи ОДНОГО раздела попадают в ОДНУ семью, хотя первые слова меток разные.

    Именно это и лечит раздробленность: «Сроки…» и «Требования…» теперь сравнимы.
    """
    _v, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=["Сроки и рассмотрение", "Документы и деньги"],
        mapping={
            "v1": "Сроки и рассмотрение",
            "v3": "Сроки и рассмотрение",
            "v2": "Документы и деньги",
            "v4": "Документы и деньги",
            "v5": "Документы и деньги",
            "v6": "Документы и деньги",
        },
    )
    assert len(seen["A"]) == 1, "проход А позван не по разделу, а по чему-то ещё"
    family, mass = seen["A"][0]
    assert family == "visa", family
    assert set(mass) == {
        "Сроки оформления визы",
        "Требования к документам",
        "Сроки рассмотрения",
        "Пакет документов",
        "Финансовые требования",
        "Спонсорство поездки",
    }, mass


def test_pass_b_gets_only_the_closed_list(tmp_path, monkeypatch):
    """Проход Б получает РОВНО тот список, что дал проход А — в этом вся защита.

    Если Б увидит что-то другое (или ничего), рот снова начнёт плодить формулировки.
    """
    deals = ["Сроки и рассмотрение", "Документы и деньги"]
    _v, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=deals,
        mapping={"v1": deals[0], "v3": deals[0]},
    )
    assert seen["B"], "проход Б не позван"
    _fam, fids, got = seen["B"][0]
    assert got == deals, got
    assert set(fids) == {f["id"] for f in VISA}, fids


def test_deal_becomes_a_view_and_rest_goes_to_tail(tmp_path, monkeypatch):
    """Дело от двух мух становится видом; неприсвоенные мухи уходят в хвост, а не теряются."""
    deals = ["Сроки и рассмотрение"]
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=deals,
        mapping={"v1": deals[0], "v3": deals[0]},
    )
    assert list(views) == deals, list(views)
    assert {i["id"] for i in views[deals[0]]} == {"v1", "v3"}
    assert set(seen["tail"]) == {"v2", "v4", "v5", "v6"}, seen["tail"]


def test_single_fly_deal_is_not_a_page(tmp_path, monkeypatch):
    """Дело из одной мухи страницей не становится — она уходит в хвост (порог не менялся)."""
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=["Одинокое дело"],
        mapping={"v1": "Одинокое дело"},
    )
    assert views == {}, views
    assert "v1" in seen["tail"]


def test_grab_bag_deal_name_is_rejected(tmp_path, monkeypatch):
    """Сборное имя дела не проходит и на этом этапе (канон §0.13): «Прочее» страницей не станет."""
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=["Прочее"],
        mapping={"v1": "Прочее", "v2": "Прочее"},
    )
    assert views == {}, views
    assert {"v1", "v2"} <= set(seen["tail"]), seen["tail"]


def test_no_deals_means_section_goes_to_tail(tmp_path, monkeypatch):
    """Проход А не дал списка → раздел целиком в хвост, и проход Б НЕ зовётся.

    ⛔ Первая версия проверяла только пустые виды — и мутация «убрать остановку» проходила
    зелёной: без списка присваивать всё равно нечего. Но вызов Б без списка — это трата
    ключей на заведомо пустую работу, поэтому проверяем именно НЕВЫЗОВ.
    """
    views, _s, _p, seen = _corpus(monkeypatch, tmp_path, VISA, deals=[], mapping={})
    assert views == {}, views
    assert "B" not in seen, "проход Б позван без списка дел — ключи впустую"
    assert set(seen["tail"]) == {f["id"] for f in VISA}


def test_thin_section_is_not_carved(tmp_path, monkeypatch):
    """Раздел тоньше порога (MIN_CARVE) не режем — дела там выделять не из чего."""
    flies = VISA[:3]  # 3 < 6
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        flies,
        deals=["Дело"],
        mapping={"v1": "Дело", "v2": "Дело"},
    )
    assert views == {}, views
    assert (
        "A" not in seen
    ), "проход А позван на тонком разделе — это трата ключей впустую"
    assert set(seen["tail"]) == {f["id"] for f in flies}


def test_old_axis_is_the_fallback_and_says_so(tmp_path, monkeypatch, capsys):
    """⚠️ ПЕРЕХОД: у мух ещё нет раздела → работаем по старой оси и ГОВОРИМ об этом.

    Без этого первый же прогон на старых данных отправил бы весь корпус гео в хвост.
    """
    flies = [_fly(f"o{i}", None, ["Сроки оформления визы"]) for i in range(6)]
    called = {}

    def fake_family(fids, by_id, fails=None, family=None):
        called["yes"] = True
        return [{"name": "Сроки оформления визы", "ids": list(fids)}]

    monkeypatch.setattr(facet, "carve_family", fake_family)
    views, _s, _p, _seen = _corpus(
        monkeypatch, tmp_path, flies, deals=["X"], mapping={}
    )
    out = capsys.readouterr().out
    assert called.get("yes"), "старая ось не сработала как откат"
    assert "РАЗДЕЛА У МУХ НЕТ" in out, out
    assert "Сроки оформления визы" in views, list(views)


def test_prochee_flies_are_not_a_section(tmp_path, monkeypatch):
    """Муха в парк-ведре `prochee` разделом не считается: дела из неё не выделяем."""
    flies = [_fly(f"p{i}", "prochee", ["Что-то"]) for i in range(6)]
    views, _s, _p, seen = _corpus(
        monkeypatch, tmp_path, flies, deals=["Дело"], mapping={}
    )
    assert views == {}, views
    assert "A" not in seen, "проход А позван на парк-ведре"


def test_pult_can_run_the_new_step():
    """Прогон обязан запускаться из пульта, а не с моего десктопа: у facet есть флаг."""
    src = (HERE / "facet.py").read_text(encoding="utf-8")
    # ⛔ Ищем РАЗБОР аргумента, а не строку подсказки: первая версия проверки была зелёной
    # на сломанном коде, потому что имя флага осталось в usage.
    assert '"--assign-flies" in sys.argv' in src, "флаг не разбирается в CLI"
    assert "assign_fly_shelves(geo" in src, "флаг ничего не зовёт"
    assert "def assign_fly_shelves(" in src


def test_prompts_carry_the_rules():
    """Промпты обоих проходов несут правила, иначе код чинит то, что рот плодит заново."""
    assert "ЗАКРЫТЫЙ список" in facet.DEALS_SYS
    assert "ЗАПРЕЩЕНЫ" in facet.DEALS_SYS, "имя дела ничем не ограничено"
    assert "ИЗ СПИСКА" in facet.DEAL_ASSIGN_SYS, "проход Б не закрыт списком"
    keys = [k for k, _n, _d in tax.SHELVES]
    for k in keys:
        assert k in facet.FLY_SHELF_SYS, f"раздел {k} не назван в промпте"
