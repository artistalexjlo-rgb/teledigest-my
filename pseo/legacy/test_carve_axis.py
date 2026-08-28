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
# `legacy` — отменённая схема: сторожу читать оттуда МОЖНО (правило живое,
# код мёртв), править нельзя. См. pseo/legacy/README.md
sys.path[:0] = [str(HERE), str(HERE.parent / "legacy")]
LEGACY = HERE.parent / "legacy"

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


def _corpus(
    monkeypatch,
    tmp_path,
    flies,
    deals=None,
    mapping=None,
    tail=None,
    answered=True,
    stall=False,
):
    """Прогнать сборку видов, подменив рты. deals — ответ прохода А, mapping — прохода Б.

    `answered=False` — проход А не доехал (429/транспорт/СТОП). `stall=True` — не доехал
    проход Б. Канон §0.17: такие мухи не размещаются нигде и остаются работой шага.
    """
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_deals(mass, fails=None, family=None):
        seen.setdefault("A", []).append((family, dict(mass)))
        return list(deals or []), answered

    def fake_assign(fids, by_id, dl, fails=None, family=None):
        seen.setdefault("B", []).append((family, list(fids), list(dl)))
        if stall:
            return {}, list(fids)
        out = {}
        for fid in fids:
            name = (mapping or {}).get(fid)
            if name:
                out.setdefault(name, []).append(fid)
        return out, []

    def fake_tail(fids, by_id, fails=None):
        seen["tail"] = list(fids)
        return (tail or {}), []

    monkeypatch.setattr(facet, "carve_deals", fake_deals)
    monkeypatch.setattr(facet, "assign_to_deals", fake_assign)
    monkeypatch.setattr(facet, "assign_tail", fake_tail)
    fails = []
    views, shelves, prochee = facet.build_views_by_carve(flies, fails=fails)
    seen["fails"] = fails
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


def test_deal_becomes_a_view_only_at_the_page_threshold(tmp_path, monkeypatch):
    """Дело становится видом, ТОЛЬКО дотянув до порога страницы; тонкое дело — в хвост.

    ⛔ ЭТОТ СТОРОЖ БЫЛ НАПИСАН НАВЫВОРОТ (19.08, мной же): он требовал, чтобы дело из ДВУХ мух
    стало видом. Страницей такой вид не станет никогда — `pages.py` режет по `PAGE_MIN`, — а
    мухи считались «нарезанными» и в хвост не попадали: абзацы уходили С САЙТА. Пока сторож
    стоял так, верная починка выглядела регрессом. Решение о странице ОДНО и живёт у владельца
    числа (`tail_taxonomy.PAGE_MIN`).
    """
    deals = ["Сроки и рассмотрение", "Документы и деньги"]
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=deals,
        mapping={
            "v1": deals[0],
            "v2": deals[0],
            "v3": deals[0],
            "v4": deals[0],  # четыре мухи — дотянуло до страницы
            "v5": deals[1],
            "v6": deals[1],  # две — не дотянуло
        },
    )
    assert list(views) == [deals[0]], list(views)
    assert {i["id"] for i in views[deals[0]]} == {"v1", "v2", "v3", "v4"}
    # ⛔ Ни одна муха тонкого дела не пропала: обе в хвосте, а не «нарезаны и забыты».
    assert set(seen["tail"]) == {"v5", "v6"}, seen["tail"]
    assert (
        tax.PAGE_MIN == 4
    ), "фикстура написана под порог 4 — сменился порог, правь фикстуру"


def test_page_threshold_comes_from_the_owner_at_runtime(tmp_path, monkeypatch):
    """Порог берётся у ВЛАДЕЛЬЦА живьём: подменили его значение — состав видов поехал.

    ⛔ Это то, чего не докажет ни один греп: файл может ссылаться на владельца, а решать по
    своей копии, снятой в момент импорта. Подменяем 4 → 3 и требуем, чтобы дело из ТРЁХ мух
    стало видом. Мутация «вернуть псевдоним PAGE_MIN = tax.PAGE_MIN и решать по нему» краснеет.
    """
    monkeypatch.setattr(tax, "PAGE_MIN", 3)
    deals = ["Сроки и рассмотрение"]
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=deals,
        mapping={"v1": deals[0], "v2": deals[0], "v3": deals[0]},
    )
    assert list(views) == deals, "порог владельца не дошёл до нарезки"
    assert {i["id"] for i in views[deals[0]]} == {"v1", "v2", "v3"}
    assert set(seen["tail"]) == {"v4", "v5", "v6"}, seen["tail"]


def test_view_that_shrank_below_the_threshold_sends_its_flies_to_tail(
    tmp_path, monkeypatch
):
    """Вид, ПРОСЕВШИЙ ниже порога после дедупа мух, страницей не считается — мухи в хвост.

    ⛔ Зачем это отдельно от гейта. Гейт смотрит на размер дела ДО дедупа, а «страховка: дедуп
    мух в карв-виде по id» (одна муха попадает дважды на стыке семей) может увести вид ниже
    порога ПОСЛЕ него. Проверяем именно предикат хвоста: он обязан считать по СТРАНИЦЕ, а не
    по факту «муха где-то в виде лежит». Мутация «считать любой вид страницей» краснеет здесь.
    """
    deals = ["Сроки и рассмотрение"]
    # рот отдал одну и ту же муху четыре раза: дело выглядит толстым, а мух в нём одна
    monkeypatch.setattr(
        facet,
        "assign_to_deals",
        lambda *a, **k: ({deals[0]: ["v1", "v1", "v1", "v1"]}, []),
    )
    monkeypatch.setattr(facet, "carve_deals", lambda *a, **k: (list(deals), True))
    tail = {}
    monkeypatch.setattr(
        facet,
        "assign_tail",
        lambda fids, by_id, fails=None: (tail.setdefault("ids", list(fids)), [])
        and ({}, []),
    )
    monkeypatch.chdir(tmp_path)
    views, _s, _p = facet.build_views_by_carve(VISA, fails=[])
    assert views == {}, f"просевший вид остался страницей: {views}"
    assert (
        "v1" in tail["ids"]
    ), "муха просевшего вида не попала в хвост — абзацы потеряны"


def test_batch_that_did_not_arrive_leaves_flies_as_work(tmp_path, monkeypatch):
    """§0.17: пачка прохода Б не доехала → мухи НЕ размещаются нигде.

    ⛔ Раньше они уезжали в хвост, а хвост — это размещение, то есть заявление «с мухой
    разобрались». После него шаг считает гео сделанным, и упавшее заново не встаёт.
    """
    views, _s, _p, seen = _corpus(
        monkeypatch, tmp_path, VISA, deals=["Сроки и рассмотрение"], stall=True
    )
    assert views == {}, views
    assert seen["tail"] == [], f"застрявшие мухи уехали в хвост: {seen['tail']}"


def test_pass_a_that_did_not_arrive_does_not_dump_the_section(tmp_path, monkeypatch):
    """§0.17 для прохода А: рот не ответил → раздел остаётся работой, а не едет в хвост.

    Обратный случай проверяет `test_no_deals_means_section_goes_to_tail`: рот ОТВЕТИЛ, годных
    имён нет — тогда хвост законен. Различать эти два случая и есть всё правило.
    """
    views, _s, _p, seen = _corpus(monkeypatch, tmp_path, VISA, deals=[], answered=False)
    assert views == {}, views
    assert (
        seen["tail"] == []
    ), f"раздел уехал в хвост, хотя рот не ответил: {seen['tail']}"


def test_stalled_batch_is_written_as_queue_for_the_pult(tmp_path, monkeypatch):
    """§0.17: застрявшее записывается в файл гео ОДНОЙ формой — это очередь, а не отчёт.

    ⛔ Именно эту запись читает пульт (`pipeline_state` → шаг 0, «сперва брак») и ставит гео
    на перепрогон. Нет записи — гео выглядит сделанным, и упавшее не встаёт заново никогда.
    Поля `flies` и `family` обязательны: читатели берут их, и на чужой форме пульт молча
    показывал «0 мух» и «?».
    """
    _v, _s, _p, seen = _corpus(
        monkeypatch, tmp_path, VISA, deals=["Сроки и рассмотрение"], stall=True
    )
    assert seen["fails"], "застряло, а очередь пустая"
    rec = seen["fails"][0]
    assert rec["flies"] == len(VISA), rec
    assert rec["family"] == "visa", rec


def test_number_from_the_rot_does_not_kill_the_batch(tmp_path, monkeypatch):
    """Рот отдал номер ЧИСЛОМ, без кавычек — пачка обязана разобраться, а не упасть.

    ⛔ Промпт просит номер в кавычках (`"<номер дела или 0>"`), и код верил этому буквально:
    `(m.get(...) or "").strip()`. На числе это AttributeError — падал весь прогон гео вместе
    с уже оплаченными вызовами. Здесь зовём САМ разбор, а не подменяем его: остальные сторожа
    подменяют проход Б целиком и эту строку не трогают.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        facet, "call", lambda *a, **k: {"map": {"0": 1, "1": "1", "2": 0, "3": "9"}}
    )
    by_id = {f["id"]: f for f in VISA}
    out, stalled = facet.assign_to_deals(["v1", "v2", "v3", "v4"], by_id, ["Дело"])
    assert out == {"Дело": ["v1", "v2"]}, out  # число и строка равноправны
    assert stalled == [], stalled  # рот ответил — это не «не доехало»


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
    # ⛔ Мух РОВНО столько, чтобы дело дотянуло до порога страницы: иначе сторож остался бы
    # зелёным из-за порога, а не из-за барьера имени, и мутация «снять барьер» не покраснела.
    views, _s, _p, seen = _corpus(
        monkeypatch,
        tmp_path,
        VISA,
        deals=["Прочее"],
        mapping={"v1": "Прочее", "v2": "Прочее", "v3": "Прочее", "v4": "Прочее"},
    )
    assert views == {}, views
    assert {"v1", "v2", "v3", "v4"} <= set(seen["tail"]), seen["tail"]


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
    src = (LEGACY / "facet.py").read_text(encoding="utf-8")
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


def test_control_entry_costs_one_call(tmp_path, monkeypatch):
    """Точечный вход для КОНТРОЛЯ: одна пара «гео × раздел» — РОВНО один вызов рта.

    ⛔ Повод — сухой прогон 19.08. Я оценивал контроль метода в «4 вызова», а проход А в
    боевом прогоне идёт по ВСЕМ разделам гео: замер дал 5–13 вызовов А и 6–31 вызов Б на
    одно гео, 87 на четыре. Без этого входа «проверить метод на четырёх парах» означало бы
    боевой прогон четырёх стран вместо четырёх вызовов.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out_facet").mkdir()
    visa = tax.SHELF_NAMES[[k for k, _n, _d in tax.SHELVES].index("visa")]
    tr = tax.SHELF_NAMES[[k for k, _n, _d in tax.SHELVES].index("transport")]
    import json as _json

    def _v(z, n, shelf):
        return {
            "zadacha": z,
            "shelf": shelf,
            "items": [{"id": f"{z}{i}", "text": "Текст."} for i in range(n)],
        }

    (tmp_path / "out_facet" / "gr.json").write_text(
        _json.dumps(
            {
                "geo": "gr",
                "views_by_task": [
                    _v("Сроки визы", 9, visa),
                    _v("Отказы", 6, visa),
                    _v("Паромы", 5, tr),  # ДРУГОЙ раздел — его трогать не должны
                ],
                "shelves": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen = []

    def fake_deals(mass, fails=None, family=None):
        seen.append((family, dict(mass)))
        return ["Сроки и отказы"], True

    monkeypatch.setattr(facet, "carve_deals", fake_deals)
    out = facet.deals_for_pair("gr", "visa", [])
    assert out == ["Сроки и отказы"], out
    assert len(seen) == 1, f"вызовов рта {len(seen)}, ожидался один"
    family, mass = seen[0]
    assert family == "gr/visa", family
    assert set(mass) == {"Сроки визы", "Отказы"}, mass  # паромов тут быть не должно
    assert mass["Сроки визы"] == 9, mass  # масса = абзацы, по ней рот и сводит


def test_control_entry_is_in_cli():
    """Контроль запускается командой, а не из моей головы: у facet есть флаг."""
    src = (LEGACY / "facet.py").read_text(encoding="utf-8")
    assert '"--deals-only" in sys.argv' in src, "флаг не разбирается"
    assert "deals_for_pair(geo" in src, "флаг ничего не зовёт"
