"""Сторож канона §0.13: МЕТКА ВИДА = ЗАГОЛОВОК СТРАНИЦЫ, сборная метка — брак нарезки.

Повод и числа. Замер 13.08: 10 видов из 1 889 остались без раздела, и рот был прав —
по метке «Прочее», «Общие советы для туристов и экспатов», «Прочие вопросы (связь, авто)»
раздел выбрать НЕЛЬЗЯ. Замер 14.08 по всему корпусу: таких меток 26 из 1 889, за ними 194
абзаца; у трёх гео меткой страницы стало имя раздела («Транспорт и логистика», 21+11+8
абзацев) — это откат карва, замаскированный под тему.

⛔ ГЛАВНОЕ: правило проверено НА КОНТРОЛЕ (все 1 889 живых меток), и первая версия его НЕ
ПРОШЛА. Я считал перечнем всё с тремя запятыми, и правило валило законные страницы:
«Условия аренды авто: страховка, состояние авто, требования» (8 абзацев), «Сравнение и
выбор локальных островов: Тодду, Фериду…» (11), «Покупка текстиля, одежды, кашемира…» (5).
Форма «<одно дело>: пример, пример» нормальна. Правило сужено до того, что на контроле не
дало ни одной ложной сработки, и в этом файле обе половины закреплены: и что отсекается,
и что отсекаться НЕ ДОЛЖНО.

Сети, ключей и БД не требует.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import facet  # noqa: E402
import tail_taxonomy as tax  # noqa: E402

# Живые метки корпуса, которые ОБЯЗАНЫ отсеяться (с их массой в абзацах).
JUNK = [
    ("Прочее", 4),
    ("Прочие вопросы (связь, авто)", 5),
    ("Общие советы для туристов и экспатов", 4),
    ("Общие советы по безопасности и ситуационной осведомлённости", 23),
    ("Полезная информация для путешественников: связь, деньги, логистика", 12),
    ("Адаптация и жизнь в стране", 4),
    ("Разное (бытовые товары, багаж, прочее)", 9),
    ("Общая информация об использовании валюты в Монголии", 4),
    ("Транспорт и логистика", 21),  # имя РАЗДЕЛА вместо запроса — откат карва
    ("Особенности", 3),
]

# Живые метки корпуса, которые отсеиваться НЕ ДОЛЖНЫ. Половина из них — те самые, на
# которых первая версия правила и провалилась.
GOOD = [
    "Сроки оформления шенгенской визы в Грецию",
    "Условия аренды авто: страховка, состояние авто, требования",
    "Сравнение и выбор локальных островов: Тодду, Фериду, Гулхи",
    "Покупка текстиля, одежды, кашемира и изделий из кожи",
    "Требования к бронированию жилья для визы (имена, логичность, статус)",
    "Экскурсии: киты, дельфины, вертолет и острова",
    "Особенности процедур в Рио-де-Жанейро",
    "Электричество и бытовые стандарты",
    "Культура курения",
    "Советы по отдыху на Крите",
]


def test_junk_labels_are_rejected_with_a_reason():
    """Каждая сборная метка отсеивается, и причина — строкой, а не булевым флагом.

    Причина уходит в лог: молчаливый отсев — то, из-за чего 33 пустых хаба год жили
    незамеченными.
    """
    for z, _n in JUNK:
        why = tax.bad_label(z)
        assert why, f"НЕ отсеклась: {z!r}"
        assert isinstance(why, str) and len(why) > 5, (z, why)


def test_real_labels_are_not_touched():
    """⛔ Контрольная половина. Ложная сработка тут дороже пропуска: она стирает готовую
    страницу с живым содержимым."""
    for z in GOOD:
        assert tax.bad_label(z) is None, f"ложная сработка: {z!r} → {tax.bad_label(z)}"


def test_label_equal_to_a_shelf_name_is_a_rubric():
    """Метка, совпадающая с именем раздела, — рубрика, а не запрос. Так выглядит откат
    карва: он даёт мухе имя её исходной широкой рубрики, и снаружи это похоже на тему.
    """
    for _k, name, _d in tax.SHELVES:
        assert tax.bad_label(name), f"имя раздела прошло как метка: {name!r}"


def test_rule_lives_in_one_place():
    """Словарь запрета — в ОДНОМ модуле, остальные его ЗОВУТ.

    Правило в двух копиях — болезнь этого проекта: правка не доезжает, а тест зелёный.
    ⚠️ Первая версия этой проверки искала слово «прочее» рядом с вызовом и краснела на
    собственном комментарии. Проверяем имя константы-словаря, а не текст вокруг.
    """
    holders = [
        f.name
        for f in sorted(HERE.glob("*.py"))
        if not f.name.startswith("test_")
        and "_JUNK_LABEL = (" in f.read_text(encoding="utf-8")
    ]
    assert holders == ["tail_taxonomy.py"], f"словарь живёт не в одном месте: {holders}"
    for mod in ("facet.py", "pages.py"):
        src = (HERE / mod).read_text(encoding="utf-8")
        assert "bad_label(" in src, f"{mod}: правило не зовётся"
        assert "_JUNK_LABEL" not in src, f"{mod}: своя копия словаря"


def test_carve_prompt_forbids_grab_bag_names():
    """Промпт карва обязан запрещать такие имена: иначе рот рождает их заново каждый прогон,
    а код только подчищает. Лечить причину, а не симптом."""
    p = facet.CARVE_SYS
    assert "ЗАПРЕЩЕНЫ" in p, "запрета нет в промпте"
    for w in ("Прочее", "Общие советы", "Адаптация"):
        assert w in p, f"в запрете не назван пример: {w}"
    assert "ЗАГОЛОВОК СТРАНИЦЫ" in p, "не сказано, ЧЕМ станет имя подпункта"


def test_unknown_key_value_is_printed_not_only_counted():
    """При неизвестном ключе раздела печатается ЧТО пришло, а не только сколько.

    Замер 13.08: счётчик показывал «неопознанный ключ 1», и понять, что рот ответил
    `prochee`, было нельзя — пришлось лезть в данные руками.
    """
    src = (HERE / "facet.py").read_text(encoding="utf-8")
    assert "unknown_keys" in src, "значение ключа нигде не собирается"
    assert "','.join(unknown_keys[:5])" in src, "значение не попадает в печать"


def test_page_is_not_built_from_a_grab_bag_label(tmp_path):
    """ПОВЕДЕНИЕ, а не текст кода: из сборной метки страница не собирается, причина в лог.

    Барьер в сборке нужен отдельно от отсева в карве: в корпусе уже лежат 26 таких меток,
    а файлы гео пересобираются не все и не сразу.
    """
    import io
    import json
    import os
    from contextlib import redirect_stdout

    import pages as pg

    def _v(z, n, key):
        items = [
            {"id": f"{key}{i}", "text": f"Совет {i}. Подробность."} for i in range(n)
        ]
        return {
            "zadacha": z,
            "key": key,
            "items": items,
            "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items],
        }

    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "out_facet").mkdir()
    (tmp_path / "out_facet" / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "views_by_task": [
                    _v("Сроки оформления шенгенской визы", 9, "visa-terms"),
                    _v("Общие советы для туристов и экспатов", 6, "general-tips"),
                ],
                "shelves": [],
                "prochee": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), str(out)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            pg.build_geo("gr", "ru")
    finally:
        pg.BUILT, pg.DATA = built, data
    paths = {
        json.load(open(f"{out}/{f}", encoding="utf-8")).get("path")
        for f in os.listdir(out)
    }
    assert "/ru/gr/visa-terms/" in paths, paths
    assert "/ru/gr/general-tips/" not in paths, "страница из сборной метки собралась"
    log = buf.getvalue()
    assert "сборная метка" in log, f"причина не напечатана: {log!r}"


def test_carve_drops_grab_bag_view_and_keeps_flies_in_tail(monkeypatch):
    """Основное место отсева — ВЫХОД КАРВА: вид не записывается, а его мухи уходят в хвост.

    Так содержимое не теряется: 194 абзаца из 26 сборных меток станут заметками разделов,
    а не страницами с заголовком «Прочее». Рот не зовём — подменяем его ответ.
    """
    # `build_views_by_carve` принимает СПИСОК размеченных мух (как отдаёт facet_one)
    flies = [
        {
            "id": f"f{i}",
            "perevod": f"Живой совет номер {i} про важное.",
            "sushnosti": [],
            "mesto": None,
            "uslovie": None,
            "zadachi": ["Визы: сроки оформления"],
        }
        for i in range(8)
    ]
    monkeypatch.setattr(
        facet,
        "carve_family",
        lambda fids, by_id, fails=None, family=None: [
            {"name": "Общие советы для туристов", "ids": list(fids)[:4]},
            {"name": "Сроки оформления визы", "ids": list(fids)[4:]},
        ],
    )
    seen_tail = {}

    def fake_tail(fids, by_id, fails=None):
        seen_tail["ids"] = list(fids)
        return {}, []

    monkeypatch.setattr(facet, "assign_tail", fake_tail)
    monkeypatch.setattr(facet, "MIN_CARVE", 2)
    views, _shelves, _prochee = facet.build_views_by_carve(flies, fails=[])
    assert "Сроки оформления визы" in views, list(views)
    assert "Общие советы для туристов" not in views, "сборный вид записан"
    assert (
        len(seen_tail.get("ids") or []) == 4
    ), seen_tail  # мухи сборного вида → в хвост


def test_dropped_paragraphs_land_in_their_section_tail(tmp_path):
    """⛔ ЗАМЕР, КОТОРЫЙ ПОЙМАЛ МОЮ ЖЕ НЕПРАВДУ (14.08, `br`).

    Я написал «содержимое не теряется» и не проверил. Барьер по метке убирает страницу
    СРАЗУ, а мухи возвращаются в хвост только на следующем прогоне карва — то есть за
    ключи и не сегодня. Факт по Бразилии: за 9 отсеянными метками 79 абзацев, из них 63
    лежат и на других страницах (мухи мульти-лейбл), а **16 не лежат нигде** — они бы
    просто исчезли с сайта.

    Поэтому сборка сама кладёт такие абзацы в хвост ИХ РАЗДЕЛА (раздел у вида уже
    проставлен ртом `assign`), и это правило проверяется здесь. Остаток — метки, у которых
    раздела нет вовсе, — печатается числом: потеря обязана быть видимой.
    """
    import io
    import json
    import os
    from contextlib import redirect_stdout

    import pages as pg

    def _v(z, n, key, shelf=None):
        items = [
            {"id": f"{key}{i}", "text": f"Уникальный абзац {key}-{i} про важное."}
            for i in range(n)
        ]
        v = {
            "zadacha": z,
            "key": key,
            "items": items,
            "groups": [{"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items],
        }
        if shelf:
            v["shelf"] = shelf
        return v

    visa = tax.SHELF_NAMES[[k for k, _n, _d in tax.SHELVES].index("visa")]
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "out_facet").mkdir()
    (tmp_path / "out_facet" / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "views_by_task": [
                    _v("Сроки оформления шенгенской визы", 9, "visa-terms", visa),
                    _v("Общие советы по подаче", 5, "general-tips", visa),  # отсев
                    _v("Прочее", 4, "misc"),  # отсев и раздела нет
                ],
                "shelves": [
                    {
                        "shelf": visa,
                        "key": "visa",
                        "items": [
                            {"id": f"s{i}", "text": f"Заметка {i}."} for i in range(3)
                        ],
                    }
                ],
                "prochee": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), str(out)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            pg.build_geo("gr", "ru")
    finally:
        pg.BUILT, pg.DATA = built, data

    pages_built = {
        json.loads((out / f).read_text(encoding="utf-8")).get("path")
        for f in os.listdir(out)
    }
    blob = "".join((out / f).read_text(encoding="utf-8") for f in os.listdir(out))
    # ⚠️ Проверяем АДРЕСА, а не подстроку: слаг «general-tips» встречается и в самих
    # спасённых текстах фикстуры, и первая версия этой проверки краснела на верном коде.
    assert "/ru/gr/general-tips/" not in pages_built, pages_built
    assert "/ru/gr/visa-terms/" in pages_built, pages_built
    for i in range(5):  # абзацы отсеянной метки С разделом обязаны быть на сайте
        assert f"general-tips-{i}" in blob, f"абзац general-tips-{i} исчез с сайта"
    log = buf.getvalue()
    # у «Прочее» раздела нет — эти абзацы пристроить нечем, и это должно быть СКАЗАНО
    assert "не пристроено" in log and "без раздела вовсе: 4" in log, log


def test_rescue_works_when_the_tail_has_dedup_groups(tmp_path):
    """Та же спасательная операция, но на ветке хвоста С ДЕДУП-ГРУППАМИ (аккордеон).

    ⛔ Мутация «спасённые абзацы не доезжают до страницы раздела» оставалась ЗЕЛЁНОЙ: в
    первой фикстуре хвост был без групп, и код шёл другой ветвью. У живых гео встречаются
    обе (у `gr` 24 заметки и 0 групп, а после прогона дедупа группы появятся), значит
    проверять надо обе.
    """
    import json
    import os

    import pages as pg

    visa = tax.SHELF_NAMES[[k for k, _n, _d in tax.SHELVES].index("visa")]
    items = [{"id": f"t{i}", "text": f"Заметка хвоста {i}."} for i in range(3)]
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "out_facet").mkdir()
    (tmp_path / "out_facet" / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "views_by_task": [
                    {
                        "zadacha": "Общие советы по подаче",
                        "key": "general-tips",
                        "shelf": visa,
                        "items": [
                            {"id": f"g{i}", "text": f"Спасаемый абзац номер {i}."}
                            for i in range(5)
                        ],
                        "groups": [
                            {"rep": f"g{i}", "ids": [f"g{i}"], "n": 1} for i in range(5)
                        ],
                    },
                    # ⛔ Уцелевший разбор нужен именно здесь: без него сборка идёт другой
                    # ветвью, и мутация «спасённые абзацы не доезжают» оставалась зелёной.
                    # Живой случай — ровно такой: в разделе есть и годные метки, и сборные.
                    {
                        "zadacha": "Сроки оформления шенгенской визы",
                        "key": "visa-terms",
                        "shelf": visa,
                        "items": [
                            {"id": f"v{i}", "text": f"Годный абзац номер {i}."}
                            for i in range(6)
                        ],
                        "groups": [
                            {"rep": f"v{i}", "ids": [f"v{i}"], "n": 1} for i in range(6)
                        ],
                    },
                ],
                "shelves": [
                    {
                        "shelf": visa,
                        "key": "visa",
                        "items": items,
                        "groups": [
                            {"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items
                        ],
                    }
                ],
                "prochee": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), str(out)
    try:
        pg.build_geo("gr", "ru")
    finally:
        pg.BUILT, pg.DATA = built, data
    sec = next(
        json.loads((out / f).read_text(encoding="utf-8"))
        for f in os.listdir(out)
        if json.loads((out / f).read_text(encoding="utf-8")).get("path")
        == "/ru/gr/s/visa/"
    )
    text = json.dumps(sec, ensure_ascii=False)
    for i in range(5):
        assert f"Спасаемый абзац номер {i}." in text, f"абзац {i} потерян на аккордеоне"
    assert "Заметка хвоста 0." in text, "свой хвост раздела пропал"
    assert "/ru/gr/visa-terms/" in text, "плитка уцелевшего разбора пропала со раздела"


def test_question_contour_is_gone(tmp_path):
    """Вопрос-контур `/q/` снесён (решение юзера 19.08: «убрать и забыть»).

    Основание фактом: 0 показов и 0 запросов за три месяца по Search Console при 550
    показах у разборов; 5 стран, 20 собираемых страниц, 12 выложенных. Держать третий
    контур в дереве ради надежды — это ружьё на стене.
    """
    import json
    import os

    import pages as pg

    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "out_facet").mkdir()
    (tmp_path / "out_questions").mkdir()
    # данные вопросов НА МЕСТЕ — и всё равно ни одной страницы из них быть не должно
    (tmp_path / "out_questions" / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "groups": [
                    {
                        "tema": "Вопросы про визы",
                        "key": "visa-questions",
                        "questions": [f"Вопрос {i}?" for i in range(6)],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    items = [{"id": f"i{i}", "text": f"Совет {i}. Подробность."} for i in range(9)]
    (tmp_path / "out_facet" / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "views_by_task": [
                    {
                        "zadacha": "Сроки оформления визы",
                        "key": "visa-terms",
                        "items": items,
                        "groups": [
                            {"rep": x["id"], "ids": [x["id"]], "n": 1} for x in items
                        ],
                    }
                ],
                "shelves": [],
                "prochee": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    built, data = pg.BUILT, pg.DATA
    pg.BUILT, pg.DATA = str(tmp_path), str(out)
    try:
        pg.build_geo("gr", "ru")
    finally:
        pg.BUILT, pg.DATA = built, data
    paths = {
        json.loads((out / f).read_text(encoding="utf-8")).get("path")
        for f in os.listdir(out)
    }
    assert "/ru/gr/visa-terms/" in paths, paths
    assert not [p for p in paths if p and "/q/" in p], paths
    hub = next(
        json.loads((out / f).read_text(encoding="utf-8"))
        for f in os.listdir(out)
        if f.endswith("gr_hub.json")
    )
    urls = [t.get("url", "") for t in hub["tiles"]]
    assert not [u for u in urls if u.endswith("/q/")], urls
    src = (HERE / "pages.py").read_text(encoding="utf-8")
    assert "_ques_dir" not in src, "остался читатель данных вопросов"
    # ⛔ Проверяем ПОСТРОЕНИЕ адреса, а не упоминание: первая версия ловила `/q/` в моём
    # же комментарии о сносе и краснела на верном коде.
    assert "{geo}/q/" not in src, "контур ещё строит адреса вопросов"
