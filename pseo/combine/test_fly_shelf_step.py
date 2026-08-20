"""Сторож пульта: шаг «раздел мухам» — ось нарезки запускается КНОПКОЙ, не с десктопа.

Ось нарезки (канон §0.15) держится на том, что у каждой мухи есть раздел. Если этот шаг
нельзя нажать из пульта, ось живёт только у ассистента на машине — а заказ юзера дословно:
«мне нужен путь, работающий без тебя».

⛔ Работа шага считается по СКЛАДУ РАЗМЕТКИ (`tags/`), а не по корпусу: раздел ставится
МУХЕ. Счётчик, который мерил бы не то, — та же болезнь, что весь день 13.08: шаг зовёт
сделанное или молчит о несделанном.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "builder")]

os.environ.setdefault("COMBINE_BOT_TOKEN", "test")
os.environ.setdefault("ADMIN_ID", "1")

import bot  # noqa: E402


def _tags(root, geo, flies):
    d = root / "tags"
    d.mkdir(exist_ok=True)
    (d / f"{geo}.json").write_text(
        json.dumps(flies, ensure_ascii=False), encoding="utf-8"
    )


def _fly(fid, shelf_key=None):
    r = {"id": fid, "perevod": f"Совет {fid}.", "zadachi": ["Дело"], "sushnosti": []}
    if shelf_key:
        r["shelf_key"] = shelf_key
    return r


def test_flies_without_section_are_work(tmp_path, monkeypatch):
    """Муха без раздела — работа шага. Муха с разделом — нет: иначе шаг зовёт сделанное."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a"), _fly("b", "visa")])
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == ["gr"], st["no_fly_shelf"]
    assert st["no_fly_shelf_n"] == 1, "с разделом не должна считаться работой"


def test_all_marked_means_no_work(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a", "visa"), _fly("b", "prochee")])
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == [] and st["no_fly_shelf_n"] == 0


def test_fails_file_is_not_counted_as_geo(tmp_path, monkeypatch):
    """⚠️ Рядом с разметкой лежит `<geo>_fails.json` — это не гео, и работой быть не может."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a")])
    # ⛔ Фикстура нарочно в ХУДШЕЙ форме — списком записей: боевой файл сбоев это словарь
    # {id: счётчик}, и на словаре мутация «не отсеивать _fails» оставалась зелёной, потому
    # что перебор словаря даёт строки. Отсев по имени обязан работать при любой форме.
    (tmp_path / "tags" / "gr_fails.json").write_text(
        json.dumps([{"id": "x"}]), encoding="utf-8"
    )
    st = bot.pipeline_state()
    assert st["no_fly_shelf"] == ["gr"], st["no_fly_shelf"]
    assert st["no_fly_shelf_n"] == 1, st["no_fly_shelf_n"]


def test_step_carries_the_job_and_says_it(tmp_path, monkeypatch):
    """Шаг обязан и выдать задание, и сказать о нём в подписи — иначе кнопка пустая."""
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    _tags(tmp_path, "gr", [_fly("a")])
    st = bot.pipeline_state()
    step = next(s for s in bot.pipeline_steps(st) if s["kind"] == "assign")
    assert ("flyshelf", "gr") in step["jobs"], step["jobs"]
    assert "мухи" in step["label"], step["label"]


def test_menu_calls_the_new_flag():
    """Кнопка зовёт тот же `facet.py` с флагом оси. Проверяем СМЫСЛ, не позиции в списке."""
    assert "flyshelf" in bot.MENU
    argv = bot.MENU["flyshelf"][1]
    assert any(a.endswith("facet.py") for a in argv), argv
    assert "--assign-flies" in argv, argv
    assert "{geo}" in argv, argv


def test_control_button_takes_a_pair(monkeypatch):
    """Кнопка КОНТРОЛЯ: «Дела раздела» берёт пару «гео:раздел» и зовёт точечный вход.

    ⛔ Повод — забор 19.08: боевые рты запускает ЮЗЕР из пульта, где есть СТОП и отчёт в
    чат. Я добавил только флаг командной строки, то есть контроль запускать было нечем.
    Проверяем СМЫСЛ команды и разбор пары, а не позиции в списке.
    """
    assert "deals" in bot.MENU
    argv = bot.MENU["deals"][1]
    assert any(a.endswith("facet.py") for a in argv), argv
    assert "--deals-only" in argv and "{geo}" in argv and "{shelf}" in argv, argv
    # разбор пары «гео:раздел» — то же правило, что применит `start()`
    geo, shelf = "br:finance".split(":", 1)
    built = [a.replace("{geo}", geo).replace("{shelf}", shelf) for a in argv]
    assert "br" in built and "finance" in built, built
    assert "{geo}" not in " ".join(built) and "{shelf}" not in " ".join(built), built


def test_pair_parsing_lives_before_the_stale_shelf_branch():
    """Пара разбирается ДО общей логики `{shelf}`.

    Иначе кнопка контроля пошла бы искать УСТАРЕВШУЮ полку (`stale_shelf`) и отказалась бы
    работать на живой таксономии — а контроль нужен именно на ней.
    """
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    i = src.index('if geo and ":" in geo:')
    j = src.index('elif any("{shelf}" in a for a in argv):')
    assert i < j, "разбор пары оказался после ветки про устаревшую полку"


def test_stalled_geo_goes_first_in_the_queue(tmp_path, monkeypatch):
    """Вторая половина канона §0.17: запись о несделанном обязана ПОДНЯТЬ гео в очередь.

    ⛔ Пара к сторожу в билдере (`test_stalled_batch_is_written_as_queue_for_the_pult`): там
    проверено, что запись появляется, здесь — что пульт по ней ставит гео на перепрогон и
    называет ЧИСЛО мух. Без этой половины запись есть, а очередь пустая, и упавшее не встаёт.
    """
    monkeypatch.setattr(bot, "BRAIN", str(tmp_path))
    d = tmp_path / "out_facet"
    d.mkdir(exist_ok=True)
    (d / "gr.json").write_text(
        json.dumps(
            {
                "geo": "gr",
                "views_by_task": [],
                "shelves": [],
                "fails": [{"step": "deal_assign", "family": "visa", "flies": 7}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    st = bot.pipeline_state()
    hit = [x for x in st["failed"] if x["geo"] == "gr"]
    assert hit, st["failed"]
    assert hit[0]["flies"] == 7, hit[0]
    assert "visa" in hit[0]["what"], hit[0]
    # и эта работа обязана попасть в шаг 0, а не остаться числом в состоянии
    assert any(
        x["geo"] == "gr" and x["broken"] for x in bot.facet_queue(st)
    ), bot.facet_queue(st)
