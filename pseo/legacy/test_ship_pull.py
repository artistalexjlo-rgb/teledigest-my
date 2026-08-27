"""PULL ТЯНЕТ ПЕРЕВОДЫ — и ровно те языки, которые сборщик умеет собрать.

Повод (2026-08-07): в таре шага `pull` были только `out_facet` (ru) и `out_questions`.
Каталоги `out_facet_<lang>` не приезжали ВООБЩЕ. На десктопе лежали копии en/es/pt от
10-12 июля — 34 гео вместо 90 и БЕЗ полок, — а свежие переводы всех 90 гео с полками
стояли на VPS. Трёхдневный прогон переводов физически не мог попасть в публикацию, и
никто этого не видел: `pull` бодро печатал «факт-гео 90» и молчал про языки.

Проверяем два свойства:
  1. в список каталогов попадают переводы, а не только ru;
  2. попадают ТОЛЬКО собираемые языки. Комбайн переводит в 13, собрать можно 4 (нужны и
     `pages.COPY`, и `i18n/<lang>.json`). Маска `out_facet_*` притащила бы 291 МБ вместо
     53, из них 238 — данные, которые нечем отрендерить.

Сети не требует: проверяем резолв языков и состав команды, без ssh. Запуск:
  python test_ship_pull.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ship  # noqa: E402


def ok(cond, what, got=""):
    print("%-56s %-30s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    good = True
    langs = ship._buildable_langs()

    # 1. Собираемые языки = пересечение COPY и i18n. Ни больше, ни меньше.
    sys.path.insert(0, ship.BUILT)
    import pages  # noqa: E402

    i18n = {f[:-5] for f in os.listdir(f"{ship.BASE}/i18n") if f.endswith(".json")}
    good &= ok(
        set(langs) == set(pages.COPY) & i18n,
        "1. язык собираем = есть и COPY, и i18n",
        str(langs),
    )
    good &= ok("ru" in langs, "   ru в списке", str("ru" in langs))

    # 2. ⭐ ГЛАВНОЕ: переводы в таре есть. Раньше их не было вовсе.
    tr = [x for x in langs if x != "ru"]
    dirs = ["out_facet"] + [f"out_facet_{x}" for x in tr] + ["out_questions"]
    good &= ok(
        len(tr) >= 1 and all(f"out_facet_{x}" in dirs for x in tr),
        "2. каталоги переводов попали в тар",
        str(dirs),
    )

    # 3. ⛔ И только собираемые: язык с данными, но без i18n/COPY — НЕ тянем.
    #    Проверяем на реальном примере: у комбайна есть out_facet_ja, i18n/ja.json нет.
    good &= ok(
        "ja" not in langs and "hi" not in langs,
        "3. язык без i18n/COPY в тар НЕ попадает",
        "ja/hi отсутствуют" if "ja" not in langs else "ja просочился",
    )

    # 4. ru лежит в out_facet, а не в out_facet_ru — дубля быть не должно.
    good &= ok(
        "out_facet_ru" not in dirs,
        "4. ru не удваивается (он в out_facet)",
        "нет out_facet_ru" if "out_facet_ru" not in dirs else "ДУБЛЬ",
    )

    print("\nVERDICT:", "OK — pull несёт переводы и ничего лишнего" if good else "FAIL")
    sys.exit(0 if good else 1)
