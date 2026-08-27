"""LASTMOD ПОСТРАНИЧНО и НЕ ВРЁТ.

Повод (2026-08-07): дата в sitemap была РУЧНЫМ аргументом `render.py --all <дата>`. Кто-то
вписал `2026-07-06`, и она месяц ехала во все 2185 адресов — то есть месяц говорила Google
«здесь ничего не менялось», при том что 19-20.07 сайт пересобрали целиком. Обход это
подавляет, а в отчёте GSC 82% висело в «обнаружено, не проиндексировано».

Проверяем два свойства, и второе важнее первого:
  1. дата ставится ПОСТРАНИЧНО и попадает в sitemap;
  2. на НЕИЗМЕНЁННОЙ странице дата НЕ обновляется. Свежий lastmod на неизменной странице —
     ложь поисковику; от таких сигналов он отучается им верить, и тогда правка №1 бесполезна.

Ни сети, ни ключей, ни боевых данных: пишем в свой временный DATA. Запуск:
  python test_lastmod.py
"""

import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pages  # noqa: E402


def ok(cond, what, got=""):
    print("%-56s %-24s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    pages.DATA = tempfile.mkdtemp()
    good = True
    P = "ru_xx_test.json"
    page = {"lang": "ru", "path": "/ru/xx/test/", "title": "T", "h1": "H"}

    # 1. Первая запись — дата сегодняшняя, в обоих форматах.
    pages.write(P, dict(page))
    d1 = json.load(open(f"{pages.DATA}/{P}", encoding="utf-8"))
    good &= ok(
        d1.get("updated_iso") == pages._TODAY_ISO,
        "1. новая страница получает сегодняшнюю дату",
        d1.get("updated_iso"),
    )
    good &= ok(
        d1.get("updated") == pages.UPDATED,
        "   подпись в подвале в формате MM.YYYY",
        d1.get("updated"),
    )

    # 2. ⭐ ГЛАВНОЕ: содержимое то же → дату НЕ переставляем. Подкладываем старую дату и
    #    пишем ТОТ ЖЕ объект: она обязана сохраниться.
    d1["updated_iso"] = "2026-01-15"
    d1["updated"] = "01.2026"
    json.dump(d1, open(f"{pages.DATA}/{P}", "w", encoding="utf-8"), ensure_ascii=False)
    pages.write(P, dict(page))
    d2 = json.load(open(f"{pages.DATA}/{P}", encoding="utf-8"))
    good &= ok(
        d2.get("updated_iso") == "2026-01-15",
        "2. содержимое не изменилось → дата ПРЕЖНЯЯ",
        d2.get("updated_iso"),
    )

    # 3. Содержимое изменилось → дата сегодняшняя.
    pages.write(P, dict(page, h1="ДРУГОЙ заголовок"))
    d3 = json.load(open(f"{pages.DATA}/{P}", encoding="utf-8"))
    good &= ok(
        d3.get("updated_iso") == pages._TODAY_ISO,
        "3. содержимое изменилось → дата сегодняшняя",
        d3.get("updated_iso"),
    )

    # 4. ⭐ НАСТОЯЩИЙ render, а не переписанная копия его логики. Берём РЕАЛЬНУЮ страницу
    #    из боевого data/ (иначе шаблон не отрендерится — полей у него много), кладём в
    #    свой временный BASE и зовём build_all БЕЗ аргумента-даты. В sitemap обязана
    #    оказаться дата СТРАНИЦЫ.
    #    ⛔ Раньше здесь была локальная копия `_lm` — именно так сегодня уже дважды
    #    получалась зелёная фикстура на непроверенном коде. Проверяем то, что исполняется.
    import pathlib
    import shutil

    PSEO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PSEO)
    import render  # noqa: E402

    real = sorted(glob.glob(f"{PSEO}/data/ru_*_hub.json"))
    if not real:
        print("ПРОПУСК проверки 4: в data/ нет реальной страницы для образца")
    else:
        tmp2 = tempfile.mkdtemp()
        os.makedirs(f"{tmp2}/data")
        os.makedirs(f"{tmp2}/out")
        d = json.load(open(real[0], encoding="utf-8"))
        d["updated_iso"] = "2026-01-15"  # заведомо НЕ сегодняшняя и не 07-06
        json.dump(d, open(f"{tmp2}/data/p.json", "w", encoding="utf-8"))
        render.BASE = pathlib.Path(tmp2)
        shutil.copytree(f"{PSEO}/i18n", f"{tmp2}/i18n", dirs_exist_ok=True)
        try:
            render.build_all()  # БЕЗ аргумента: дата должна прийти из страницы
            sm = open(f"{tmp2}/out/sitemap.xml", encoding="utf-8").read()
            good &= ok(
                "<lastmod>2026-01-15</lastmod>" in sm,
                "4. РЕАЛЬНЫЙ render взял дату из страницы",
                "2026-01-15 в sitemap" if "2026-01-15" in sm else sm[:60],
            )
        except Exception as e:
            good &= ok(False, "4. РЕАЛЬНЫЙ render", "%s: %s" % (type(e).__name__, e))

    print("\nVERDICT:", "OK — lastmod постранично и не врёт" if good else "FAIL")
    sys.exit(0 if good else 1)
