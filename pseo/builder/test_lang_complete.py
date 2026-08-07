"""ЯЗЫК ЛИБО ПОЛНЫЙ, ЛИБО ОТСУТСТВУЕТ. Половинчатого не бывает.

Повод (2026-08-08). Комбайн переводит в 13 языков, а собрать сайт можно было в 4: у
остальных нет ни `COPY`, ни `HOME_ABOUT`, ни `i18n/<lang>.json`, ни имён стран. Это не
гейт и не сбой — просто текст портала никто не написал, а ~22 000 готовых переведённых
страниц из-за этого не выкладывались.

Опасно тут ПОЛОВИНЧАТОЕ состояние: один и тот же недосмотр даёт три разных исхода, и
угадать без прогона нельзя.
  `_buildable_langs()` = COPY ∩ i18n — язык в COPY без i18n молча не собирается;
  `build_home` читает `HOME_ABOUT[lang]` ПРЯМЫМ индексом — падает с KeyError;
  `GEO_NAMES` промахивается тихо и печатает код страны («bo» вместо «Bolivien»);
  `N_WORD`/`TYPE_SHORT`/`ROL` тихо съезжают на английский фолбэк.
Поэтому правило одно: объявлен сборным — полон ВЕЗДЕ.

Сети, ключей и БД не требует. Запуск:  python test_lang_complete.py
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import facet_lang as fl  # noqa: E402
import pages as pg  # noqa: E402

REF = "ru"  # эталон полноты: русский заполнялся первым и полнее всех


def ok(cond, what, got=""):
    print("%-56s %-30s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


def i18n_langs():
    return {os.path.basename(p)[:-5] for p in glob.glob(f"{BASE}/i18n/*.json")}


def i18n_of(lang):
    return json.load(open(f"{BASE}/i18n/{lang}.json", encoding="utf-8"))


def flat_keys(d, prefix=""):
    """Ключи вглубь — иначе пропуск ВНУТРИ `cta_pools` не поймается, а он там и живёт."""
    out = set()
    for k, v in d.items():
        if k.startswith("_"):  # _note — служебная памятка, не контент
            continue
        out.add(prefix + k)
        if isinstance(v, dict):
            out |= flat_keys(v, prefix + k + ".")
    return out


if __name__ == "__main__":
    good = True
    have_i18n = i18n_langs()
    buildable = sorted(set(pg.COPY) & have_i18n)

    good &= ok(
        REF in buildable, "эталон (%s) сам сборный" % REF, "сборные: %s" % buildable
    )

    # ── 1. Половинчатых языков нет: COPY и i18n идут ПАРОЙ.
    half = sorted(set(pg.COPY) ^ have_i18n)
    good &= ok(
        not half,
        "1. нет языка с COPY без i18n или наоборот",
        ("половинчатые: %s" % half) if half else "%d языков парой" % len(buildable),
    )

    # ── 2. Каждый сборный язык полон во ВСЕХ словарях, где промах виден или роняет.
    ref_copy = {
        k for k in pg.COPY[REF] if not k.endswith("_w")
    }  # `*_w` — ru-only склонения
    ref_ha = set(pg.HOME_ABOUT[REF])
    ref_geo = set(pg.GEO_NAMES[REF])
    ref_types = set(pg.TYPE_SHORT[REF])
    ref_i18n = flat_keys(i18n_of(REF))
    for lang in buildable:
        bad = {}
        for nm, ref, got in (
            ("COPY", ref_copy, set(pg.COPY[lang])),
            ("HOME_ABOUT", ref_ha, set(pg.HOME_ABOUT.get(lang, {}))),
            ("GEO_NAMES", ref_geo, set(pg.GEO_NAMES.get(lang, {}))),
            ("TYPE_SHORT", ref_types, set(pg.TYPE_SHORT.get(lang, {}))),
            ("i18n", ref_i18n, flat_keys(i18n_of(lang))),
        ):
            miss = sorted(ref - got)
            if miss:
                bad[nm] = miss[:3]
        good &= ok(
            not bad,
            "2. %s полон везде" % lang,
            str(bad) if bad else "COPY/HOME_ABOUT/GEO/типы/i18n",
        )
        good &= ok(
            lang == REF or lang in pg.N_WORD,
            "   %s: форма счётчика задана" % lang,
            str(pg.N_WORD.get(lang, "— англ. фолбэк")),
        )

    # ── 3. Пулы CTA одинаковой длины. Рендер выбирает вариант по хэшу от пути (`_pick`),
    #    поэтому короткий пул — не «меньше вариантов», а ДРУГОЙ вариант на той же странице.
    ref_pools = i18n_of(REF)["cta_pools"]
    for lang in buildable:
        p = i18n_of(lang)["cta_pools"]
        diff = {
            k: (len(v), len(p.get(k, [])))
            for k, v in ref_pools.items()
            if isinstance(v, list) and len(p.get(k, [])) != len(v)
        }
        good &= ok(
            not diff, "3. %s: пулы CTA той же длины" % lang, str(diff) or "ровно"
        )

    # ── 4. Роли сущностей: фолбэк англ. и это ЗАЯВЛЕНО — проверяем, что он существует,
    #    иначе KeyError на первом же языке без ROL.
    good &= ok("en" in fl.ROL, "4. англ. фолбэк ролей на месте")
    good &= ok(
        all(set(fl.ROL[x]) == set(fl.ROL["en"]) for x in fl.ROL),
        "   у всех заданных ролей одинаковый набор",
        str({x: len(fl.ROL[x]) for x in fl.ROL}),
    )

    # ── 5. Незаполненный язык обязан быть НЕ сборным: иначе сайт выйдет полупустым, а
    #    прогон отрапортует успех — та же болезнь «шаг считает не то».
    unfilled = [x for x in fl.LANG_NAME if x not in pg.COPY]
    good &= ok(
        all(x not in buildable for x in unfilled),
        "5. незаполненные языки не сборные",
        "переводим %d, собираем %d" % (len(fl.LANG_NAME), len(buildable)),
    )

    print("\nVERDICT:", "OK — сборные языки полны" if good else "FAIL")
    sys.exit(0 if good else 1)
