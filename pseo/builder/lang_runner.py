"""lang_runner.py — НЕПРЕРЫВНЫЙ билдер данных на ВСЕ языки. Без остановки.
Языки берёт САМ из target_languages корпуса (union, минус ru=база). Для каждого языка × гео
гонит facet_lang.py (перевод текста+меток) ПОД GOVERNOR (facet_lang сам чекает pool_state/окно,
возвращает 3 на бюджет). Резюмируемый (скип готовых out_facet_<lang>/<geo>.json). На бюджет —
сон и продолжил. Всё переведено — чек раз в час на новые данные/гео. Живёт под systemd.
"""

import glob
import json
import os
import subprocess
import time

from facet_lang import budget_ok  # для перепроверки на ложный rc3

PY = "/root/embed_ab/venv/bin/python"
HERE = "/root/pseo_builder"
STOP = f"{HERE}/LANG_RUNNER_STOP"
STATUS = f"{HERE}/lang_runner_status.json"


def now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# КАНОН: 14 языков Luky-переводчика (привязка продукта). ru — база (не переводим), остальные 13
# гоним facet_lang'ом. en=текст из ai_lesson (бесплатно), прочие=перевод.
LUKY_14 = ["en", "es", "pt", "zh", "fr", "de", "ja", "ko", "ar", "th", "it", "hi", "tr"]


def target_langs():
    return list(LUKY_14)


def geos():
    return sorted(
        os.path.basename(f)[:-5] for f in glob.glob(f"{HERE}/out_facet/*.json")
    )


def done(geo, lang):
    return os.path.exists(f"{HERE}/out_facet_{lang}/{geo}.json")


def save_status(obj):
    tmp = STATUS + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, STATUS)


def build_one(geo, lang):
    """facet_lang.py geo lang. rc: 0=ок/скип, 3=бюджет/окно (governor стоп)."""
    r = subprocess.run(
        [PY, "facet_lang.py", geo, lang],
        cwd=HERE,
        env={**os.environ, "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
    )
    return r.returncode


def main():
    while True:
        if os.path.exists(STOP):
            save_status({"state": "stopped", "ts": now()})
            return
        langs = target_langs()
        gs = geos()
        pending = [(g, lang) for lang in langs for g in gs if not done(g, lang)]
        total = len(langs) * len(gs)
        save_status(
            {
                "state": "building",
                "языков": len(langs),
                "langs": langs,
                "гео": len(gs),
                "осталось": len(pending),
                "готово": total - len(pending),
                "ts": now(),
            }
        )
        if not pending:  # всё переведено — ждём новых данных/гео
            save_status(
                {"state": "all-done", "языков": len(langs), "гео": len(gs), "ts": now()}
            )
            time.sleep(3600)
            continue
        stalled = False
        for geo, lang in pending:
            if os.path.exists(STOP):
                save_status({"state": "stopped", "ts": now()})
                return
            rc = build_one(geo, lang)
            if (
                rc == 3
            ):  # facet_lang сказал бюджет/окно — но ПЕРЕПРОВЕРИМ (бывает ложно)
                if budget_ok():
                    continue  # ложная тревога — бюджет есть, идём дальше без сна
                save_status(
                    {"state": "budget-sleep", "осталось": len(pending), "ts": now()}
                )
                stalled = True
                break
        time.sleep(1800 if stalled else 30)


if __name__ == "__main__":
    main()
