"""lang_runner.py — НЕПРЕРЫВНЫЙ билдер данных на ВСЕ языки. Без остановки.
Языки берёт САМ из target_languages корпуса (union, минус ru=база). Для каждого языка × гео
гонит facet_lang.py (перевод текста+меток). Квоту держит МОЗГ (keybroker) — губернатора/окна нет.
Резюмируемый (скип готовых out_facet_<lang>/<geo>.json). facet_lang вернул 3 (перевод провалился:
мозг на капе / 429) — досыпаем и продолжаем. Всё переведено — чек раз в час на новые данные/гео.
Живёт под systemd.
"""

import glob
import json
import os
import subprocess
import sys
import time

# ДУБЛЬ ДЛЯ КОМБАЙНА: питон СВОЙ (в контейнере хостового venv нет), а рот facet_lang —
# из своего же каталога дублей, не из /root/pseo_builder.
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/root/pseo_builder"  # данные хоста через маунт (out_facet_*, флаги, статус)
STOP = f"{DATA}/LANG_RUNNER_STOP"
STATUS = f"{DATA}/lang_runner_status.json"


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
        os.path.basename(f)[:-5] for f in glob.glob(f"{DATA}/out_facet/*.json")
    )


_fresh = {}  # (path, mtime) → bool; файл не менялся — не перечитываем (их 36×13)


def done(geo, lang):
    """Готово = файл ЕСТЬ и в НОВОМ формате (несёт groups — укладка 0.10).
    Старый формат (до-карвовый перевод стен) = не готово → пересборка facet_lang."""
    p = f"{DATA}/out_facet_{lang}/{geo}.json"
    if not os.path.exists(p):
        return False
    key = (p, os.path.getmtime(p))
    if key not in _fresh:
        try:
            d = json.load(open(p, encoding="utf-8"))
            vs = d.get("views_by_task", [])
            _fresh[key] = (not vs) or any("groups" in v for v in vs)
        except Exception:
            _fresh[key] = False
    return _fresh[key]


def save_status(obj):
    tmp = STATUS + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, STATUS)


def build_one(geo, lang):
    """facet_lang.py geo lang. rc: 0=ок/скип, 3=перевод провалился (мозг на капе / 429) → ретрай."""
    r = subprocess.run(
        [PY, f"{HERE}/facet_lang.py", geo, lang],
        cwd=DATA,
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
        if not pending:  # всё переведено → ВЫХОД (канон PROCESSES.md: прогон умирает
            # по завершении, вечных ждунов нет; новые данные = новый запуск по отмашке)
            save_status(
                {"state": "all-done", "языков": len(langs), "гео": len(gs), "ts": now()}
            )
            return
        stalled = False
        for geo, lang in pending:
            if os.path.exists(STOP):
                save_status({"state": "stopped", "ts": now()})
                return
            rc = build_one(geo, lang)
            if (
                rc == 3
            ):  # мозг на капе/исчерпании (call вернул None) → гео отложен, досыпаем
                save_status(
                    {"state": "cap-sleep", "осталось": len(pending), "ts": now()}
                )
                stalled = True
                break
        time.sleep(1800 if stalled else 30)


if __name__ == "__main__":
    main()
