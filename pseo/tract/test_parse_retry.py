"""ПАРС-ФЕЙЛ = обычная неудача, а не конец вызова.

Повод (2026-07-27): `call()` на 200-с-непарсящимся-телом возвращал None БЕЗ единого повтора,
хотя на инфраструктурные сбои у него бюджет MAX_FAILS. Цена реальная: один парс-фейл в 10:03
отнял нарезку у 37 мух и поставил гео vn на перепрогон — при НУЛЕ отказов пула в том окне.
`carve_family` зовёт `call()` ОДИН раз, поэтому у него не было ни одной второй попытки.

Проверяем три вещи разом:
  1. битый ответ ретраится и следующий валидный принимается (а не теряется);
  2. «не та форма» (массив вместо объекта) и «мусор» пишутся РАЗНЫМИ событиями —
     раньше оба были `parse_fail` и в базе не различались;
  3. упорный брак не уходит в бесконечность: ровно MAX_FAILS обращений, потом None.

⚠️ Требует KB_DB на ОТДЕЛЬНУЮ базу: тест сносит файл. Ключи фиктивные, сети нет —
urlopen подменён. KB_GROUP_SIZE=0: звенья тут ни при чём.
  KB_DB=/tmp/kb_parse.db KB_GRANT_MAX=0.01 KB_GRANT_MIN=0.01 KB_GROUP_SIZE=0 \
  python test_parse_retry.py
"""

import json
import os
import sys

import keybroker as k

KEYS = ["key-%d" % i for i in range(5)]


class _Resp:
    """Дублёр ответа urlopen. ⚠️ `.headers` обязателен: call() читает их для разведки
    квота-заголовков, и без атрибута падает AttributeError — а он ловится общим `except`
    и превращается в status=-1 «сеть». Тест тогда молча проверяет НЕ ТО (проверено собой).
    """

    headers = {}

    def __init__(self, text):
        self.body = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        ).encode()

    def read(self):
        return self.body


def run(texts):
    """Прогнать call() так, будто Google отдаёт 200 с этими телами по очереди.
    Возвращает (результат, сколько раз реально сходили в сеть)."""
    seq = list(texts)
    calls = [0]

    def fake_urlopen(req, timeout=None):
        calls[0] += 1
        return _Resp(seq.pop(0) if seq else seq_last[0])

    seq_last = [texts[-1]]
    orig = k.urllib.request.urlopen
    k.urllib.request.urlopen = fake_urlopen
    k.get_keys = lambda: KEYS  # ключей из env тут нет и не надо
    try:
        return k.call("u", "s", "t"), calls[0]
    finally:
        k.urllib.request.urlopen = orig


def events():
    c = k._conn()
    rows = dict(
        c.execute(
            "SELECT event, COUNT(*) FROM request_log "
            "WHERE event NOT IN ('grant','report') GROUP BY event"
        ).fetchall()
    )
    c.close()
    return rows


def ok(cond, what, got=""):
    print("%-56s %-26s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    if not os.environ.get("KB_DB"):
        sys.exit("ОТКАЗ: задай KB_DB на тестовую базу — тест сносит файл")
    try:
        os.remove(k.DB)
    except OSError:
        pass
    k.init()
    k.seed_caps()
    good = True

    # 1. Мусор → повтор → валидный ответ принят. Раньше тут был мгновенный None.
    res, n = run(["это не json вообще", '{"intents": [1]}'])
    good &= ok(
        res == {"intents": [1]},
        "1. мусор ретраится, следующий ответ принят",
        "%r за %d обращения" % (res, n),
    )
    good &= ok(n == 2, "   ровно 2 обращения, не больше", "%d" % n)
    good &= ok(events().get("parse_junk") == 1, "   событие parse_junk", str(events()))

    # 2. Массив вместо объекта — ДРУГОЕ событие, не то же самое имя.
    res, n = run(['[{"a": 1}]', '{"intents": [2]}'])
    good &= ok(res == {"intents": [2]}, "2. не та форма ретраится", "%r" % res)
    good &= ok(
        events().get("parse_shape") == 1 and events().get("parse_junk") == 1,
        "   parse_shape отдельно от parse_junk",
        str(events()),
    )

    # 3. Упорный брак: сдаёмся ровно на MAX_FAILS, а не крутимся вечно.
    res, n = run(["мусор" for _ in range(k.MAX_FAILS + 3)])
    good &= ok(res is None, "3. упорный брак → None", "%r" % res)
    good &= ok(
        n == k.MAX_FAILS,
        "   потолок обращений не вырос: ровно MAX_FAILS",
        "%d при MAX_FAILS=%d" % (n, k.MAX_FAILS),
    )

    # 4. ⛔ Ключ за брак модели НЕ наказан: 200 пришёл, ключ отработал. Наказывать значило
    #    бы гасить здоровые ключи за чужую вину.
    c = k._conn()
    cd = c.execute(
        "SELECT COUNT(*) FROM key_clock WHERE cooldown_until > 0"
    ).fetchone()[0]
    lvl = c.execute("SELECT COALESCE(SUM(cd_level),0) FROM key_clock").fetchone()[0]
    c.close()
    good &= ok(
        cd == 0 and lvl == 0,
        "4. ключи НЕ наказаны за брак модели",
        "в кулдауне %d, сумма ступеней %d" % (cd, lvl),
    )

    print("\nVERDICT:", "OK — парс-фейл ретраится и виден поимённо" if good else "FAIL")
    sys.exit(0 if good else 1)
