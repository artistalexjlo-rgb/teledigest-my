"""Сторож конфига отдачи: правила Pages переводятся в nginx без потерь и без поломки.

Проверяются те четыре вещи, каждая из которых уже стреляла или гарантированно выстрелит:
дубль ключа (nginx вообще не стартует), звёздочка Pages, адрес без хвостового слеша
(живьём отдавал 404) и наличие в конфиге своей 404-страницы с относительными редиректами.
"""

import nginx_conf

SAMPLE = """/landing/* / 301

# комментарий: его в правила пускать нельзя
/en/br/card-payments/ /en/br/bank-card-payments/ 301
/ru/br/staryy/ /ru/br/novyy/ 301
/ru/br/staryy/ /ru/br/tretiy/ 301
"""


def test_comments_and_blanks_dropped():
    rules, _ = nginx_conf.parse(SAMPLE)
    assert not any(k.startswith("#") for k in rules)
    assert all(k.startswith("/") or k.startswith("~^") for k in rules)


def test_wildcard_becomes_regex():
    """`/landing/*` у Pages — префикс. В nginx это регексп-ключ, иначе правило молча мертво."""
    rules, _ = nginx_conf.parse(SAMPLE)
    assert rules.get("~^/landing/") == "/"
    assert "/landing/*" not in rules


def test_no_duplicate_keys():
    """⛔ Главное правило: `map` с повторяющимся ключом не даёт nginx запуститься — сайт
    ляжет целиком, а не «одно правило не сработает». Дубль в источнике есть (замер: 1).
    """
    rules, dups = nginx_conf.parse(SAMPLE)
    body = nginx_conf.render_map(rules)
    keys = [ln.split('"')[1] for ln in body.strip().splitlines()]
    assert len(keys) == len(set(keys)), "дубль ключа в map — nginx не стартует"
    assert "/ru/br/staryy/" in dups, "конфликтующий дубль должен попасть в отчёт"
    assert rules["/ru/br/staryy/"] == "/ru/br/novyy/", "оставляем ПЕРВОЕ вхождение"


def test_both_slash_variants():
    """Правила записаны со слешем, а ходят и без него: `/en/br/card-payments` живьём
    отдавал 404. Значит оба варианта ключа обязаны быть в карте."""
    rules, _ = nginx_conf.parse(SAMPLE)
    assert rules["/en/br/card-payments/"] == "/en/br/bank-card-payments/"
    assert rules["/en/br/card-payments"] == "/en/br/bank-card-payments/"


def test_map_hash_sizes_present():
    """⛔ Без этих двух строк nginx НЕ СТАРТУЕТ на нашем объёме правил: живой прогон
    2026-08-11 дал `could not build map_hash ... bucket_size: 64`. Дело не в длине ключа
    (максимум 48 символов), а в коллизиях на ~2000 записях. Проверять только `nginx -t`
    нельзя — сторож нужен, чтобы строки не выпали при следующей правке шаблона."""
    conf = nginx_conf.CONF.format(src_name="_redirects", n_rules=3)
    assert "map_hash_bucket_size" in conf
    assert "map_hash_max_size" in conf


def test_conf_has_what_pages_did_for_us():
    conf = nginx_conf.CONF.format(src_name="_redirects", n_rules=3)
    assert "error_page 404 /404.html" in conf, "свой 404 вместо nginx-овского"
    assert (
        "absolute_redirect off" in conf
    ), "за прокси Location обязан быть относительным"
    assert "include /etc/nginx/conf.d/redirects.map" in conf
    assert "map $uri $pseo_redirect" in conf
    assert "root /usr/share/nginx/html" in conf
