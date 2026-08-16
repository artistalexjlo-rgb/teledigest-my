"""nginx_conf.py — конфиг отдачи портала из файла правил `_redirects`.

Зачем. Портал ушёл с Cloudflare Pages на свой nginx (2026-08-11). Вместе с Pages потерялись
две его фичи: файл `_redirects` (992 правила; старые проиндексированные адреса стали 404) и
свой `404.html` — nginx по умолчанию отдаёт собственную страницу на 153 байта.

⛔ Правила НЕ переписываются руками во второй формат. Источник один — `_redirects`, который
едет в дереве сайта; тут только перевод его в синтаксис nginx. Иначе получим ровно ту
болезнь, что стоила проекту описи бэкапа и месяца публикации: одно знание в двух копиях,
вторая отстаёт.

Запуск:  python nginx_conf.py <путь к _redirects> <каталог вывода>
Пишет:   site.conf (server-блок) и redirects.map (тело map).
⚠️ Имя `redirects.map` без `.conf` намеренно: nginx сам подхватывает из conf.d только
`*.conf`, а этот файл — не конфиг, а список пар, и при автовключении дал бы синтаксис-ошибку.
"""

import pathlib
import sys

CONF = """# СГЕНЕРИРОВАНО pseo/builder/nginx_conf.py — руками не править, правка исчезнет.
# Источник правил: _redirects из дерева сайта ({src_name}, правил {n_rules}).

# ⛔ Размеры хеша ОБЯЗАТЕЛЬНЫ. С дефолтами (bucket 64, max 2048) nginx не стартует вовсе:
# `could not build map_hash, you should increase map_hash_bucket_size: 64` — проверено
# прогоном в одноразовом контейнере 2026-08-11 ДО того, как конфиг коснулся живого сайта.
# Дело не в длине ключа (самый длинный 48 символов), а в коллизиях на ~2000 записях.
map_hash_max_size 8192;
map_hash_bucket_size 256;

map $uri $pseo_redirect {{
    default "";
    include /etc/nginx/conf.d/redirects.map;
}}

server {{
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # За прокси адрес в Location обязан быть относительным: с absolute_redirect on nginx
    # подставил бы своё имя контейнера и порт 80, и слеш-редирект уводил бы наружу в никуда.
    absolute_redirect off;

    # 301 со старых адресов. Пустая строка = правила нет.
    if ($pseo_redirect != "") {{
        return 301 $pseo_redirect;
    }}

    gzip on;
    gzip_comp_level 5;
    gzip_min_length 512;
    # `application/json` — из-за поискового индекса (шаг 7): `/<язык>/search.json` весит
    # 352 КБ, в gzip 77 КБ. Без этого типа nginx отдаёт его несжатым.
    gzip_types text/css application/javascript application/json application/xml image/svg+xml;

    # Ассеты помечены хешем в имени запроса (?v=), поэтому кешировать можно надолго.
    location /assets/ {{
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}

    location / {{
        try_files $uri $uri/ $uri/index.html =404;
    }}

    # Своя страница вместо стандартной nginx-овской.
    error_page 404 /404.html;
    location = /404.html {{
        internal;
    }}
}}
"""


def parse(text: str) -> tuple[dict, list]:
    """`_redirects` → {ключ map: цель}. Возвращает (правила, отброшенные_дубли).

    Три вещи, которые ломают nginx или теряют трафик, и потому сделаны здесь:
    - ДУБЛИ источников: `map` с повторяющимся ключом не даёт nginx запуститься вообще.
      Оставляем первое вхождение, остальные возвращаем наверх для отчёта;
    - звёздочка Pages (`/landing/*`) → регексп-ключ nginx (`~^/landing/`);
    - вариант БЕЗ хвостового слеша: правила записаны со слешем, а Google и люди ходят и
      без него — проверено живьём, `/en/br/card-payments` отдавал 404.
    """
    rules: dict[str, str] = {}
    dups: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("/"):
            continue
        src, dst = parts[0], parts[1]
        keys = []
        if src.endswith("/*"):
            keys.append("~^" + src[:-1])  # /landing/* → ~^/landing/
        else:
            keys.append(src)
            twin = src[:-1] if src.endswith("/") else src + "/"
            if twin:
                keys.append(twin)
        for k in keys:
            if k in rules:
                if rules[k] != dst:
                    dups.append(k)
                continue
            rules[k] = dst
    return rules, dups


def render_map(rules: dict) -> str:
    lines = [f'    "{k}" "{v}";' for k, v in rules.items()]
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) < 3:
        return "нужно: nginx_conf.py <_redirects> <каталог вывода>"
    src = pathlib.Path(argv[1])
    dst = pathlib.Path(argv[2])
    rules, dups = parse(src.read_text(encoding="utf-8"))
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "redirects.map").write_text(render_map(rules), encoding="utf-8")
    (dst / "site.conf").write_text(
        CONF.format(src_name=src.name, n_rules=len(rules)), encoding="utf-8"
    )
    print(f"правил в map: {len(rules)}, отброшено конфликтующих дублей: {len(dups)}")
    if dups:
        print("  дубли:", ", ".join(dups[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
