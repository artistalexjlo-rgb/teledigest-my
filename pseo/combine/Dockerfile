FROM python:3.12-slim
WORKDIR /app

# ⚠️ КОНТЕКСТ СБОРКИ = КОРЕНЬ РЕПО, не pseo/combine. В Dokploy: Dockerfile Path
# `pseo/combine/Dockerfile`, Docker Context Path `.`. Так сделано ради рендер-половины:
# из контекста pseo/combine файлы pseo/render.py, templates/, i18n/ недостижимы, и их
# пришлось бы держать ВТОРОЙ копией в git. Копии в этом проекте уже стоили прода трижды
# (facet_lang, keybroker, lang_runner) — новых не заводим.
COPY pseo/combine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Пульт: бот, дубли модулей тракта (tract/), тесты, README.
COPY pseo/combine/ /app/

# Рендер-половина — БЕЗ дублей в git, забираем из pseo/ по месту жительства.
# Пути внутри образа держим ТЕ ЖЕ, что на десктопе относительно pseo/: render.py (в tract/)
# считает ROOT на уровень выше себя. Отсюда /app = pseo/, /app/tract = pseo/tract.
# ⛔ favicon.svg НЕ копируем: его не читает ни render.py, ни readycheck.py (проверено
# грепом 28.08) — отдаёт его тот, кто раздаёт ГОТОВЫЙ сайт, не пульт.
COPY pseo/config/ /app/config/
COPY pseo/templates/ /app/templates/
COPY pseo/i18n/ /app/i18n/
COPY pseo/static/ /app/static/
COPY pseo/mirror_static/ /app/mirror_static/
# `country_codes.py` — справочник стран рядом со сборщиком (249 стран, имена и флаги).
# Копия файла бота, дословная; сторож `test_country_codes_copy` не даст ей отстать. В образ
# едет как обычный модуль builder — никакого захода в `src/` и никаких путей вне /app.
# ⛔ `themes.json` здесь — ТОЛЬКО СИД (английский источник, статика из git). Рабочий файл,
# в который звено 6 дописывает купленные языки, живёт на смонтированном BRAIN_DIR и этой
# копии не касается — иначе редеплой стирал бы покупки (28.08).
# ⛔ `render.py` и `readycheck.py` — тут же, в tract/: переписаны заново 28.08, старые
# копии в pseo/render.py и pseo/builder/readycheck.py сняты (в legacy).
COPY pseo/tract/site.py pseo/tract/translation.py pseo/tract/render.py pseo/tract/readycheck.py pseo/tract/readiness.py pseo/tract/countries.json pseo/tract/themes.json pseo/tract/about.json /app/tract/

# BRAIN_DIR (/brain = /root/pseo_builder хоста) монтируется Dokploy: keybroker.db, out_facet и пр.
CMD ["python", "-u", "bot.py"]
