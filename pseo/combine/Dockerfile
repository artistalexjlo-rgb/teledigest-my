FROM python:3.12-slim
WORKDIR /app

# ⚠️ КОНТЕКСТ СБОРКИ = КОРЕНЬ РЕПО, не pseo/combine. В Dokploy: Dockerfile Path
# `pseo/combine/Dockerfile`, Docker Context Path `.`. Так сделано ради рендер-половины:
# из контекста pseo/combine файлы pseo/render.py, templates/, i18n/ недостижимы, и их
# пришлось бы держать ВТОРОЙ копией в git. Копии в этом проекте уже стоили прода трижды
# (facet_lang, keybroker, lang_runner) — новых не заводим.
COPY pseo/combine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Пульт: бот, дубли ртов (builder/), тесты, README.
COPY pseo/combine/ /app/

# Рендер-половина — БЕЗ дублей в git, забираем из pseo/ по месту жительства.
# Пути внутри образа держим ТЕ ЖЕ, что на десктопе относительно pseo/: render.py считает
# BASE от себя, pages.py и readycheck.py — на уровень выше себя. Отсюда /app = pseo/.
COPY pseo/render.py pseo/favicon.svg /app/
COPY pseo/config/ /app/config/
COPY pseo/templates/ /app/templates/
COPY pseo/i18n/ /app/i18n/
COPY pseo/static/ /app/static/
COPY pseo/mirror_static/ /app/mirror_static/
# `country_codes.py` — справочник стран рядом с pages.py (249 стран, имена и флаги).
# Копия файла бота, дословная; сторож `test_country_codes_copy` не даст ей отстать. В образ
# едет как обычный модуль builder — никакого захода в `src/` и никаких путей вне /app.
COPY pseo/builder/pages.py pseo/builder/readycheck.py pseo/builder/country_codes.py pseo/builder/shelf_assign.py /app/builder/

# BRAIN_DIR (/brain = /root/pseo_builder хоста) монтируется Dokploy: keybroker.db, out_facet и пр.
CMD ["python", "-u", "bot.py"]
