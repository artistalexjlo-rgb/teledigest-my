FROM python:3.12-slim
WORKDIR /app

# ⛔ Новый файл, не патч старого (28.08, юзер: «докерфайл старый, ты на него натягиваешь
# новое» — так делать нельзя). Один COPY всего pseo/tract/ целиком, С ТОЙ ЖЕ вложенностью,
# что в репозитории: bot.py остаётся внутри tract/combine/. Поэтому пути у render.py,
# site.py и bot.py (все ищут соседей вроде config/ через собственный каталог или через
# dirname(dirname())) работают ОДИНАКОВО что на диске разработчика, что в контейнере —
# никакой отдельной арифметики путей под прод, никаких дублей модулей под сборку образа.
COPY pseo/tract/combine/requirements.txt /app/combine/requirements.txt
RUN pip install --no-cache-dir -r /app/combine/requirements.txt

COPY pseo/tract/ /app/

# BRAIN_DIR (/brain = /root/pseo_builder хоста) монтируется Dokploy: keybroker.db, out_facet и пр.
WORKDIR /app/combine
CMD ["python", "-u", "bot.py"]
