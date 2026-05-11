# Установка автопрогона исторических BR-логов

Этот таймер раз в сутки достаёт следующий пакет уже-сграбленных-но-
не-прокачанных BR-сообщений из SQLite, формирует sample-файлы и кладёт
их в `samples/br/`. Дальше `drive_uploader` и Apps Script подхватывают
их сами.

## Сколько ждать

- 15 файлов в день × ~25 секунд на файл в Apps Script = подкачка прозрачна
- Год архива (~365 дней × 1-3 источника) = ~24-30 суток до полного дренажа
- В сутки расходуется ~3% от Gemini free tier (500 RPD), 97% остаётся
  под ежедневные новые файлы + ретраи

## Установка на VPS (root)

Бот живёт в Docker'е под Dokploy. Имя контейнера меняется при ребилде,
поэтому юнит ищет его динамически по имени образа.

```bash
# 1. Положи юнит-файлы в /etc/systemd/system/
cp /path/to/repo/scripts/feed-br-history.service /etc/systemd/system/
cp /path/to/repo/scripts/feed-br-history.timer   /etc/systemd/system/

# 2. Перечитай юниты
systemctl daemon-reload

# 3. Включи и запусти таймер
systemctl enable --now feed-br-history.timer

# 4. Проверь что таймер встал
systemctl list-timers feed-br-history.timer
# Должен показать NEXT = ближайшие 03:17 UTC и LEFT = времени до запуска
```

## Проверить вручную (без ожидания таймера)

```bash
# Прогнать прямо сейчас один цикл
systemctl start feed-br-history.service

# Смотреть лог последнего запуска
journalctl -u feed-br-history.service --since "10 minutes ago"
```

В логе ищи строки:
- `Country=br: N pending pairs total, processing 15 this run`
- `Dumped country=br channel=... day=... -> 1 file(s)`
- `Run complete: N files written from 15 pairs. M pairs remaining`

## Запуск руками внутри контейнера (без systemd)

Если хочешь без таймера:

```bash
CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep teledigest | awk '{print $1}' | head -1)
docker exec "$CID" python -m scripts.feed_br_history --batch 15

# Посмотреть что бы дампнулось без записи:
docker exec "$CID" python -m scripts.feed_br_history --batch 15 --dry-run

# Больший батч (если хочется быстрее, всё равно упрётся в Gemini RPD):
docker exec "$CID" python -m scripts.feed_br_history --batch 30
```

## Остановить когда архив прокачается

Когда скрипт залогирует `no pending pairs — archive fully fed`:

```bash
systemctl disable --now feed-br-history.timer
```

Юнит-файлы можно оставить — пригодятся если решишь повторить процедуру
на другой стране позднее (запускать с `--country lk`, например).

## Чанкование

Сэмпл-файл больше 80K символов (≈25-30K Gemini-токенов) автоматически
режется на `_part01.txt`, `_part02.txt`, ... Это страхует от случайно-
огромных дней (5K сообщений во время Carnival) которые могли бы не
влезть в один Gemini-запрос. Apps Script обрабатывает каждый part
независимо — больше шансов выцепить уникальные patterns.
