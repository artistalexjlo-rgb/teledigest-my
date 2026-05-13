# Установка автобатча WikiVoyage → Firestore

Таймер ежедневно в 12:00 UTC берёт следующие 3 страны из очереди ~100 стран,
заливает WikiVoyage-данные в Firestore (`wikivoyage_base`), пишет прогресс
в state-файл и log-файл.

## Первая установка на VPS

### 1. Залей unit-файлы

```bash
cp /home/teledigest/repo/scripts/wikivoyage-batch.service /etc/systemd/system/
cp /home/teledigest/repo/scripts/wikivoyage-batch.timer   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wikivoyage-batch.timer
```

Проверь что встал:
```bash
systemctl list-timers wikivoyage-batch.timer
```

### 2. Посмотри статус

```bash
docker exec "$CID" python -m teledigest.scripts.wikivoyage_batch --status
```

Должен показать th как ✓ done, остальные — pending.

## Мониторинг

**Статус таблица:**
```bash
CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep bots-grab | awk '{print $1}' | head -1)
docker exec "$CID" python -m teledigest.scripts.wikivoyage_batch --status
```

**Лог последних запусков:**
```bash
cat /home/teledigest/data/wikivoyage_batch.log | tail -50
```

**Лог systemd:**
```bash
journalctl -u wikivoyage-batch.service --since "1 hour ago"
```

## Ручной запуск (без ожидания таймера)

```bash
systemctl start wikivoyage-batch.service
```

Или прямо в контейнере:
```bash
CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep bots-grab | awk '{print $1}' | head -1)

# Посмотреть что будет импортировано без записи
docker exec "$CID" python -m teledigest.scripts.wikivoyage_batch --dry-run

# Запустить с нестандартным размером батча
docker exec "$CID" python -m teledigest.scripts.wikivoyage_batch --n 5

# Перезалить конкретную страну (сброс + следующий запуск подхватит)
docker exec "$CID" python -m teledigest.scripts.wikivoyage_batch --reset fr
```

## Остановить когда все страны залиты

```bash
systemctl disable --now wikivoyage-batch.timer
```

## Параметры батча

| Параметр | Default | Описание |
|----------|---------|----------|
| `--n` | 3 | Стран за один запуск |
| `--status` | — | Таблица прогресса |
| `--dry-run` | — | Показать без записи |
| `--reset CC` | — | Сбросить страну в pending |
| `--state` | `/home/teledigest/data/wikivoyage_batch_state.json` | State-файл |
| `--config` | `/config/teledigest.conf` | Config бота |
