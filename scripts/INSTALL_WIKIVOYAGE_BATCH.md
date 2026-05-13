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

### 2. Отметь уже залитые страны (Thailand залита руками раньше)

```bash
CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep bots-grab | awk '{print $1}' | head -1)

docker exec "$CID" python3 -c "
import json
from pathlib import Path
from datetime import datetime, timezone

state_path = Path('/home/teledigest/data/wikivoyage_batch_state.json')
state = json.loads(state_path.read_text()) if state_path.exists() else {}

# Страны, залитые вручную до запуска батча
already_done = {
    'th': 9649,   # Thailand — первый тестовый прогон 2026-05-11
}
now = datetime.now(timezone.utc).isoformat()
for cc, patterns in already_done.items():
    state[cc] = {'status': 'done', 'patterns': patterns, 'finished_at': now}

state_path.write_text(json.dumps(state, indent=2))
print('Marked:', list(already_done.keys()))
"
```

### 3. Посмотри статус

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
