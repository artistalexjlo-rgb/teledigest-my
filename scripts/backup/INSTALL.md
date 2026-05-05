# VPS daily backup → Telegram

Раз в сутки в 04:00 по UTC на хосте (не в Coolify) запускается systemd-timer.
Он тарит важные пути, делает дамп Coolify-БД, отправляет архив тебе в
Telegram как документ. Локально хранит последние 7 копий.

## Что бэкапится

- `/data/` целиком (Coolify metadata, File Mounts, root-level app data).
- `/home/` целиком (per-app persistent storage, Telethon-сессии, SQLite-БД).
- `/root/3x-ui/` + `/usr/local/etc/xray/` + xray systemd units.
- `pg_dumpall` всей `coolify-db` postgres базы.
- Сами `vps-backup.*` файлы (на случай если на новой машине надо повторить).

Исключаются: `*.db-shm`, `*.db-wal` (временные SQLite WAL/SHM), `__pycache__`.

## Один раз на VPS — установка

### 1. Создать Telegram-бота для бэкапов

В Telegram открой @BotFather → `/newbot` → имя на свой выбор (например
`my_vps_backup_bot`). Получишь токен `123:AAA...`.

**Важно:** напиши новому боту `/start` в личке — иначе Telegram API не даёт
ему писать тебе.

### 2. Узнать свой user ID

Напиши @userinfobot в Telegram → `/start`. Ответит числом (например
`492206411`). Это и есть твой `TG_CHAT_ID`.

### 3. Положить файлы на VPS

```sh
# Скопировать с локалки на VPS (с винды в CMD):
scp scripts\backup\vps-backup.sh root@VPS_IP:/usr/local/bin/
scp scripts\backup\vps-backup.service root@VPS_IP:/etc/systemd/system/
scp scripts\backup\vps-backup.timer root@VPS_IP:/etc/systemd/system/
```

На VPS:

```sh
chmod +x /usr/local/bin/vps-backup.sh

# Положить токен в защищённый env-файл
cat > /etc/vps-backup.env <<'EOF'
TG_TOKEN=123456789:AABBccDDeeFFggHHiiJJkkLL
TG_CHAT_ID=492206411
EOF
chmod 600 /etc/vps-backup.env

# Активировать timer
systemctl daemon-reload
systemctl enable --now vps-backup.timer
```

### 4. Проверить что таймер встал

```sh
systemctl list-timers vps-backup.timer
# В колонке NEXT — следующий запуск (должно быть завтра 04:00).
```

### 5. Прогнать вручную сейчас (тест)

```sh
systemctl start vps-backup.service
journalctl -u vps-backup.service -f
# Ctrl+C чтобы выйти из follow.
```

В Telegram должен прийти документ от твоего бота с архивом.

## Логи

```sh
journalctl -u vps-backup.service --since today
journalctl -u vps-backup.service -n 100
```

## Восстановление из бэкапа

Скачай архив из Telegram на машину где будешь восстанавливать (новый VPS
или текущий после reinstall ОС). Потом:

```sh
# На свежем VPS — после установки Coolify
systemctl stop coolify-db   # чтобы не конфликтовало с restore
cd /
tar xzf vps-backup-YYYY-MM-DD.tar.gz

# Восстановить Coolify-DB
docker exec -i coolify-db psql -U coolify -d coolify < coolify-db.sql
# Или, если структура schema не совпадает между версиями Coolify:
# вручную добавить приложения через Coolify UI, env vars из заметок.

systemctl start coolify-db
docker compose -f /data/coolify/source/docker-compose.yml restart
```

## Ротация

Скрипт оставляет 7 последних архивов в `/root/backups/`. Старые удаляет.
Если пере-нагрузка диска — поменяй `head -n -7` в конце скрипта на меньше.

## Troubleshooting

**Бот не пишет / `chat not found`:** ты не нажал `/start` боту в личке.

**`Document is too big`:** архив >50MB. Скрипт автоматически делит на части
по 45MB. Если split упал — посмотри `journalctl`.

**`pg_dump failed`:** coolify-db контейнер не запущен. Бэкап продолжится
без БД — посмотри `docker ps | grep coolify-db`.

**Архив пустой / маленький:** некоторые пути могут быть недоступны
(permissions). Tar warnings подавлены, но `--ignore-failed-read` пропускает
такие файлы.
