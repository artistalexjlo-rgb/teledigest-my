# Teledigest — project rules

## Обязательно
- **После любых значимых изменений в коде/архитектуре** — обновить memory файлы в `C:\Users\Servak\.claude\projects\D--temp1--Grab\memory\`. Не ждать пока попросят.
- **Язык общения** — русский, коротко, по делу. Не лить воду.
- **Не делать предположений** — если не уверен, спроси. Не мысли жопой.
- **Секреты** — НИКОГДА не коммитить ключи, токены, api_hash. Только через env переменные.
- **Инструкции пользователю** — давать ПОЛНЫЙ цикл действий, а не обрывки. Юзер не devops. Если нужно заполнить форму — перечислить ВСЕ поля и что в них писать. Если нужно выполнить шаги — пронумеровать ВСЕ, включая очевидные.
- **Git workflow** — всегда через ветку + PR, не в main напрямую.

## Архитектура
- Память проекта: `memory/MEMORY.md` — индекс, остальные файлы — детали
- Код бота: `src/teledigest/`
- Codex (GPT desktop) делает extraction/backfill, НЕ трогает код бота
- DeepSeek — ежедневная работа (дайджесты, МОЗГ)
- SQLite — единая база, путь задаётся в конфиге `[storage] db_path`
- Бот запускается ТОЛЬКО в Docker на VPS, не локально
- Страны: ISO коды из `country_codes.py` (~195 стран), хардкод больше не нужен
- Digest target: хранится в `sources` таблице, меняется через бота без редеплоя

## Деплой
- GitHub → Coolify auto-deploy при push в main
- Docker multi-stage, secrets через env (TELEGRAM_API_ID/HASH/BOT_TOKEN, DEEPSEEK_API_KEY)
- Конфиг: File Mount в Coolify → `/config/teledigest.conf`
- Данные (SQLite + сессии): Directory Mount в Coolify — Source path И Mount path = `/home/teledigest/data`
- db_path в конфиге: `/home/teledigest/data/messages_fts.db`
- sessions_dir в конфиге: `/home/teledigest/data`

## Pending
- Редактирование digest target через бота (в работе)
- Telegram Forum с топиками по странам вместо отдельных каналов
- Daily samples для анализа паттернов чата
- Дайджест без повторов (контекст из истории постов)
