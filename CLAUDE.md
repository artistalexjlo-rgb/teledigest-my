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

- GitHub → Dokploy auto-deploy при push в main
- Docker multi-stage, secrets через env (TELEGRAM_API_ID/HASH/BOT_TOKEN, GEMINI_API_KEY)
- Конфиг: File Mount в Dokploy → `/config/teledigest.conf`
- Данные (SQLite + сессии): Directory Mount — `/home/teledigest/data`
- db_path в конфиге: `/home/teledigest/data/messages_fts.db`
- sessions_dir в конфиге: `/home/teledigest/data`
- VPS: 199.195.252.114 (BuyVM), контейнер: `docker ps | grep bots-grab`

## Qdrant auth

- Qdrant защищён глобальным `QDRANT__SERVICE__API_KEY` (включён на контейнере
  Qdrant, т.к. он расшарен наружу по HTTPS для другого сервиса). После включения
  Qdrant требует ключ на ЛЮБОЙ запрос, включая внутренние.
- teledigest шлёт ключ через env **`QDRANT_API_KEY`** → `config._parse_qdrant`
  (env имеет приоритет над `[qdrant] api_key`) → единственный клиент
  `qdrant_db.get_client()` передаёт `api_key=cfg.api_key or None`. Пусто = None =
  старое поведение без auth.
- URL внутренний: `host=qdrant`, `port=6333` (dokploy-network). Публичный домен
  teledigest НЕ нужен — не менять host/port.
- **Порядок включения (критично):** сначала задеплоить teledigest с заданным
  `QDRANT_API_KEY`, ПОТОМ владелец включает `QDRANT__SERVICE__API_KEY` на Qdrant.
  Иначе teledigest получает 401 на запись и перестаёт наполнять базу.
- Ключ обязателен синхронно у ВСЕХ клиентов Qdrant (luky тоже шлёт).
- Контракт эмбеддингов (модель `gemini-embedding-2` / dim 1536 / COSINE /
  task_type) — НЕ трогать; менять только синхронно со всеми потребителями.
