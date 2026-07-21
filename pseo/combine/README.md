# Комбайн-пульт — отдельный ТГ-бот + исполнитель в одном контейнере

Канон: запуск/стоп/статус — только юзер из своего ТГ. Отчёт каждые 50 попыток мозга
с прогрессом и кнопкой ⛔ СТОП. Финальный отчёт при любом исходе. Одна задача за раз.
Исходный код ртов НЕ трогается: тут их **дубли** (`builder/`), данные — с хоста через маунты.

## Что внутри

- `bot.py` — пульт+исполнитель: long-poll ТГ, whitelist chat_id, jobs-аудит
  (`combine_jobs.db` рядом с мозгом), spawn дублей, отчётчик, стоп через флаги+SIGTERM.
- `builder/` — дубли ртов (keybroker, facet, facet_lang, dedup, lang_runner,
  tail_taxonomy) с ветки feat/brain-serial-queue (капы сняты). Синк дублей — руками,
  осознанно, НЕ автоматом.
- Стоп: ставятся `RUNNER_STOP` + `LANG_RUNNER_STOP` (рты чтут сами) + SIGTERM группе.

## Деплой (Dokploy на VPS 199.195.252.114) — полный цикл

1. BotFather: `/newbot` → имя, например `luky_combine_bot` → скопировать **токен**.
2. Узнать свой chat_id: написать новому боту что угодно, затем открыть
   `https://api.telegram.org/bot<ТОКЕН>/getUpdates` — в ответе `message.from.id`.
3. Dokploy → Create Service → **Application**:
   - Source: GitHub `artistalexjlo-rgb/teledigest-my`, branch `main` (после мёржа),
     Build Path: `pseo/combine`, тип сборки Dockerfile.
4. Environment (вкладка Environment):
   - `COMBINE_BOT_TOKEN` = токен из п.1
   - `ADMIN_ID` = число из п.2 (твой личный telegram-id; в приватном чате он же chat_id)
   - `GEMINI_API_KEY_1` … `GEMINI_API_KEY_12` = те же, что у bots-grab (ртам нужны ключи)
5. Mounts (вкладка Mounts) — три Directory Mount, пути в контейнере ТЕ ЖЕ, что на хосте:
   - `/root/pseo_builder` → `/root/pseo_builder` (мозг, out_facet, флаги — RW)
   - `/home/teledigest/data` → `/home/teledigest/data` (messages_fts.db — мухи)
   - `/root/embed_ab` → `/root/embed_ab` (local_vec.db — вектора для dedup)
6. Deploy. В ТГ придёт «🟢 комбайн-пульт на связи».
7. Проверка без ключей: `/status` — должен показать расход PT-дня из мозга.

## Команды

- `/combine` — меню кнопками (kratko / переводы / facet+carve / хвост→полки).
- Текстом с гео: `facet br`, `assign vn`.
- `/stop` — стоп-флаги + SIGTERM, финальный отчёт придёт.
- `/status` — что бежит, расход дня, макс-ключ, 429 за час.

## Границы

- Ship/pages/render — ДЕСКТОП, в комбайне их нет (репо pages и CF-доступ там).
- Прод-бот bots-grab не затронут. Осиротевших процессов нет: рты — дети контейнера,
  умирают вместе с ним (start_new_session + killpg на стопе).
