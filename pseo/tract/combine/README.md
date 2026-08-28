# Комбайн-пульт — ТГ-бот + исполнитель тракта в одном контейнере

⛔ Новый файл (28.08), не правка старого — старый описывал снятую схему публикации
(«Публикация без ассистента», Cloudflare Pages, git-транспорт страниц) и старые команды
меню (`facet br`, `assign vn`). Всё это отменено; актуальная схема — `pseo/tract/docs/PLAN.md`.

Канон: запуск/стоп/статус — только юзер из своего ТГ. Отчёт каждые 50 попыток мозга с
прогрессом и кнопкой ⛔ СТОП. Финальный отчёт при любом исходе. Одна задача за раз.

## Что внутри образа

Один `COPY pseo/tract/ /app/` — образ несёт ВЕСЬ тракт (см. `pseo/tract/docs/PLAN.md`),
не только пульт: `bot.py` (сам пульт), `tract.py`/`site.py`/`translation.py`/`render.py`/
`readycheck.py`/`readiness.py` (звенья 2–7), `config/`, `templates/`, `i18n/`, `static/`
(рендер-половина). Никаких дублей модулей под сборку образа больше нет (были в
`combine/tract/`, сняты 28.08) — оригиналы едут напрямую.

## Деплой (Dokploy на VPS 199.195.252.114) — полный цикл

1. BotFather: `/newbot` → имя, например `luky_combine_bot` → скопировать **токен**.
2. Узнать свой chat_id: написать новому боту что угодно, затем открыть
   `https://api.telegram.org/bot<ТОКЕН>/getUpdates` — в ответе `message.from.id`.
3. Dokploy → Create Service → **Application**:
   - Source: GitHub `artistalexjlo-rgb/teledigest-my`, branch `main` (после мёржа),
     тип сборки Dockerfile.
   - **Dockerfile Path**: `pseo/tract/combine/Dockerfile`
   - **Docker Context Path**: `.` (корень репо — без него `COPY pseo/tract/...` внутри
     Dockerfile не находит файлы)
4. Environment (вкладка Environment):
   - `COMBINE_BOT_TOKEN` = токен из п.1
   - `ADMIN_ID` = число из п.2 (твой личный telegram-id; в приватном чате он же chat_id)
   - `GEMINI_API_KEY_1` … `GEMINI_API_KEY_12` = те же, что у bots-grab (ртам нужны ключи)
5. Mounts (вкладка Mounts) — три Directory Mount, пути в контейнере ТЕ ЖЕ, что на хосте:
   - `/root/pseo_builder` → `/root/pseo_builder` (мозг, тестовый прогон тракта, флаги — RW)
   - `/home/teledigest/data` → `/home/teledigest/data` (messages_fts.db — мухи)
   - `/root/embed_ab` → `/root/embed_ab` (local_vec.db — вектора для dedup)
6. Deploy. В ТГ придёт «🟢 комбайн-пульт на связи».
7. Проверка без ключей: `/status` — должен показать расход PT-дня из мозга.

## Команды

- `/combine` — меню кнопками, шаги тракта по порядку (см. `PLAN.md`): схлопывание,
  разметка, обобщение, корпус, переводы, сборка сайта, готовность снимка.
- `/stop` — стоп-флаги + SIGTERM, финальный отчёт придёт.
- `/status` — что бежит, расход дня, макс-ключ, 429 за час.
- `/geo <код>` — страна пробы (пока схема испытывается, работа идёт по одной стране).

## Границы

- **Публикации ещё нет.** Готовность (последний шаг) — гейт над снимком в `{BRAIN}/tests`,
  не выкладка наружу: боевого каталога раздачи и второго домена сегодня не существует.
- Прод-бот `bots-grab` не затронут. Осиротевших процессов нет: рты — дети контейнера,
  умирают вместе с ним (`start_new_session` + `killpg` на стопе).
