# Apps Script — MultySpeak miner

Apps Script проект который читает daily-sample файлы из Google Drive,
прогоняет их через Gemini, и складывает результаты в Firestore:

- `wisdom_base` — мухи (структурированные факты для МОЗГ-ассистента)
- `telegram_queue` — котлеты (сторителлинг для канального автопостинга)

## Структура

- `Code.gs` — главный скрипт (логика обработки)
- `appsscript.json` — манифест (timezone, OAuth scopes)
- `.clasp.json` — привязка к Google Script ID

## Локальная разработка через clasp

[clasp](https://github.com/google/clasp) — официальный CLI Google для
двусторонней синхронизации Apps Script между локальным репо и облачным
редактором.

### Один раз

```sh
npm install -g @google/clasp
clasp login
```

Откроется браузер, авторизуешь Гугл-аккаунт.

### Развернуть локально текущее состояние из Гугла

Если хочешь сначала вытащить ровно то что сейчас в облаке (например для
бэкапа перед нашим refactor'ом):

```sh
cd apps_script
clasp pull
```

Это перезапишет `Code.gs` и `appsscript.json` тем что у тебя в Apps
Script. Используй когда вручную правил в редакторе и хочешь забрать
изменения в репо.

### Залить в облако нашу версию

```sh
cd apps_script
clasp push
```

Это перезапишет облачный код тем что в репе. После этого открываешь
[редактор](https://script.google.com/home/projects/10eHqb9x16Sut1WarTeFDs_cwvnRrkZCtRA4VKyJx-W0aCwC-BwFBMC8r/edit) — увидишь обновлённый код.

## Один раз в редакторе после первого `clasp push`

1. Открой проект в Apps Script.
2. **File → Project properties → Script properties → Add row:**
   - `key` = `GEMINI_API_KEY`
   - `value` = твой Gemini API key
   - **Save**
3. Опционально, тоже как Script properties:
   - `FIREBASE_PROJECT_ID` = полный project id (если default в коде обрезан)
   - `FOLDER_ID` = id папки Drive с input-файлами (если хочешь переопределить)
   - `SUMMARY_FOLDER_ID` = id отдельной папки Drive для дневных summary-TXT
     (см. ниже). Если не задано — summary не пишется, пайплайн работает.
   - `SUMMARY_TZ` = timezone для даты в имени summary-файла. Дефолт
     `Europe/Moscow`. Менять только если хочешь привязать дату к другому
     поясу.
4. Сверху выбери из dropdown функцию `processNewLogs` → жми **▶ Run**.
   Гугл попросит авторизовать scopes (Drive, Firestore, External requests).
   Жми **Review permissions** → выбери аккаунт → **Advanced → Go to
   <project> (unsafe) → Allow**.
5. Дождись завершения. Внизу в **Execution log** должно быть
   `>>> Processing: ...` и `Saved N of M patterns ...`.
6. Открой **Triggers** (значок будильника слева) → **Add Trigger**:
   - Function: `processNewLogs`
   - Event source: `Time-driven`
   - Type of time: `Minutes timer` → `Every 15 minutes`
   - **Save**

С этого момента скрипт сам себя запускает каждые 15 минут, ловит новые
файлы, обрабатывает.

Почему 15 минут, а не час: у Apps Script consumer-аккаунта **hard limit 6
минут на одно выполнение**. Скрипт сам выходит на 5-й минуте, оставляя
непомеченные файлы следующему триггеру. Час между запусками = до часа
лага если первая пачка не влезла. 15 минут = быстрее догоняем.

Если триггеров несколько (старый hourly + новый 15-min) — удали старый,
иначе будут параллельные запуски на одни и те же файлы. Idempotent, но
бессмысленно жрёт квоту.

## Daily mining summary в Drive

Каждый раз когда Apps Script успешно записывает pattern в Firestore — он
ещё дописывает строку в дневной TXT файл на Drive. Это даёт быстрый
человекочитаемый отчёт "что налили в БД сегодня" без необходимости
ползать в Firestore Console.

### Один раз — создать папку

1. В Google Drive создай папку, например `mining_summaries`.
2. Скопируй её ID из URL: `https://drive.google.com/drive/folders/<это_id>`.
3. В Apps Script: **Project settings → Script properties → Add property**:
   - `SUMMARY_FOLDER_ID` = `<тот ID>`
   - **Save**.

С этого момента в этой папке появятся два файла в сутки:

- `YYYY-MM-DD_wisdom.txt` — мухи (assistant data, ai_lesson)
- `YYYY-MM-DD_stories.txt` — котлеты (channel stories, human_story)

Формат одной строки:
```
[br/Finance] Bank card for non-residents :: Use Caixa branch in SP with passport + tax ID  (src: 2026-05-07_br_chatforum.txt)
```

Дата = дата запуска, не дата исходного лога. FORCE_REPROCESS-прогон
архива за неделю → всё в сегодняшний summary. Удобно для оценки качества
очередного апдейта промпта.

Если `SUMMARY_FOLDER_ID` не задан — summary не пишется, остальной
пайплайн работает как обычно.

## Что изменилось в v3.2 vs v3.1

1. **Retry на 503/429.** Gemini free tier регулярно отдаёт
   `503 UNAVAILABLE` ("model overloaded"). Раньше файл сразу падал и
   ждал следующего триггера. Теперь 3 ретрая с backoff 5s → 20s → 60s.
2. **Пауза между файлами 4.5s.** Без неё пробивали RPM-лимит на длинных
   прогонах (15 RPM на 3.1-flash-lite-preview). С паузой держимся
   ~13 RPM, есть запас на ретраи.
3. **Runtime budget 5 минут.** Apps Script consumer аккаунт = hard 6 min
   на execution. Скрипт сам выходит на 5-й минуте, оставшиеся файлы
   подхватит следующий триггер (они не помечены processed).
4. **Триггер каждые 15 минут** вместо часа. 4 запуска в час × 5 мин =
   20 мин полезной работы, на ежедневный объём с запасом.
5. **Промпт: правила выбора routing.** Было: "одна из строк, выбирай
   сам". Стало: явные критерии для `both` / `assistant_only` /
   `channel_only` с дефолтом на `both`. Без этого Gemini дрейфовал в
   `channel_only` — мухи в `wisdom_base` переставали приходить.

## Что изменилось в v3.1 vs твоей оригинальной v3.0

1. **Безопасность.** API key больше не в коде — читается из
   `PropertiesService`. Не попадёт в git.
2. **Country codes.** В промпте теперь явно перечислены ISO-коды твоих
   стран в нижнем регистре (`br, id, lk, mu, at, ar, be`) — соответствует
   `messages.country` в SQLite. Раньше было `BR, TR, ID, SL` — `SL`
   некорректен (это Сьерра-Леоне; Шри-Ланка — `LK`), и регистр не сходился.
3. **Идемпотентность.** Документы в Firestore теперь имеют detеrministic
   doc ID = SHA1(file_id : index : collection). При повторной обработке
   того же файла Firestore возвращает `409 ALREADY_EXISTS` — мы
   трактуем как успех, дубликаты не плодятся.
4. **Реальное отслеживание успеха.** Файл помечается `processed` только
   если **все** Firestore-записи для его patterns прошли. Раньше пометка
   ставилась если парсинг JSON прошёл — даже если все записи в БД
   упали тихо.
5. **Default routing.** Если Gemini не вернул `routing` — дефолтим на
   `"both"` (раньше pattern молча терялся).

## Re-process архив

Чтобы прогнать заново уже помеченные файлы (например после смены
промпта или схемы):

1. В `Code.gs` поменяй `var FORCE_REPROCESS = false;` на `true`.
2. `clasp push`
3. В редакторе запусти `processNewLogs`.
4. После прогона — верни обратно в `false` и `clasp push`, чтобы триггер
   не молотил повторно.

Дубликатов не будет — deterministic doc IDs.

## Схема `telegram_queue` документа

```js
{
  title: "Bank card for non-residents",
  country: "br",
  tag: "Finance",
  createdAt: <timestamp>,
  sourceFileId: "...",  sourceIdx: 3,
  content: "Если вы всё ещё мучаетесь...",       // канонический RU
  target_languages: ["ru", "en", "es", "pt"],     // на каких имеет смысл
  postedTo: {}                                     // бот заполнит per-channel
}
```

`postedTo` остаётся пустым на стороне Apps Script. Бот при первой обработке
pattern'а у себя в коде enрichит:
```js
postedTo: {
  telegram_ru_br: { posted: false, scheduled_at: <now>, posted_at: null,
                    language: "ru", text: null },
  vk_main:        { posted: false, scheduled_at: <now>, posted_at: null,
                    language: "ru", text: null },
  telegram_en_br: { posted: false, scheduled_at: <now>, posted_at: null,
                    language: "en", text: null }
  // ...только для каналов где channel.language ∈ target_languages
}
```

При постинге бот:
1. Спрашивает Gemini «адаптируй для канала X на языке L», передавая
   контекст уже отпощенного (если есть в других `postedTo.*.text`).
2. Постит в реальную платформу.
3. Обновляет `postedTo.<channel> = {posted: true, posted_at: now, text: ...}`.

## TODO

- Бот-постер (PR #6) — реализует enrichment `postedTo` и per-channel
  адаптацию через Gemini. Сейчас Apps Script производит только канонические
  pattern'ы; постинг ещё не сделан.
