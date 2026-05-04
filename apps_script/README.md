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
   - `FOLDER_ID` = id папки Drive (если хочешь переопределить)
4. Сверху выбери из dropdown функцию `processNewLogs` → жми **▶ Run**.
   Гугл попросит авторизовать scopes (Drive, Firestore, External requests).
   Жми **Review permissions** → выбери аккаунт → **Advanced → Go to
   <project> (unsafe) → Allow**.
5. Дождись завершения. Внизу в **Execution log** должно быть
   `>>> Processing: ...` и `Saved N of M patterns ...`.
6. Открой **Triggers** (значок будильника слева) → **Add Trigger**:
   - Function: `processNewLogs`
   - Event source: `Time-driven`
   - Type of time: `Hour timer` → `Every hour`
   - **Save**

С этого момента скрипт сам себя запускает раз в час, ловит новые файлы,
обрабатывает.

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
