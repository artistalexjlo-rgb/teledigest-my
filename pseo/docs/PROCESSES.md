# РЕЕСТР ПРОЦЕССОВ VPS + ВЕСЬ СТЕК СТРАНИЦ — единственный источник правды

## ПОЛНЫЙ СТЕК (от мухи до прода; «процессов вне этой схемы не существует»)

```
ПОСТОЯННО (реестр ниже):
  бот bots-grab (docker) ── граббер чатов → SQLite
       └─ экстрактор: сообщения → мухи ai_lesson   [Gemini flash-lite, учёт gemini_quota]
       └─ embed → Qdrant                            [gemini-embedding-2, свой RPD]
  bge-sweep.timer (2ч) ── мухи → local_vec.db/Qdrant-local  [ЛОКАЛЬНАЯ модель, без Gemini]
  sentinel (cron /6ч) ── HTTP-сторож портала, чирик при поломке [без Gemini]

ПО ОТМАШКЕ (руками, nohup на время прогона, умирает по завершении):
  1. facet.py <geo>            тег+ru-перевод [рот facet] + карв семей [carve] → out_facet/
  2. facet.py <geo> --assign-tail   хвост → полки×типы [assign]
  3. dedup.py <geo|--all> [--kratko]  дедуп-группы (KEYLESS, bge) + короткий ответ [kratko]
  4. lang_runner.py            переводы по очереди языков → out_facet_<lang> [translate, labels]
  5. questions_page.py         вопрос-контур → out_questions [questions]

СБОРКА/ДЕПЛОЙ (KEYLESS, одна команда с десктопа):
  ship.py = pull(VPS) → pages.py(json) → render.py(HTML) → readycheck(гейт)
            → git push multyspeak-pages → Cloudflare Pages → info.multyspeak.online
```

Все Gemini-вызовы билдера — ТОЛЬКО через `keybroker.call` (капы ртов: facet 1500 · carve 300 ·
assign 300 · kratko 600 · translate 400 · labels 200 · questions 300; поверх них — per-key
шаг/RPD и вина-vs-среда). Мимо мозга запросов не существует.

**Правило (2026-07-20, после инцидента pseo-runner):** на VPS имеет право жить ТОЛЬКО то,
что есть в этой таблице. Новый юнит/таймер/крон = явное «да» юзера + строка здесь В ТОМ ЖЕ
заходе. Всё, чего в таблице нет, — сносится при первой сверке. Сверка (`systemctl
list-timers`, `crontab -l`, running services) — в начале каждой рабочей сессии.

**Контекст инцидента:** юнит `pseo-runner.service` (авто-тегание, «удобная автоматика»)
был заведён 6 июля вне git и вне доков, пережил все уборки и три недели молотил ключи,
пока юзер считал, что «ничего не работает». 2026-07-19 он выел 1764 гранта за день и
устроил 429-фон. Самоходных демонов у билдера НЕ СУЩЕСТВУЕТ КАК КЛАССА: весь тракт
(facet → dedup → pages → render → ship) — по отмашке.

## Разрешено (2026-07-20)

| процесс | тип | что делает | Gemini? | владелец/причина |
|---|---|---|---|---|
| `bge-sweep.timer` (2ч) + `bge-deadman.timer` | systemd | свипер локального bge-эмбеддинга → local_vec.db/Qdrant | НЕТ (локальная модель) | юзер, второй контур |
| `vps-backup.timer` | systemd | tar /data целиком | нет | юзер, бэкапы |
| cron `sentinel.py` (каждые 6ч) | cron root | HTTP-сторож сайта: обход /ru/, sitemap, квота-порог; чирик ТОЛЬКО при поломке | НЕТ (curl + sqlite) | детерминированный аудит (ferrari); проверка раннера ВЫКЛЮЧЕНА (раннера нет) |
| бот `bots-grab` (Dokploy) | docker | телеграм-бот: граббер/дайджест/экстрактор/эмбед | ДА (экстрактор+эмбед, свой учёт gemini_quota) | прод |
| `bots-luky3` (Dokploy) | docker | продукт Luky (multyspeak.online) | ДА (Live, свой) | прод |
| инфра: xray, ttyd, tinyproxy, fail2ban, chrony, cockpit, docker | systemd | доступ/VPN/секьюрити | нет | юзер |

## Снесено (не возвращать)

| процесс | снесён | почему |
|---|---|---|
| `pseo-runner.service` | 2026-07-20 | самоходный демон тегания, никем не заказан; раннеров у билдера нет как класса |
| `wikivoyage-batch.timer/.service` | 2026-07-20 | зомби Firestore-эпохи (мертва), ежедневный no-op |

Прогоны билдера (facet/dedup/facet_lang/ship) запускаются РУКАМИ по отмашке юзера,
живут в nohup на время прогона и умирают. Постоянных драйверов нет.
