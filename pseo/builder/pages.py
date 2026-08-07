"""pages.py — из BUILT-данных гео (out_facet[_<lang>]/<geo>.json + out_questions[_<lang>]/<geo>.json)
собирает portal-схему data/ (октагон-шаблон): гео-хаб + факт-тема-страницы + вопрос-хаб/темы.
Оба контура, единообразно. Дизайн гарантирован шаблоном (page/qlist/index.html.j2).

Мультиязык: lang="ru" читает out_facet/, любой другой — out_facet_<lang>/ (структура ×1, текст из
оригинала). Копия страниц — из COPY[lang]. render.py сам подхватит i18n/<lang>.json по page.lang.

Запуск: python pages.py <geo> [<geo2> ...]   (или --all по out_facet/*.json; строит ВСЕ языки, у кого
есть built-данные). Дальше — render.py --all + валидация (readycheck).
"""

import datetime
import glob
import hashlib
import json
import os
import re
import sys

import tail_taxonomy as _tax
from slugs import slug  # ЕДИНСТВЕННОЕ определение хвоста адреса

# полка → стабильный латинский ключ для URL (/ru/<geo>/s/finance/), не транслит-slug
SHELF_KEY = {name: key for key, name, _ in _tax.SHELVES}
# тип абзаца → латинский ключ (css-класс тега на карточке/аккордеоне)
TYPE_KEY = {name: key for key, name, _ in _tax.TYPES}
# короткий ярлык тега (полное имя типа громоздко для чипа в аккордеоне)
# Ярлык типа абзаца ПО ЯЗЫКАМ. Был плоским русским словарём — и это была вторая причина,
# по которой полочный контур держали под `if lang == "ru"`: на английской странице чип
# напечатал бы «лайфхак». Нет языка — англ. фолбэк, не русский (русский виден как брак).
TYPE_SHORT = {
    "ru": {
        "lifehack": "лайфхак",
        "reglament": "регламент",
        "howto": "инструкция",
        "risk": "риск",
        "case": "кейс",
        "service": "сервис",
    },
    "en": {
        "lifehack": "tip",
        "reglament": "rule",
        "howto": "how-to",
        "risk": "risk",
        "case": "case",
        "service": "service",
    },
    "es": {
        "lifehack": "truco",
        "reglament": "norma",
        "howto": "guía",
        "risk": "riesgo",
        "case": "caso",
        "service": "servicio",
    },
    "pt": {
        "lifehack": "dica",
        "reglament": "regra",
        "howto": "guia",
        "risk": "risco",
        "case": "caso",
        "service": "serviço",
    },
    "de": {
        "lifehack": "Tipp",
        "reglament": "Regel",
        "howto": "Anleitung",
        "risk": "Risiko",
        "case": "Fall",
        "service": "Service",
    },
    "fr": {
        "lifehack": "astuce",
        "reglament": "règle",
        "howto": "guide",
        "risk": "risque",
        "case": "cas",
        "service": "service",
    },
    "it": {
        "lifehack": "trucco",
        "reglament": "regola",
        "howto": "guida",
        "risk": "rischio",
        "case": "caso",
        "service": "servizio",
    },
    "tr": {
        "lifehack": "ipucu",
        "reglament": "kural",
        "howto": "rehber",
        "risk": "risk",
        "case": "vaka",
        "service": "servis",
    },
}
SHELF_MIN = 3  # полка становится страницей от 3 абзацев (мельче — тонковато)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pseo
DATA = f"{BASE}/data"
# built-данные лежат либо локально (pull с VPS), либо укажи путь
BUILT = os.environ.get("BUILT_DIR", f"{BASE}/builder")

GEO_NAMES = {
    "ru": {
        "br": "Бразилия",
        "vn": "Вьетнам",
        "me": "Черногория",
        "id": "Индонезия",
        "gr": "Греция",
        "kr": "Южная Корея",
        "ph": "Филиппины",
        "de": "Германия",
        "gb": "Великобритания",
        "bg": "Болгария",
        "jp": "Япония",
        "by": "Беларусь",
        "fr": "Франция",
        "au": "Австралия",
        "ar": "Аргентина",
        "hu": "Венгрия",
        "at": "Австрия",
        "ru": "Россия",
        "cl": "Чили",
        "fi": "Финляндия",
        "ge": "Грузия",
        "cz": "Чехия",
        "mu": "Маврикий",
        "lk": "Шри-Ланка",
        "be": "Бельгия",
        "ch": "Швейцария",
        "cn": "Китай",
        "cu": "Куба",
        "eg": "Египет",
        "hr": "Хорватия",
        "il": "Израиль",
        "in": "Индия",
        "kz": "Казахстан",
        "tr": "Турция",
        "kg": "Киргизия",
    },
    "en": {
        "br": "Brazil",
        "vn": "Vietnam",
        "me": "Montenegro",
        "id": "Indonesia",
        "gr": "Greece",
        "kr": "South Korea",
        "ph": "Philippines",
        "de": "Germany",
        "gb": "United Kingdom",
        "bg": "Bulgaria",
        "jp": "Japan",
        "by": "Belarus",
        "fr": "France",
        "au": "Australia",
        "ar": "Argentina",
        "hu": "Hungary",
        "at": "Austria",
        "ru": "Russia",
        "cl": "Chile",
        "fi": "Finland",
        "ge": "Georgia",
        "cz": "Czechia",
        "mu": "Mauritius",
        "lk": "Sri Lanka",
        "be": "Belgium",
        "ch": "Switzerland",
        "cn": "China",
        "cu": "Cuba",
        "eg": "Egypt",
        "hr": "Croatia",
        "il": "Israel",
        "in": "India",
        "kz": "Kazakhstan",
        "tr": "Turkey",
        "kg": "Kyrgyzstan",
    },
    "es": {
        "br": "Brasil",
        "vn": "Vietnam",
        "me": "Montenegro",
        "id": "Indonesia",
        "gr": "Grecia",
        "kr": "Corea del Sur",
        "ph": "Filipinas",
        "de": "Alemania",
        "gb": "Reino Unido",
        "bg": "Bulgaria",
        "jp": "Japón",
        "by": "Bielorrusia",
        "fr": "Francia",
        "au": "Australia",
        "ar": "Argentina",
        "hu": "Hungría",
        "at": "Austria",
        "ru": "Rusia",
        "cl": "Chile",
        "fi": "Finlandia",
        "ge": "Georgia",
        "cz": "Chequia",
        "mu": "Mauricio",
        "lk": "Sri Lanka",
        "be": "Bélgica",
        "ch": "Suiza",
        "cn": "China",
        "cu": "Cuba",
        "eg": "Egipto",
        "hr": "Croacia",
        "il": "Israel",
        "in": "India",
        "kz": "Kazajistán",
        "tr": "Turquía",
        "kg": "Kirguistán",
    },
    "pt": {
        "br": "Brasil",
        "vn": "Vietnã",
        "me": "Montenegro",
        "id": "Indonésia",
        "gr": "Grécia",
        "kr": "Coreia do Sul",
        "ph": "Filipinas",
        "de": "Alemanha",
        "gb": "Reino Unido",
        "bg": "Bulgária",
        "jp": "Japão",
        "by": "Bielorrússia",
        "fr": "França",
        "au": "Austrália",
        "ar": "Argentina",
        "hu": "Hungria",
        "at": "Áustria",
        "ru": "Rússia",
        "cl": "Chile",
        "fi": "Finlândia",
        "ge": "Geórgia",
        "cz": "Chéquia",
        "mu": "Maurício",
        "lk": "Sri Lanka",
        "be": "Bélgica",
        "ch": "Suíça",
        "cn": "China",
        "cu": "Cuba",
        "eg": "Egito",
        "hr": "Croácia",
        "il": "Israel",
        "in": "Índia",
        "kz": "Cazaquistão",
        "tr": "Turquia",
        "kg": "Quirguistão",
    },
    "de": {
        "br": "Brasilien",
        "vn": "Vietnam",
        "me": "Montenegro",
        "id": "Indonesien",
        "gr": "Griechenland",
        "kr": "Südkorea",
        "ph": "Philippinen",
        "de": "Deutschland",
        "gb": "Vereinigtes Königreich",
        "bg": "Bulgarien",
        "jp": "Japan",
        "by": "Belarus",
        "fr": "Frankreich",
        "au": "Australien",
        "ar": "Argentinien",
        "hu": "Ungarn",
        "at": "Österreich",
        "ru": "Russland",
        "cl": "Chile",
        "fi": "Finnland",
        "ge": "Georgien",
        "cz": "Tschechien",
        "mu": "Mauritius",
        "lk": "Sri Lanka",
        "be": "Belgien",
        "ch": "Schweiz",
        "cn": "China",
        "cu": "Kuba",
        "eg": "Ägypten",
        "hr": "Kroatien",
        "il": "Israel",
        "in": "Indien",
        "kz": "Kasachstan",
        "tr": "Türkei",
        "kg": "Kirgisistan",
    },
    "fr": {
        "br": "Brésil",
        "vn": "Vietnam",
        "me": "Monténégro",
        "id": "Indonésie",
        "gr": "Grèce",
        "kr": "Corée du Sud",
        "ph": "Philippines",
        "de": "Allemagne",
        "gb": "Royaume-Uni",
        "bg": "Bulgarie",
        "jp": "Japon",
        "by": "Biélorussie",
        "fr": "France",
        "au": "Australie",
        "ar": "Argentine",
        "hu": "Hongrie",
        "at": "Autriche",
        "ru": "Russie",
        "cl": "Chili",
        "fi": "Finlande",
        "ge": "Géorgie",
        "cz": "Tchéquie",
        "mu": "Maurice",
        "lk": "Sri Lanka",
        "be": "Belgique",
        "ch": "Suisse",
        "cn": "Chine",
        "cu": "Cuba",
        "eg": "Égypte",
        "hr": "Croatie",
        "il": "Israël",
        "in": "Inde",
        "kz": "Kazakhstan",
        "tr": "Turquie",
        "kg": "Kirghizistan",
    },
    "it": {
        "br": "Brasile",
        "vn": "Vietnam",
        "me": "Montenegro",
        "id": "Indonesia",
        "gr": "Grecia",
        "kr": "Corea del Sud",
        "ph": "Filippine",
        "de": "Germania",
        "gb": "Regno Unito",
        "bg": "Bulgaria",
        "jp": "Giappone",
        "by": "Bielorussia",
        "fr": "Francia",
        "au": "Australia",
        "ar": "Argentina",
        "hu": "Ungheria",
        "at": "Austria",
        "ru": "Russia",
        "cl": "Cile",
        "fi": "Finlandia",
        "ge": "Georgia",
        "cz": "Cechia",
        "mu": "Mauritius",
        "lk": "Sri Lanka",
        "be": "Belgio",
        "ch": "Svizzera",
        "cn": "Cina",
        "cu": "Cuba",
        "eg": "Egitto",
        "hr": "Croazia",
        "il": "Israele",
        "in": "India",
        "kz": "Kazakistan",
        "tr": "Turchia",
        "kg": "Kirghizistan",
    },
    "tr": {
        "br": "Brezilya",
        "vn": "Vietnam",
        "me": "Karadağ",
        "id": "Endonezya",
        "gr": "Yunanistan",
        "kr": "Güney Kore",
        "ph": "Filipinler",
        "de": "Almanya",
        "gb": "Birleşik Krallık",
        "bg": "Bulgaristan",
        "jp": "Japonya",
        "by": "Belarus",
        "fr": "Fransa",
        "au": "Avustralya",
        "ar": "Arjantin",
        "hu": "Macaristan",
        "at": "Avusturya",
        "ru": "Rusya",
        "cl": "Şili",
        "fi": "Finlandiya",
        "ge": "Gürcistan",
        "cz": "Çekya",
        "mu": "Mauritius",
        "lk": "Sri Lanka",
        "be": "Belçika",
        "ch": "İsviçre",
        "cn": "Çin",
        "cu": "Küba",
        "eg": "Mısır",
        "hr": "Hırvatistan",
        "il": "İsrail",
        "in": "Hindistan",
        "kz": "Kazakistan",
        "tr": "Türkiye",
        "kg": "Kırgızistan",
    },
}
# ru: «где» с предлогом (в/на + предложный падеж) — «в {name}» даёт «в Бразилия»
GEO_LOC = {
    "br": "в Бразилии",
    "vn": "во Вьетнаме",
    "me": "в Черногории",
    "id": "в Индонезии",
    "gr": "в Греции",
    "kr": "в Южной Корее",
    "ph": "на Филиппинах",
    "de": "в Германии",
    "gb": "в Великобритании",
    "bg": "в Болгарии",
    "jp": "в Японии",
    "by": "в Беларуси",
    "fr": "во Франции",
    "au": "в Австралии",
    "ar": "в Аргентине",
    "hu": "в Венгрии",
    "at": "в Австрии",
    "ru": "в России",
    "cl": "в Чили",
    "fi": "в Финляндии",
    "ge": "в Грузии",
    "cz": "в Чехии",
    "mu": "на Маврикии",
    "lk": "на Шри-Ланке",
    "be": "в Бельгии",
    "ch": "в Швейцарии",
    "cn": "в Китае",
    "cu": "на Кубе",
    "eg": "в Египте",
    "hr": "в Хорватии",
    "il": "в Израиле",
    "in": "в Индии",
    "kz": "в Казахстане",
    "tr": "в Турции",
    "kg": "в Киргизии",
}
GEO_FLAG = {
    "br": "🇧🇷",
    "vn": "🇻🇳",
    "me": "🇲🇪",
    "id": "🇮🇩",
    "gr": "🇬🇷",
    "kr": "🇰🇷",
    "ph": "🇵🇭",
    "de": "🇩🇪",
    "gb": "🇬🇧",
    "bg": "🇧🇬",
    "jp": "🇯🇵",
    "by": "🇧🇾",
    "fr": "🇫🇷",
    "au": "🇦🇺",
    "ar": "🇦🇷",
    "hu": "🇭🇺",
    "at": "🇦🇹",
    "ru": "🇷🇺",
    "cl": "🇨🇱",
    "fi": "🇫🇮",
    "ge": "🇬🇪",
    "cz": "🇨🇿",
    "mu": "🇲🇺",
    "lk": "🇱🇰",
    "be": "🇧🇪",
    "ch": "🇨🇭",
    "cn": "🇨🇳",
    "cu": "🇨🇺",
    "eg": "🇪🇬",
    "hr": "🇭🇷",
    "il": "🇮🇱",
    "in": "🇮🇳",
    "kz": "🇰🇿",
    "tr": "🇹🇷",
    "kg": "🇰🇬",
}
ICON = {
    "документ": "🛂",
    "виз": "🛂",
    "внж": "🛂",
    "деньг": "💰",
    "банк": "💰",
    "финанс": "💰",
    "обмен": "💱",
    "перевод": "💱",
    "жиль": "🏠",
    "аренд": "🏠",
    "безопас": "🛡",
    "транспорт": "🚕",
    "логист": "🚕",
    "здоров": "🩺",
    "медиц": "🩺",
    "прививк": "🩺",
    "покупк": "🛒",
    "связ": "📶",
    "интернет": "📶",
    "sim": "📶",
    "еда": "🍽",
    "пита": "🍽",
    "посмотреть": "🗺",
    "достоприм": "🗺",
    "путешеств": "🗺",
    "досуг": "🗺",
    "культур": "🗣",
    "язык": "🗣",
    "работ": "💼",
    "налог": "💼",
    "образован": "🎓",
    "почт": "📦",
    "посылк": "📦",
    "билет": "🎟",
    "развлеч": "🎟",
    # английские ключи (EN-метки)
    "document": "🛂",
    "visa": "🛂",
    "money": "💰",
    "bank": "💰",
    "financ": "💰",
    "exchange": "💱",
    "transfer": "💱",
    "hous": "🏠",
    "rent": "🏠",
    "safet": "🛡",
    "transport": "🚕",
    "logist": "🚕",
    "health": "🩺",
    "medic": "🩺",
    "shop": "🛒",
    "internet": "📶",
    "food": "🍽",
    "eat": "🍽",
    "sightsee": "🗺",
    "travel": "🗺",
    "leisure": "🗺",
    "cultur": "🗣",
    "languag": "🗣",
    "work": "💼",
    "tax": "💼",
    "educ": "🎓",
    "post": "📦",
    "parcel": "📦",
    "ticket": "🎟",
    "entertain": "🎟",
}

# Копия страниц по языкам. {name}=страна, {t}=тема, {n}=число. RU — дословно как было.
COPY = {
    "ru": {
        "FHEAD": [
            "{t} {gp}: живой опыт из чатов",
            "{t}: как это {gp} — из первых рук",
            "{t} {gp}: что реально важно знать",
        ],
        "QHEAD": [
            "{t}: что спрашивают в чатах",
            "{t}: частые вопросы из живых чатов",
            "{t}: что спрашивают в чатах часто, но не всегда получают ответ",
        ],
        "fact_title": "{name}: {tl} — живой опыт · Luky",
        "fact_desc": "Живой опыт из чатов про {tl} {namep}: как есть, из первых рук. Под твой случай — у Luky.",
        "fact_intro": "Реальный опыт людей из чатов по теме «{tl}» {namep} — как есть, без воды. Под свой случай — <a href='#luky'>спроси Luky</a>.",
        "fact_list_label": "Из живого опыта",
        "fact_blurb": "{n} {w} из чатов",
        "fact_w": ("совет", "совета", "советов"),
        "q_title": "{name}: {tl} — что спрашивают · Luky",
        "q_desc": "Реальные вопросы про {tl} в {name} из живых чатов. Ответ под твой случай — у Luky.",
        "q_intro": "Живые вопросы из чатов сообществ — с чем реально сталкиваются. Узнаёшь свой? Ответ под твой случай — <a href='#luky'>спроси Luky</a>.",
        "q_list_label": "Вопросы из чатов",
        "q_blurb": "{n} {w} из чатов",
        "q_w": ("вопрос", "вопроса", "вопросов"),
        "qhub_title": "{name}: что спрашивают в чатах — реальные вопросы · Luky",
        "qhub_desc": "Реальные вопросы про {name} из живых чатов: визы, деньги, жильё, безопасность. Ответ под твой случай — у Luky.",
        "qhub_h1": "Что спрашивают в чатах",
        "qhub_intro": "Сотни людей — одни и те же непонятки. Выбери тему, посмотри реальные вопросы. Ответы у людей находятся не сразу… а у <a href='#luky'>Luky</a> — сразу.",
        "bridge_title": "Что спрашивают в чатах",
        "bridge_blurb": "Реальные вопросы людей — под свой случай спроси Luky",
        "shelf_title": "{name}: {tl} — живой опыт из чатов · Luky",
        "shelf_desc": "Живой опыт по теме «{tl}» {namep}: реальные советы, случаи и правила из чатов. Под твой случай — у Luky.",
        "shelf_intro": "Собрано из живого опыта: «{tl}» {namep} — советы, случаи и правила как есть. Под свой случай — <a href='#luky'>спроси Luky</a>.",
        "shelf_list_label": "Из живого опыта",
        "shelf_blurb": "{n} {w} из чатов",
        "shelf_w": ("заметка", "заметки", "заметок"),
        "shub_title": "{name}: разделы живого опыта — всё из чатов · Luky",
        "shub_desc": "Живой опыт по {name} по разделам: визы, деньги, транспорт, документы, безопасность и другое. Под твой случай — у Luky.",
        "shub_h1": "Разделы живого опыта",
        "shub_intro": "Всё, что люди прошли сами — по разделам. Выбери свой, а под конкретный случай <a href='#luky'>спроси Luky</a>.",
        "bridge_shelf_title": "Разделы живого опыта",
        "bridge_shelf_blurb": "Реальные заметки по темам — под свой случай спроси Luky",
        "hub_title": "{name}: документы, деньги, жильё — живой опыт из чатов · Luky",
        "hub_desc": "Живой опыт по {name} из чатов сообществ: документы, деньги, жильё, безопасность, транспорт. Без воды, под твой случай.",
        "hub_intro": "Живой опыт тех, кто реально через это прошёл — по делу, без воды. Выбери тему, а под свой случай <a href='#luky'>спроси Luky</a>.",
        "list_label_topics": "Темы",
        "lower": True,  # темы в тайтле в нижнем регистре (русский стиль)
    },
    "en": {
        "FHEAD": [
            "{t} in {g}: real experience from chats",
            "{t}: how it works in {g} — first-hand",
            "{t} in {g}: what actually matters to know",
        ],
        "QHEAD": [
            "{t}: what people ask in chats",
            "{t}: common questions from live chats",
            "{t}: what people often ask in chats but don't always get answered",
        ],
        "fact_title": "{name}: {tl} — real experience · Luky",
        "fact_desc": "Real experience from chats about {tl} in {name}: as it is, first-hand. For your case — ask Luky.",
        "fact_intro": "Real experience of people from chats on «{tl}» in {name} — as it is, no fluff. For your case — <a href='#luky'>ask Luky</a>.",
        "fact_list_label": "From real experience",
        "shelf_title": "{name}: {tl} — real experience from chats · Luky",
        "shelf_desc": "Real experience on «{tl}» in {name}: hands-on tips, cases and rules from chats. For your case — ask Luky.",
        "shelf_intro": "Collected from real experience: «{tl}» in {name} — tips, cases and rules as they are. For your case — <a href='#luky'>ask Luky</a>.",
        "shelf_list_label": "From real experience",
        "fact_blurb": "{n} tips from chats",
        "shelf_blurb": "{n} notes from chats",
        "bridge_shelf_blurb": "Real notes by topic — for your case ask Luky",
        "bridge_shelf_title": "Sections of real experience",
        "shub_title": "{name}: sections of real experience — all from chats · Luky",
        "shub_desc": "Real experience in {name} by section: visas, money, transport, documents, safety and more. For your case — ask Luky.",
        "shub_h1": "Sections of real experience",
        "shub_intro": "Everything people went through themselves — by section. Pick yours, and for your specific case <a href='#luky'>ask Luky</a>.",
        "q_title": "{name}: {tl} — what people ask · Luky",
        "q_desc": "Real questions about {tl} in {name} from live chats. An answer for your case — ask Luky.",
        "q_intro": "Live questions from community chats — what people actually run into. Recognise yours? An answer for your case — <a href='#luky'>ask Luky</a>.",
        "q_list_label": "Questions from chats",
        "q_blurb": "{n} questions from chats",
        "qhub_title": "{name}: what people ask in chats — real questions · Luky",
        "qhub_desc": "Real questions about {name} from live chats: visas, money, housing, safety. An answer for your case — ask Luky.",
        "qhub_h1": "What people ask in chats",
        "qhub_intro": "Hundreds of people — the same confusions. Pick a topic, see the real questions. People find answers slowly… but <a href='#luky'>Luky</a> — right away.",
        "bridge_title": "What people ask in chats",
        "bridge_blurb": "Real questions from people — for your case ask Luky",
        "hub_title": "{name}: documents, money, housing — real experience from chats · Luky",
        "hub_desc": "Real experience for {name} from community chats: documents, money, housing, safety, transport. No fluff, for your case.",
        "hub_intro": "Real experience of those who actually went through it — to the point, no fluff. Pick a topic, and for your case <a href='#luky'>ask Luky</a>.",
        "list_label_topics": "Topics",
        "lower": False,  # английские заголовки — как есть (Title-case меток)
    },
    "es": {
        "FHEAD": [
            "{t} en {g}: experiencia real de los chats",
            "{t}: cómo es en {g} — de primera mano",
            "{t} en {g}: lo que de verdad importa saber",
        ],
        "QHEAD": [
            "{t}: qué preguntan en los chats",
            "{t}: preguntas frecuentes de chats reales",
            "{t}: lo que preguntan seguido pero no siempre responden",
        ],
        "fact_title": "{name}: {tl} — experiencia real · Luky",
        "fact_desc": "Experiencia real de los chats sobre {tl} en {name}: tal cual, de primera mano. Para tu caso — pregúntale a Luky.",
        "fact_intro": "Experiencia real de gente de los chats sobre «{tl}» en {name} — tal cual, sin relleno. Para tu caso — <a href='#luky'>pregúntale a Luky</a>.",
        "fact_list_label": "De la experiencia real",
        "shelf_title": "{name}: {tl} — experiencia real de los chats · Luky",
        "shelf_desc": "Experiencia real sobre «{tl}» en {name}: consejos, casos y normas de los chats. Para tu caso — pregúntale a Luky.",
        "shelf_intro": "Recopilado de la experiencia real: «{tl}» en {name} — consejos, casos y normas tal cual. Para tu caso — <a href='#luky'>pregúntale a Luky</a>.",
        "shelf_list_label": "De la experiencia real",
        "fact_blurb": "{n} consejos de los chats",
        "shelf_blurb": "{n} notas de los chats",
        "bridge_shelf_blurb": "Notas reales por tema — para tu caso pregúntale a Luky",
        "bridge_shelf_title": "Secciones de experiencia real",
        "shub_title": "{name}: secciones de experiencia real — todo de los chats · Luky",
        "shub_desc": "Experiencia real en {name} por secciones: visados, dinero, transporte, documentos, seguridad y más. Para tu caso — pregúntale a Luky.",
        "shub_h1": "Secciones de experiencia real",
        "shub_intro": "Todo lo que la gente vivió en persona — por secciones. Elige la tuya, y para tu caso concreto <a href='#luky'>pregúntale a Luky</a>.",
        "q_title": "{name}: {tl} — qué preguntan · Luky",
        "q_desc": "Preguntas reales sobre {tl} en {name} de chats en vivo. Una respuesta para tu caso — pregúntale a Luky.",
        "q_intro": "Preguntas en vivo de chats de comunidades — con lo que la gente realmente se topa. ¿Reconoces la tuya? Una respuesta para tu caso — <a href='#luky'>pregúntale a Luky</a>.",
        "q_list_label": "Preguntas de los chats",
        "q_blurb": "{n} preguntas de los chats",
        "qhub_title": "{name}: qué preguntan en los chats — preguntas reales · Luky",
        "qhub_desc": "Preguntas reales sobre {name} de chats en vivo: visas, dinero, vivienda, seguridad. Una respuesta para tu caso — pregúntale a Luky.",
        "qhub_h1": "Qué preguntan en los chats",
        "qhub_intro": "Cientos de personas — las mismas dudas. Elige un tema, mira las preguntas reales. La gente encuentra respuestas despacio… pero <a href='#luky'>Luky</a> — al instante.",
        "bridge_title": "Qué preguntan en los chats",
        "bridge_blurb": "Preguntas reales de la gente — para tu caso pregúntale a Luky",
        "hub_title": "{name}: documentos, dinero, vivienda — experiencia real de los chats · Luky",
        "hub_desc": "Experiencia real de {name} de chats de comunidades: documentos, dinero, vivienda, seguridad, transporte. Sin relleno, para tu caso.",
        "hub_intro": "Experiencia real de quienes ya pasaron por ello — al grano, sin relleno. Elige un tema, y para tu caso <a href='#luky'>pregúntale a Luky</a>.",
        "list_label_topics": "Temas",
        "lower": False,
    },
    "pt": {
        "FHEAD": [
            "{t} em {g}: experiência real dos chats",
            "{t}: como é em {g} — em primeira mão",
            "{t} em {g}: o que realmente importa saber",
        ],
        "QHEAD": [
            "{t}: o que perguntam nos chats",
            "{t}: perguntas frequentes de chats reais",
            "{t}: o que perguntam com frequência mas nem sempre respondem",
        ],
        "fact_title": "{name}: {tl} — experiência real · Luky",
        "fact_desc": "Experiência real dos chats sobre {tl} em {name}: como é, em primeira mão. Para o seu caso — pergunte ao Luky.",
        "fact_intro": "Experiência real de pessoas dos chats sobre «{tl}» em {name} — como é, sem enrolação. Para o seu caso — <a href='#luky'>pergunte ao Luky</a>.",
        "fact_list_label": "Da experiência real",
        "shelf_title": "{name}: {tl} — experiência real dos chats · Luky",
        "shelf_desc": "Experiência real sobre «{tl}» em {name}: dicas, casos e regras dos chats. Para o seu caso — pergunte ao Luky.",
        "shelf_intro": "Reunido da experiência real: «{tl}» em {name} — dicas, casos e regras como são. Para o seu caso — <a href='#luky'>pergunte ao Luky</a>.",
        "shelf_list_label": "Da experiência real",
        "fact_blurb": "{n} dicas dos chats",
        "shelf_blurb": "{n} notas dos chats",
        "bridge_shelf_blurb": "Notas reais por tema — para o seu caso pergunte ao Luky",
        "bridge_shelf_title": "Seções de experiência real",
        "shub_title": "{name}: seções de experiência real — tudo dos chats · Luky",
        "shub_desc": "Experiência real em {name} por seção: vistos, dinheiro, transporte, documentos, segurança e mais. Para o seu caso — pergunte ao Luky.",
        "shub_h1": "Seções de experiência real",
        "shub_intro": "Tudo o que as pessoas viveram por conta própria — por seção. Escolha a sua, e para o seu caso específico <a href='#luky'>pergunte ao Luky</a>.",
        "q_title": "{name}: {tl} — o que perguntam · Luky",
        "q_desc": "Perguntas reais sobre {tl} em {name} de chats ao vivo. Uma resposta para o seu caso — pergunte ao Luky.",
        "q_intro": "Perguntas ao vivo de chats de comunidades — com o que as pessoas realmente se deparam. Reconhece a sua? Uma resposta para o seu caso — <a href='#luky'>pergunte ao Luky</a>.",
        "q_list_label": "Perguntas dos chats",
        "q_blurb": "{n} perguntas dos chats",
        "qhub_title": "{name}: o que perguntam nos chats — perguntas reais · Luky",
        "qhub_desc": "Perguntas reais sobre {name} de chats ao vivo: vistos, dinheiro, moradia, segurança. Uma resposta para o seu caso — pergunte ao Luky.",
        "qhub_h1": "O que perguntam nos chats",
        "qhub_intro": "Centenas de pessoas — as mesmas dúvidas. Escolha um tema, veja as perguntas reais. As pessoas acham respostas devagar… mas o <a href='#luky'>Luky</a> — na hora.",
        "bridge_title": "O que perguntam nos chats",
        "bridge_blurb": "Perguntas reais das pessoas — para o seu caso pergunte ao Luky",
        "hub_title": "{name}: documentos, dinheiro, moradia — experiência real dos chats · Luky",
        "hub_desc": "Experiência real de {name} de chats de comunidades: documentos, dinheiro, moradia, segurança, transporte. Sem enrolação, para o seu caso.",
        "hub_intro": "Experiência real de quem já passou por isso — direto ao ponto, sem enrolação. Escolha um tema, e para o seu caso <a href='#luky'>pergunte ao Luky</a>.",
        "list_label_topics": "Temas",
        "lower": False,
    },
    "de": {
        "FHEAD": [
            "{t} in {g}: echte Erfahrungen aus Chats",
            "{t}: wie es in {g} wirklich läuft — aus erster Hand",
            "{t} in {g}: was man wirklich wissen muss",
        ],
        "QHEAD": [
            "{t}: was in Chats gefragt wird",
            "{t}: häufige Fragen aus lebendigen Chats",
            "{t}: was oft gefragt, aber selten beantwortet wird",
        ],
        "fact_title": "{name}: {tl} — echte Erfahrungen · Luky",
        "fact_desc": "Echte Erfahrungen aus Chats zu {tl} in {name}: so wie es ist, aus erster Hand. Für deinen Fall — frag Luky.",
        "fact_intro": "Echte Erfahrungen von Leuten aus Chats zum Thema «{tl}» in {name} — ohne Geschwätz. Für deinen Fall — <a href='#luky'>frag Luky</a>.",
        "fact_list_label": "Aus echter Erfahrung",
        "shelf_title": "{name}: {tl} — echte Erfahrungen aus Chats · Luky",
        "shelf_desc": "Echte Erfahrungen zu «{tl}» in {name}: praktische Tipps, Fälle und Regeln aus Chats. Für deinen Fall — frag Luky.",
        "shelf_intro": "Gesammelt aus echter Erfahrung: «{tl}» in {name} — Tipps, Fälle und Regeln, so wie sie sind. Für deinen Fall — <a href='#luky'>frag Luky</a>.",
        "shelf_list_label": "Aus echter Erfahrung",
        "fact_blurb": "{n} Tipps aus Chats",
        "shelf_blurb": "{n} Notizen aus Chats",
        "bridge_shelf_blurb": "Echte Notizen nach Thema — für deinen Fall frag Luky",
        "bridge_shelf_title": "Bereiche echter Erfahrung",
        "shub_title": "{name}: Bereiche echter Erfahrung — alles aus Chats · Luky",
        "shub_desc": "Echte Erfahrungen in {name} nach Bereichen: Visa, Geld, Transport, Dokumente, Sicherheit und mehr. Für deinen Fall — frag Luky.",
        "shub_h1": "Bereiche echter Erfahrung",
        "shub_intro": "Alles, was Leute selbst durchgemacht haben — nach Bereichen. Wähle deinen, und für deinen konkreten Fall <a href='#luky'>frag Luky</a>.",
        "q_title": "{name}: {tl} — was gefragt wird · Luky",
        "q_desc": "Echte Fragen zu {tl} in {name} aus lebendigen Chats. Eine Antwort für deinen Fall — frag Luky.",
        "q_intro": "Lebendige Fragen aus Community-Chats — worauf Leute wirklich stoßen. Kennst du das? Antwort für deinen Fall — <a href='#luky'>frag Luky</a>.",
        "q_list_label": "Fragen aus Chats",
        "q_blurb": "{n} Fragen aus Chats",
        "qhub_title": "{name}: was in Chats gefragt wird — echte Fragen · Luky",
        "qhub_desc": "Echte Fragen zu {name} aus lebendigen Chats: Visa, Geld, Wohnen, Sicherheit. Antwort für deinen Fall — frag Luky.",
        "qhub_h1": "Was in Chats gefragt wird",
        "qhub_intro": "Hunderte Leute — dieselben Unklarheiten. Wähle ein Thema und sieh die echten Fragen. Leute finden Antworten langsam… aber <a href='#luky'>Luky</a> — sofort.",
        "bridge_title": "Was in Chats gefragt wird",
        "bridge_blurb": "Echte Fragen von Leuten — für deinen Fall frag Luky",
        "hub_title": "{name}: Dokumente, Geld, Wohnen — echte Erfahrungen aus Chats · Luky",
        "hub_desc": "Echte Erfahrungen für {name} aus Community-Chats: Dokumente, Geld, Wohnen, Sicherheit, Transport. Ohne Geschwätz, für deinen Fall.",
        "hub_intro": "Echte Erfahrungen von denen, die es selbst durchgemacht haben — auf den Punkt, ohne Geschwätz. Wähle ein Thema, und für deinen Fall <a href='#luky'>frag Luky</a>.",
        "list_label_topics": "Themen",
        "lower": False,
    },
    "fr": {
        "FHEAD": [
            "{t} en {g} : l'expérience réelle des chats",
            "{t} : comment ça marche vraiment en {g} — de première main",
            "{t} en {g} : ce qu'il faut vraiment savoir",
        ],
        "QHEAD": [
            "{t} : ce que les gens demandent dans les chats",
            "{t} : questions fréquentes des chats en direct",
            "{t} : ce qu'on demande souvent sans obtenir de réponse",
        ],
        "fact_title": "{name} : {tl} — expérience réelle · Luky",
        "fact_desc": "L'expérience réelle des chats sur {tl} en {name} : telle quelle, de première main. Pour ton cas — demande à Luky.",
        "fact_intro": "L'expérience réelle de gens des chats sur « {tl} » en {name} — telle quelle, sans blabla. Pour ton cas — <a href='#luky'>demande à Luky</a>.",
        "fact_list_label": "D'après l'expérience réelle",
        "shelf_title": "{name} : {tl} — expérience réelle des chats · Luky",
        "shelf_desc": "L'expérience réelle sur « {tl} » en {name} : conseils pratiques, cas et règles issus des chats. Pour ton cas — demande à Luky.",
        "shelf_intro": "Rassemblé à partir de l'expérience réelle : « {tl} » en {name} — conseils, cas et règles tels quels. Pour ton cas — <a href='#luky'>demande à Luky</a>.",
        "shelf_list_label": "D'après l'expérience réelle",
        "fact_blurb": "{n} conseils des chats",
        "shelf_blurb": "{n} notes des chats",
        "bridge_shelf_blurb": "De vraies notes par thème — pour ton cas demande à Luky",
        "bridge_shelf_title": "Sections d'expérience réelle",
        "shub_title": "{name} : sections d'expérience réelle — tout vient des chats · Luky",
        "shub_desc": "L'expérience réelle en {name} par section : visas, argent, transport, documents, sécurité et plus. Pour ton cas — demande à Luky.",
        "shub_h1": "Sections d'expérience réelle",
        "shub_intro": "Tout ce que les gens ont vécu eux-mêmes — par section. Choisis la tienne, et pour ton cas précis <a href='#luky'>demande à Luky</a>.",
        "q_title": "{name} : {tl} — ce qu'on demande · Luky",
        "q_desc": "De vraies questions sur {tl} en {name} issues des chats en direct. Une réponse pour ton cas — demande à Luky.",
        "q_intro": "Questions vivantes des chats communautaires — ce que les gens rencontrent vraiment. Ça te parle ? Une réponse pour ton cas — <a href='#luky'>demande à Luky</a>.",
        "q_list_label": "Questions des chats",
        "q_blurb": "{n} questions des chats",
        "qhub_title": "{name} : ce qu'on demande dans les chats — vraies questions · Luky",
        "qhub_desc": "De vraies questions sur {name} issues des chats en direct : visas, argent, logement, sécurité. Une réponse pour ton cas — demande à Luky.",
        "qhub_h1": "Ce qu'on demande dans les chats",
        "qhub_intro": "Des centaines de personnes — les mêmes casse-têtes. Choisis un thème et vois les vraies questions. Les gens trouvent lentement… mais <a href='#luky'>Luky</a> — tout de suite.",
        "bridge_title": "Ce qu'on demande dans les chats",
        "bridge_blurb": "De vraies questions de gens — pour ton cas demande à Luky",
        "hub_title": "{name} : documents, argent, logement — expérience réelle des chats · Luky",
        "hub_desc": "L'expérience réelle pour {name} issue des chats communautaires : documents, argent, logement, sécurité, transport. Sans blabla, pour ton cas.",
        "hub_intro": "L'expérience réelle de ceux qui l'ont vécu — droit au but, sans blabla. Choisis un thème, et pour ton cas <a href='#luky'>demande à Luky</a>.",
        "list_label_topics": "Thèmes",
        "lower": False,
    },
    "it": {
        "FHEAD": [
            "{t} in {g}: l'esperienza reale dalle chat",
            "{t}: come funziona davvero in {g} — di prima mano",
            "{t} in {g}: cosa conta davvero sapere",
        ],
        "QHEAD": [
            "{t}: cosa si chiede nelle chat",
            "{t}: domande frequenti dalle chat dal vivo",
            "{t}: quello che si chiede spesso e resta senza risposta",
        ],
        "fact_title": "{name}: {tl} — esperienza reale · Luky",
        "fact_desc": "Esperienza reale dalle chat su {tl} in {name}: così com'è, di prima mano. Per il tuo caso — chiedi a Luky.",
        "fact_intro": "L'esperienza reale delle persone dalle chat sul tema «{tl}» in {name} — così com'è, senza fronzoli. Per il tuo caso — <a href='#luky'>chiedi a Luky</a>.",
        "fact_list_label": "Dall'esperienza reale",
        "shelf_title": "{name}: {tl} — esperienza reale dalle chat · Luky",
        "shelf_desc": "Esperienza reale su «{tl}» in {name}: consigli pratici, casi e regole dalle chat. Per il tuo caso — chiedi a Luky.",
        "shelf_intro": "Raccolto dall'esperienza reale: «{tl}» in {name} — consigli, casi e regole così come sono. Per il tuo caso — <a href='#luky'>chiedi a Luky</a>.",
        "shelf_list_label": "Dall'esperienza reale",
        "fact_blurb": "{n} consigli dalle chat",
        "shelf_blurb": "{n} appunti dalle chat",
        "bridge_shelf_blurb": "Appunti reali per tema — per il tuo caso chiedi a Luky",
        "bridge_shelf_title": "Sezioni di esperienza reale",
        "shub_title": "{name}: sezioni di esperienza reale — tutto dalle chat · Luky",
        "shub_desc": "Esperienza reale in {name} per sezioni: visti, soldi, trasporti, documenti, sicurezza e altro. Per il tuo caso — chiedi a Luky.",
        "shub_h1": "Sezioni di esperienza reale",
        "shub_intro": "Tutto quello che le persone hanno passato in prima persona — per sezioni. Scegli la tua, e per il tuo caso specifico <a href='#luky'>chiedi a Luky</a>.",
        "q_title": "{name}: {tl} — cosa si chiede · Luky",
        "q_desc": "Domande reali su {tl} in {name} dalle chat dal vivo. Una risposta per il tuo caso — chiedi a Luky.",
        "q_intro": "Domande vive dalle chat della community — quello che le persone incontrano davvero. Ti suona familiare? Una risposta per il tuo caso — <a href='#luky'>chiedi a Luky</a>.",
        "q_list_label": "Domande dalle chat",
        "q_blurb": "{n} domande dalle chat",
        "qhub_title": "{name}: cosa si chiede nelle chat — domande reali · Luky",
        "qhub_desc": "Domande reali su {name} dalle chat dal vivo: visti, soldi, casa, sicurezza. Una risposta per il tuo caso — chiedi a Luky.",
        "qhub_h1": "Cosa si chiede nelle chat",
        "qhub_intro": "Centinaia di persone — gli stessi dubbi. Scegli un tema e guarda le domande reali. Le persone trovano risposte lentamente… ma <a href='#luky'>Luky</a> — subito.",
        "bridge_title": "Cosa si chiede nelle chat",
        "bridge_blurb": "Domande reali di persone — per il tuo caso chiedi a Luky",
        "hub_title": "{name}: documenti, soldi, casa — esperienza reale dalle chat · Luky",
        "hub_desc": "Esperienza reale per {name} dalle chat della community: documenti, soldi, casa, sicurezza, trasporti. Senza fronzoli, per il tuo caso.",
        "hub_intro": "L'esperienza reale di chi ci è passato davvero — diretta al punto, senza fronzoli. Scegli un tema, e per il tuo caso <a href='#luky'>chiedi a Luky</a>.",
        "list_label_topics": "Temi",
        "lower": False,
    },
    "tr": {
        "FHEAD": [
            "{g} için {t}: sohbetlerden gerçek deneyim",
            "{t}: {g} için gerçekte nasıl işliyor — ilk elden",
            "{g} için {t}: gerçekten bilmen gerekenler",
        ],
        "QHEAD": [
            "{t}: sohbetlerde neler soruluyor",
            "{t}: canlı sohbetlerden sık sorulanlar",
            "{t}: sık sorulup çoğu zaman yanıtsız kalanlar",
        ],
        "fact_title": "{name}: {tl} — gerçek deneyim · Luky",
        "fact_desc": "{name} için {tl} konusunda sohbetlerden gerçek deneyim: olduğu gibi, ilk elden. Kendi durumun için — Luky'ye sor.",
        "fact_intro": "{name} hakkında «{tl}» konusunda insanların sohbetlerdeki gerçek deneyimi — olduğu gibi, laf kalabalığı yok. Kendi durumun için — <a href='#luky'>Luky'ye sor</a>.",
        "fact_list_label": "Gerçek deneyimden",
        "shelf_title": "{name}: {tl} — sohbetlerden gerçek deneyim · Luky",
        "shelf_desc": "{name} için «{tl}» konusunda gerçek deneyim: sohbetlerden pratik ipuçları, vakalar ve kurallar. Kendi durumun için — Luky'ye sor.",
        "shelf_intro": "Gerçek deneyimden derlendi: {name} için «{tl}» — ipuçları, vakalar ve kurallar olduğu gibi. Kendi durumun için — <a href='#luky'>Luky'ye sor</a>.",
        "shelf_list_label": "Gerçek deneyimden",
        "fact_blurb": "sohbetlerden {n} ipucu",
        "shelf_blurb": "sohbetlerden {n} not",
        "bridge_shelf_blurb": "Konuya göre gerçek notlar — kendi durumun için Luky'ye sor",
        "bridge_shelf_title": "Gerçek deneyim bölümleri",
        "shub_title": "{name}: gerçek deneyim bölümleri — hepsi sohbetlerden · Luky",
        "shub_desc": "{name} için bölüm bölüm gerçek deneyim: vize, para, ulaşım, belgeler, güvenlik ve daha fazlası. Kendi durumun için — Luky'ye sor.",
        "shub_h1": "Gerçek deneyim bölümleri",
        "shub_intro": "İnsanların bizzat yaşadığı her şey — bölüm bölüm. Kendine uygun olanı seç, kendi somut durumun için <a href='#luky'>Luky'ye sor</a>.",
        "q_title": "{name}: {tl} — neler soruluyor · Luky",
        "q_desc": "{name} için {tl} konusunda canlı sohbetlerden gerçek sorular. Kendi durumuna yanıt — Luky'ye sor.",
        "q_intro": "Topluluk sohbetlerinden canlı sorular — insanların gerçekten karşılaştıkları. Tanıdık geldi mi? Kendi durumuna yanıt — <a href='#luky'>Luky'ye sor</a>.",
        "q_list_label": "Sohbetlerden sorular",
        "q_blurb": "sohbetlerden {n} soru",
        "qhub_title": "{name}: sohbetlerde neler soruluyor — gerçek sorular · Luky",
        "qhub_desc": "{name} hakkında canlı sohbetlerden gerçek sorular: vize, para, konut, güvenlik. Kendi durumuna yanıt — Luky'ye sor.",
        "qhub_h1": "Sohbetlerde neler soruluyor",
        "qhub_intro": "Yüzlerce insan — aynı kafa karışıklıkları. Bir konu seç, gerçek soruları gör. İnsanlar yanıtı yavaş buluyor… ama <a href='#luky'>Luky</a> — hemen.",
        "bridge_title": "Sohbetlerde neler soruluyor",
        "bridge_blurb": "İnsanlardan gerçek sorular — kendi durumun için Luky'ye sor",
        "hub_title": "{name}: belgeler, para, konut — sohbetlerden gerçek deneyim · Luky",
        "hub_desc": "{name} için topluluk sohbetlerinden gerçek deneyim: belgeler, para, konut, güvenlik, ulaşım. Laf kalabalığı yok, kendi durumun için.",
        "hub_intro": "Bunu bizzat yaşamış olanların gerçek deneyimi — doğrudan konuya, laf kalabalığı yok. Bir konu seç, kendi durumun için <a href='#luky'>Luky'ye sor</a>.",
        "list_label_topics": "Konular",
        "lower": False,
    },
}

# home + about по языкам (нав ссылается на /<lang>/ и /<lang>/about/)
HOME_ABOUT = {
    "ru": {
        "home_title": "Luky — живой опыт по странам: деньги, документы, жильё",
        "home_desc": "Инфопортал Luky: реальный опыт из чатов сообществ по странам — деньги, документы, жильё, безопасность. Без воды, под твой случай.",
        "home_h1": "Куда едешь?",
        "home_intro": "Реальный опыт тех, кто уже прошёл через местные непонятки — по делу, без воды. Выбери страну, а под свой случай <a href='#luky'>спроси Luky</a>.",
        "home_list_label": "Страны",
        "geo_blurb": "живой опыт",
        "about_crumb": "О проекте",
        "about_title": "О проекте · Luky",
        "about_desc": "Luky собирает живой опыт из открытых чатов сообществ по странам — по делу, без воды.",
        "about_h1": "О проекте",
        "about_body": "<p><a href='#luky'>Luky</a> — это опыт живых людей, а не сухая теория. Мы собираем реальные советы из открытых чатов сообществ: как что устроено на месте, чего избегать, что работает сейчас.</p><p>Портал — витрина этого опыта по темам и странам. А под твой конкретный случай можно спросить <a href='#luky'>Luky</a> — он подскажет по недавним отзывам людей.</p>",
    },
    "en": {
        "home_title": "Luky — real experience by country: money, documents, housing",
        "home_desc": "Luky info portal: real experience from community chats by country — money, documents, housing, safety. No fluff, for your case.",
        "home_h1": "Where are you headed?",
        "home_intro": "Real experience of those who already went through the local confusions — to the point, no fluff. Pick a country, and for your case <a href='#luky'>ask Luky</a>.",
        "home_list_label": "Countries",
        "geo_blurb": "real experience",
        "about_crumb": "About",
        "about_title": "About · Luky",
        "about_desc": "Luky gathers real experience from open community chats by country — to the point, no fluff.",
        "about_h1": "About",
        "about_body": "<p><a href='#luky'>Luky</a> is the experience of real people, not dry theory. We gather real tips from open community chats: how things actually work on the ground, what to avoid, what works right now.</p><p>The portal is a showcase of that experience by topic and country. And for your specific case you can <a href='#luky'>ask Luky</a> — it answers from people's recent reports.</p>",
    },
    "es": {
        "home_title": "Luky — experiencia real por país: dinero, documentos, vivienda",
        "home_desc": "Portal de info Luky: experiencia real de chats de comunidades por país — dinero, documentos, vivienda, seguridad. Sin relleno, para tu caso.",
        "home_h1": "¿A dónde vas?",
        "home_intro": "Experiencia real de quienes ya pasaron por los líos locales — al grano, sin relleno. Elige un país, y para tu caso <a href='#luky'>pregúntale a Luky</a>.",
        "home_list_label": "Países",
        "geo_blurb": "experiencia real",
        "about_crumb": "Acerca de",
        "about_title": "Acerca de · Luky",
        "about_desc": "Luky reúne experiencia real de chats abiertos de comunidades por país — al grano, sin relleno.",
        "about_h1": "Acerca de",
        "about_body": "<p><a href='#luky'>Luky</a> es la experiencia de gente real, no teoría seca. Reunimos consejos reales de chats abiertos de comunidades: cómo funcionan las cosas sobre el terreno, qué evitar, qué funciona ahora mismo.</p><p>El portal es un escaparate de esa experiencia por tema y país. Y para tu caso concreto puedes <a href='#luky'>preguntarle a Luky</a> — responde según reportes recientes de la gente.</p>",
    },
    "pt": {
        "home_title": "Luky — experiência real por país: dinheiro, documentos, moradia",
        "home_desc": "Portal de info Luky: experiência real de chats de comunidades por país — dinheiro, documentos, moradia, segurança. Sem enrolação, para o seu caso.",
        "home_h1": "Para onde você vai?",
        "home_intro": "Experiência real de quem já passou pelas confusões locais — direto ao ponto, sem enrolação. Escolha um país, e para o seu caso <a href='#luky'>pergunte ao Luky</a>.",
        "home_list_label": "Países",
        "geo_blurb": "experiência real",
        "about_crumb": "Sobre",
        "about_title": "Sobre · Luky",
        "about_desc": "O Luky reúne experiência real de chats abertos de comunidades por país — direto ao ponto, sem enrolação.",
        "about_h1": "Sobre",
        "about_body": "<p><a href='#luky'>Luky</a> é a experiência de pessoas reais, não teoria seca. Reunimos dicas reais de chats abertos de comunidades: como as coisas funcionam na prática, o que evitar, o que funciona agora.</p><p>O portal é uma vitrine dessa experiência por tema e país. E para o seu caso concreto você pode <a href='#luky'>perguntar ao Luky</a> — ele responde a partir de relatos recentes das pessoas.</p>",
    },
    "de": {
        "home_title": "Luky — echte Erfahrungen nach Land: Geld, Dokumente, Wohnen",
        "home_desc": "Luky Infoportal: echte Erfahrungen aus Community-Chats nach Land — Geld, Dokumente, Wohnen, Sicherheit. Ohne Geschwätz, für deinen Fall.",
        "home_h1": "Wohin geht es?",
        "home_intro": "Echte Erfahrungen von denen, die die örtlichen Unklarheiten schon hinter sich haben — auf den Punkt, ohne Geschwätz. Wähle ein Land, und für deinen Fall <a href='#luky'>frag Luky</a>.",
        "home_list_label": "Länder",
        "geo_blurb": "echte Erfahrungen",
        "about_crumb": "Über uns",
        "about_title": "Über uns · Luky",
        "about_desc": "Luky sammelt echte Erfahrungen aus offenen Community-Chats nach Land — auf den Punkt, ohne Geschwätz.",
        "about_h1": "Über uns",
        "about_body": "<p><a href='#luky'>Luky</a> ist die Erfahrung echter Menschen, keine trockene Theorie. Wir sammeln echte Tipps aus offenen Community-Chats: wie es vor Ort wirklich läuft, was man meiden sollte, was gerade funktioniert.</p><p>Das Portal ist ein Schaufenster dieser Erfahrung nach Thema und Land. Und für deinen konkreten Fall kannst du <a href='#luky'>Luky fragen</a> — es antwortet aus den aktuellen Berichten der Leute.</p>",
    },
    "fr": {
        "home_title": "Luky — l'expérience réelle par pays : argent, documents, logement",
        "home_desc": "Portail d'info Luky : l'expérience réelle des chats communautaires par pays — argent, documents, logement, sécurité. Sans blabla, pour ton cas.",
        "home_h1": "Où vas-tu ?",
        "home_intro": "L'expérience réelle de ceux qui ont déjà traversé les casse-têtes locaux — droit au but, sans blabla. Choisis un pays, et pour ton cas <a href='#luky'>demande à Luky</a>.",
        "home_list_label": "Pays",
        "geo_blurb": "expérience réelle",
        "about_crumb": "À propos",
        "about_title": "À propos · Luky",
        "about_desc": "Luky rassemble l'expérience réelle des chats communautaires ouverts par pays — droit au but, sans blabla.",
        "about_h1": "À propos",
        "about_body": "<p><a href='#luky'>Luky</a>, c'est l'expérience de vraies personnes, pas de la théorie sèche. Nous rassemblons de vrais conseils issus de chats communautaires ouverts : comment ça se passe vraiment sur place, ce qu'il faut éviter, ce qui marche en ce moment.</p><p>Le portail est une vitrine de cette expérience par thème et par pays. Et pour ton cas précis tu peux <a href='#luky'>demander à Luky</a> — il répond à partir des retours récents des gens.</p>",
    },
    "it": {
        "home_title": "Luky — esperienza reale per paese: soldi, documenti, casa",
        "home_desc": "Portale info Luky: esperienza reale dalle chat della community per paese — soldi, documenti, casa, sicurezza. Senza fronzoli, per il tuo caso.",
        "home_h1": "Dove sei diretto?",
        "home_intro": "L'esperienza reale di chi ha già superato i grattacapi locali — diretta al punto, senza fronzoli. Scegli un paese, e per il tuo caso <a href='#luky'>chiedi a Luky</a>.",
        "home_list_label": "Paesi",
        "geo_blurb": "esperienza reale",
        "about_crumb": "Chi siamo",
        "about_title": "Chi siamo · Luky",
        "about_desc": "Luky raccoglie l'esperienza reale dalle chat aperte della community per paese — diretta al punto, senza fronzoli.",
        "about_h1": "Chi siamo",
        "about_body": "<p><a href='#luky'>Luky</a> è l'esperienza di persone reali, non teoria astratta. Raccogliamo consigli veri dalle chat aperte della community: come funzionano le cose sul posto, cosa evitare, cosa funziona proprio ora.</p><p>Il portale è una vitrina di questa esperienza per tema e per paese. E per il tuo caso specifico puoi <a href='#luky'>chiedere a Luky</a> — risponde partendo dalle segnalazioni recenti delle persone.</p>",
    },
    "tr": {
        "home_title": "Luky — ülkeye göre gerçek deneyim: para, belgeler, konut",
        "home_desc": "Luky bilgi portalı: ülkeye göre topluluk sohbetlerinden gerçek deneyim — para, belgeler, konut, güvenlik. Laf kalabalığı yok, kendi durumun için.",
        "home_h1": "Yolun nereye?",
        "home_intro": "Yerel kafa karışıklıklarını çoktan aşmış olanların gerçek deneyimi — doğrudan konuya, laf kalabalığı yok. Bir ülke seç, kendi durumun için <a href='#luky'>Luky'ye sor</a>.",
        "home_list_label": "Ülkeler",
        "geo_blurb": "gerçek deneyim",
        "about_crumb": "Hakkında",
        "about_title": "Hakkında · Luky",
        "about_desc": "Luky, ülkeye göre açık topluluk sohbetlerinden gerçek deneyimi derler — doğrudan konuya, laf kalabalığı yok.",
        "about_h1": "Hakkında",
        "about_body": "<p><a href='#luky'>Luky</a>, kuru teori değil gerçek insanların deneyimidir. Açık topluluk sohbetlerinden gerçek ipuçları derliyoruz: yerinde işler gerçekte nasıl yürüyor, neden kaçınmalı, şu anda ne işe yarıyor.</p><p>Portal, bu deneyimin konuya ve ülkeye göre vitrinidir. Kendi somut durumun için ise <a href='#luky'>Luky'ye sorabilirsin</a> — insanların son bildirimlerinden yanıtlar.</p>",
    },
}


# ── Портал-home: регионы + образные вайбы стран ──
REGION_ORDER = ["la", "eu", "asia", "mea", "cis", "oce"]
CODE2REGION = {
    "br": "la",
    "ar": "la",
    "cl": "la",
    "cu": "la",
    "de": "eu",
    "gb": "eu",
    "fr": "eu",
    "at": "eu",
    "be": "eu",
    "ch": "eu",
    "cz": "eu",
    "bg": "eu",
    "hu": "eu",
    "fi": "eu",
    "hr": "eu",
    "gr": "eu",
    "me": "eu",
    "vn": "asia",
    "id": "asia",
    "ph": "asia",
    "kr": "asia",
    "jp": "asia",
    "cn": "asia",
    "in": "asia",
    "lk": "asia",
    "kz": "asia",
    "kg": "asia",
    "ge": "asia",
    "tr": "mea",
    "eg": "mea",
    "il": "mea",
    "mu": "mea",
    "ru": "cis",
    "by": "cis",
    "au": "oce",
}
REGION_NAMES = {
    "ru": {
        "la": "Латинская Америка",
        "eu": "Европа",
        "asia": "Азия",
        "mea": "Ближний Восток и Африка",
        "cis": "СНГ",
        "oce": "Океания",
        "oth": "Другие",
    },
    "en": {
        "la": "Latin America",
        "eu": "Europe",
        "asia": "Asia",
        "mea": "Middle East & Africa",
        "cis": "CIS",
        "oce": "Oceania",
        "oth": "Other",
    },
    "es": {
        "la": "América Latina",
        "eu": "Europa",
        "asia": "Asia",
        "mea": "Oriente Medio y África",
        "cis": "CEI",
        "oce": "Oceanía",
        "oth": "Otros",
    },
    "pt": {
        "la": "América Latina",
        "eu": "Europa",
        "asia": "Ásia",
        "mea": "Oriente Médio e África",
        "cis": "CEI",
        "oce": "Oceania",
        "oth": "Outros",
    },
}
# Образный блёрб «характер страны в двух мазках» — seed-тон (roadmap_portal_skeleton).
# Пока только ru (34); прочие языки — без вайба (флаг+имя), перевод = отдельный шаг билдера.
VIBE = {
    "ru": {
        "br": "Карнавалы и фавелы. Самба и футбол",
        "vn": "Байки и фо. Джунгли и океан",
        "me": "Горы над Адриатикой. Евро без ЕС",
        "id": "Вулканы и рис. Бали и сёрф",
        "gr": "Острова и руины. Оливки и сиеста",
        "kr": "Кимчи и небоскрёбы. K-pop и корпорации",
        "ph": "Пальмы и острова. Рис и тайфуны",
        "de": "Орднунг и автобаны. Пиво и бумаги",
        "gb": "Туман и пабы. Очереди и вежливость",
        "bg": "Море и горы. Баница и ракия",
        "jp": "Синкансэны и храмы. Вежливость и неон",
        "by": "Драники и зубры. Тишь и порядок",
        "fr": "Багеты и вино. Забастовки и шарм",
        "au": "Кенгуру и сёрф. Простор и солнце",
        "ar": "Танго и стейки. Футбол и инфляция",
        "hu": "Купальни и гуляш. Дунай и Токай",
        "at": "Горы и вальсы. Кофейни и порядок",
        "ru": "Простор и берёзы. Дачи и электрички",
        "cl": "Анды и океан. Вино и пустыня",
        "fi": "Озёра и сауна. Тишина и северное сияние",
        "ge": "Горы и вино. Хачапури и гостеприимство",
        "cz": "Пиво и замки. Прага и мосты",
        "mu": "Пляжи и лагуны. Океан и пальмы",
        "lk": "Чай и слоны. Пляжи и муссоны",
        "be": "Вафли и пиво. Шоколад и еврочиновники",
        "ch": "Горы и часы. Сыр и банки",
        "cn": "Мегаполисы и древность. Чай и скорость",
        "cu": "Ром и сигары. Сальса и ретро-авто",
        "eg": "Пирамиды и Нил. Пустыня и базары",
        "hr": "Адриатика и стены. Острова и солнце",
        "il": "Пустыня и вера. Хайтек и базары",
        "in": "Специи и хаос. Тадж и краски",
        "kz": "Степь и космодром. Простор и бешбармак",
        "tr": "Базары и Босфор. Всё включено",
        "kg": "Горы и юрты. Озёра и кочевники",
    },
}


def home_data(lang, geos, counts):
    """Портал-данные главной: популярные (по числу тем) + регионы + поиск-индекс.
    Единый источник для ru (pages) и прочих языков — форма одинаковая."""
    names = GEO_NAMES.get(lang, {})

    def nm(g):
        return names.get(g, g)

    def tile(g):
        return {"flag": GEO_FLAG.get(g, "•"), "name": nm(g), "url": f"/{lang}/{g}/"}

    gs = sorted(geos, key=nm)
    search_index = [tile(g) for g in gs]
    pop_codes = sorted(geos, key=lambda g: (-counts.get(g, 0), nm(g)))[:12]
    vibe = VIBE.get(lang, {})
    popular = [{**tile(g), "vibe": vibe.get(g, "")} for g in pop_codes]
    rn = REGION_NAMES.get(lang, REGION_NAMES["en"])
    groups = {}
    for g in geos:
        groups.setdefault(CODE2REGION.get(g, "oth"), []).append(g)
    regions = []
    for rk in REGION_ORDER + ["oth"]:
        gl = groups.get(rk)
        if not gl:
            continue
        regions.append(
            {"name": rn.get(rk, rk), "geos": [tile(g) for g in sorted(gl, key=nm)]}
        )
    return popular, regions, search_index


def icon(t):
    tl = t.lower()
    for k, v in ICON.items():
        if k in tl:
            return v
    return "•"


# счётчик подтверждений группы: «✓ N <n_word> из чатов» (префикс/суффикс в i18n)
N_WORD = {
    "en": ("report", "reports"),
    "es": ("reporte", "reportes"),
    "pt": ("relato", "relatos"),
    "de": ("Bericht", "Berichte"),
    "fr": ("retour", "retours"),
    "it": ("segnalazione", "segnalazioni"),
    "tr": ("bildirim", "bildirim"),
}


def ru_w(n, forms):
    """Склонение блёрбов плиток («3 заметки», не «3 заметок» — юзер поймал скрином)."""
    one, few, many = forms
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def blurb(C, key, n):
    """Блёрб плитки с числом: ru — со склонением ({w}), прочие языки — как были."""
    forms = C.get(key + "_w")
    return C[key + "_blurb"].format(n=n, w=ru_w(n, forms) if forms else "")


def n_word(lang, n):
    if lang == "ru":
        if n % 10 == 1 and n % 100 != 11:
            return "сообщение"
        if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            return "сообщения"
        return "сообщений"
    one, many = N_WORD.get(lang, N_WORD["en"])
    return one if n == 1 else many


def lead_split(text):
    """Лид-фраза абзаца → заголовок аккордеона, остальное → тело. Точка ищется
    после 40-го символа (иначе короткий обрывок-лид), точка после цифры не режет
    («шаги: 1. Получите CPF» — не лид). Нет точки — весь текст заголовком,
    тело пустое (рендер покажет только счётчик)."""
    i = text.find(". ", 40)
    while i != -1 and text[i - 1].isdigit():
        i = text.find(". ", i + 1)
    if i == -1:
        return text, ""
    return text[: i + 1], text[i + 2 :]


def groups_to_faqs(v, lang):
    """Дедуп-группы вида (dedup.py) → пункты аккордеона page.html.j2.
    Пункт = репрезентант группы: лид → q, остальное → a, n = подтверждений."""
    by_id = {it["id"]: it for it in v["items"]}
    faqs = []
    for g in v["groups"]:
        rep = by_id[g["rep"]]
        q, a = lead_split(rep["text"])
        f = {"q": q, "a": a, "n": g["n"], "n_word": n_word(lang, g["n"])}
        typ = rep.get(
            "type"
        )  # у хвост-антологий абзац типизирован (lifehack/reglament/…)
        if typ and typ in TYPE_KEY:
            key = TYPE_KEY[typ]
            f["type"] = TYPE_SHORT.get(lang, TYPE_SHORT["en"]).get(key, typ)
            f["type_key"] = key
        faqs.append(f)
    return faqs


def addr(obj, label_field):
    """Слаг узла = ЛАТИНСКИЙ `key`, несомый данными. Один на все языки.

    ⭐ ПРАВИЛО (канон §0.11, слова юзера): адрес = /<язык>/<страна>/ + ОДИНАКОВЫЙ
    английский хвост. `/ru/br/money/` и `/zh/br/money/` — один и тот же хвост.

    ⛔ Почему нельзя слаг от метки (так было с 11.07 по 08.08, коммит d825245):
    метка локализована, значит адрес получался свой в каждом языке. Три следствия,
    все живые: свитчер языка падал в 404 и его увели на хаб страны (ce103c9) вместо
    лечения причины; hreflang по сей день объявляет Google адреса, которых нет
    (проверено: `/ru/ar/bank-i-dengi/` шлёт на `/en/ar/bank-i-dengi/` = 404); а на
    нелатинице (zh ja ko ar hi th) `slug()` вычищает ВСЕ символы и отдаёт "tema" —
    уникализации слагов нигде нет, поэтому страницы молча перезаписывали бы друг
    друга и в гео осталась бы ОДНА вместо двадцати.

    Фолбэк на слаг метки оставлен только как переходный: пока `key` в данных не
    проштампован (`facet_lang.py --stamp-keys`), страница честно объявляет
    `shared_tail=False`, и свитчер с hreflang на неё не рассчитывают.
    """
    return obj.get("key") or slug(obj[label_field])


def pick(pool, seed):
    return pool[int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(pool)]


def cap(s):
    """Заглавная в начале h1 (метки carve бывают строчными: «обмен валюты»)."""
    return s[:1].upper() + s[1:] if s else s


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


_TODAY_ISO = datetime.date.today().isoformat()
UPDATED = datetime.date.today().strftime("%m.%Y")  # подпись в подвале, формат MM.YYYY


def write(name, obj):
    """Записать страницу, проставив дату. ЕДИНСТВЕННОЕ место, где она ставится.

    ⭐ ЗАЧЕМ (2026-08-07). `<lastmod>` в sitemap — подсказка Google «стоит ли перечитывать».
    Раньше дата была РУЧНЫМ аргументом `render.py --all <дата>`: кто-то вписал `2026-07-06`,
    и она месяц ехала во все 2185 адресов. То есть мы говорили «здесь ничего не менялось» —
    при том что 19-20.07 сайт пересобрали целиком. Обход это подавляет.

    ⛔ И НЕ штампуем сегодняшнюю дату всем подряд: если содержимое не изменилось, дата
    остаётся прежней. Свежий `lastmod` на неизменной странице — ложь поисковику, и он от
    таких сигналов быстро отучается им верить. Поэтому сравниваем с тем, что уже лежит,
    игнорируя сами поля даты, и переставляем дату ТОЛЬКО при реальном отличии.
    """
    p = f"{DATA}/{name}"
    keep = None
    try:
        prev = json.load(open(p, encoding="utf-8"))
        strip = {"updated", "updated_iso"}
        if {k: v for k, v in prev.items() if k not in strip} == {
            k: v for k, v in obj.items() if k not in strip
        }:
            keep = prev.get("updated_iso")  # содержимое то же → дату не трогаем
    except Exception:
        pass  # файла нет или битый — считаем изменением, ставим сегодня
    obj["updated"] = UPDATED
    obj["updated_iso"] = keep or _TODAY_ISO
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def build_branches(sv, *, url_pref, file_pref, C, keys, ctx, lang, write_fn):
    """Страница-гигант → хаб с ветками. ОДНА реализация для полок И фактов.

    ⭐ ЗАЧЕМ ОБЩАЯ (2026-08-07). Механизм писался под полки, и `pages.py` читал
    `subshelves` ТОЛЬКО в полочной ветке. Поэтому 561 фактовая страница осталась простынёй
    (максимум 471 пункт на одном адресе) при том что данные для ветвления уже считались.
    ⛔ Копировать эти 60 строк в фактовую ветку НЕЛЬЗЯ: одно правило в двух копиях — та
    самая болезнь, на которой 07.08 поймали четыре промаха подряд (правка не доехала, а
    тест зелёный). Различаются только адрес, имя файла и ключи копирайта — они и параметры.

    Возвращает (subtiles, rest_groups): плитки веток для хаба и группы, не попавшие ни в
    одну ветку (их хаб покажет аккордеоном внизу — иначе пункты пропадут молча).
    """
    geo, name, namep = ctx["geo"], ctx["name"], ctx["namep"]
    by_rep = {g["rep"]: g for g in sv["groups"]}
    sub_sibs = [
        {"name": sub["name"], "slug": addr(sub, "name")} for sub in sv["subshelves"]
    ]
    subtiles = []
    for sub in sv["subshelves"]:
        ss = addr(sub, "name")
        sub_groups = [by_rep[r] for r in sub["reps"] if r in by_rep]
        sub_view = {"items": sv["items"], "groups": sub_groups}
        tl_ = ctx["tl"]
        spage = {
            "lang": lang,
            "template": "page.html.j2",
            "path": f"{url_pref}{ss}/",
            "shared_tail": bool(sub.get("key")),
            "geo": geo,
            "geo_name": name,
            "intent_name": sub["name"],
            "title": C[keys["title"]].format(name=name, tl=tl_(sub["name"])),
            "meta_desc": C[keys["desc"]].format(
                name=name, namep=namep, tl=tl_(sub["name"])
            ),
            "h1": pick(C["FHEAD"], geo + file_pref + ss).format(
                t=cap(sub["name"]), g=name, gp=namep
            ),
            "intro": C[keys["intro"]].format(
                name=name, namep=namep, tl=tl_(sub["name"])
            ),
            "list_label": C[keys["list_label"]],
            "faqs": groups_to_faqs(sub_view, lang),
            "chips": [
                {
                    "icon": icon(x["name"]),
                    "label": x["name"],
                    "url": f"{url_pref}{x['slug']}/",
                    "soon": False,
                }
                for x in sub_sibs
                if x["slug"] != ss
            ][:6],
        }
        write_fn(f"{file_pref}{ss}.json", spage)
        subtiles.append(
            {
                "icon": icon(sub["name"]),
                "title": sub["name"],
                "blurb": blurb(C, keys["blurb"], len(sub_groups)),
                "url": f"{url_pref}{ss}/",
            }
        )
    covered = {r for sub in sv["subshelves"] for r in sub["reps"]}
    return subtiles, [g for g in sv["groups"] if g["rep"] not in covered]


def chips_for(cur_slug, siblings):
    return [
        {"icon": icon(s["tema"]), "label": s["tema"], "url": s["url"], "soon": False}
        for s in siblings
        if s["slug"] != cur_slug
    ][:6]


def _facet_dir(lang):
    return f"{BUILT}/out_facet" if lang == "ru" else f"{BUILT}/out_facet_{lang}"


def _ques_dir(lang):
    return f"{BUILT}/out_questions" if lang == "ru" else f"{BUILT}/out_questions_{lang}"


def build_geo(geo, lang="ru"):
    C = COPY[lang]
    name = GEO_NAMES.get(lang, {}).get(geo, geo)
    # «где» для ru-строк («{tl} в Бразилии»); прочие языки — имя как есть
    namep = GEO_LOC.get(geo, f"в {name}") if lang == "ru" else name
    facts = load(f"{_facet_dir(lang)}/{geo}.json")
    ques = load(f"{_ques_dir(lang)}/{geo}.json")
    n = 0

    def tl(t):
        return t.lower() if C["lower"] else t

    # --- ФАКТ-ТЕМЫ (советы-список, ≥4 факта = страница) ---
    fact_tiles, fact_sibs = [], []
    fviews = sorted(
        (facts or {}).get("views_by_task", []), key=lambda v: -len(v["items"])
    )
    fviews = [v for v in fviews if len(v["items"]) >= 4]
    if (
        lang != "ru"
    ):  # страховка: непереведённая (кириллическая) метка → не плодим кириллический URL
        fviews = [v for v in fviews if not re.search("[а-яёА-ЯЁ]", v["zadacha"])]
    for v in fviews:
        s = addr(v, "zadacha")
        fact_sibs.append(
            {"tema": v["zadacha"], "slug": s, "url": f"/{lang}/{geo}/{s}/"}
        )
    for v in fviews:
        tema = v["zadacha"]
        s = addr(v, "zadacha")
        items = [it["text"] for it in v["items"]]
        page = {
            "lang": lang,
            "path": f"/{lang}/{geo}/{s}/",
            # хвост адреса общий для всех языков? от этого зависят свитчер и hreflang
            "shared_tail": bool(v.get("key")),
            "geo": geo,
            "geo_name": name,
            "intent_name": tema,
            "title": C["fact_title"].format(name=name, tl=tl(tema)),
            "meta_desc": C["fact_desc"].format(name=name, namep=namep, tl=tl(tema)),
            "h1": pick(C["FHEAD"], geo + s).format(t=cap(tema), g=name, gp=namep),
            "intro": C["fact_intro"].format(name=name, namep=namep, tl=tl(tema)),
            "chips": chips_for(s, fact_sibs),
        }
        if v.get("subshelves"):
            # ⭐ ТЕМА-ГИГАНТ ВЕТВИТСЯ так же, как полка (2026-08-07). До этого `subshelves`
            # читались ТОЛЬКО в полочной ветке, поэтому 561 фактовая страница оставалась
            # простынёй (максимум 471 пункт) — при том что данные для ветвления считались.
            subtiles, rest = build_branches(
                v,
                url_pref=f"/{lang}/{geo}/{s}/",
                file_pref=f"{lang}_{geo}_{s}_",
                C=C,
                keys={
                    "title": "fact_title",
                    "desc": "fact_desc",
                    "intro": "fact_intro",
                    "list_label": "fact_list_label",
                    "blurb": "fact",
                },
                ctx={"geo": geo, "name": name, "namep": namep, "tl": tl},
                lang=lang,
                write_fn=write,
            )
            n += len(subtiles)
            page["template"] = "index.html.j2"
            page["list_label"] = C["list_label_topics"]
            page["tiles"] = subtiles
            if rest:  # пункты вне ветвей — аккордеоном внизу, иначе пропали бы молча
                page["faqs"] = groups_to_faqs(
                    {"items": v["items"], "groups": rest}, lang
                )
                page["faqs_label"] = C["fact_list_label"]
        elif v.get("groups"):  # дедуп прошёл (dedup.py) → компактная страница-аккордеон
            page["template"] = "page.html.j2"
            page["short_answer"] = v.get("kratko")  # None → блок скрыт шаблоном
            page["list_label"] = C[
                "fact_list_label"
            ]  # «Из живого опыта», не «Частые вопросы»
            page["faqs"] = groups_to_faqs(v, lang)
        else:  # без дедупа (гео/язык ещё не прогнан) → старый список
            page["template"] = "qlist.html.j2"
            page["list_label"] = C["fact_list_label"]
            page["questions"] = items
        write(f"{lang}_{geo}_{s}.json", page)
        n += 1
        fact_tiles.append(
            {
                "icon": icon(tema),
                "title": tema,
                "blurb": blurb(C, "fact", len(items)),
                "url": f"/{lang}/{geo}/{s}/",
            }
        )

    # --- ВОПРОС-КОНТУР (хаб + темы под /<lang>/<geo>/q/) ---
    q_ok = False
    qgroups = [g for g in (ques or {}).get("groups", []) if len(g["questions"]) >= 4]
    if qgroups:
        q_ok = True
        q_sibs = [
            {
                "tema": g["tema"],
                "slug": addr(g, "tema"),
                "url": f"/{lang}/{geo}/q/{addr(g, 'tema')}/",
            }
            for g in qgroups
        ]
        for g in qgroups:
            s = addr(g, "tema")
            page = {
                "lang": lang,
                "template": "qlist.html.j2",
                "path": f"/{lang}/{geo}/q/{s}/",
                "shared_tail": bool(g.get("key")),
                "geo": geo,
                "geo_name": name,
                "intent_name": g["tema"],
                "title": C["q_title"].format(name=name, tl=tl(g["tema"])),
                "meta_desc": C["q_desc"].format(name=name, tl=tl(g["tema"])),
                "h1": pick(C["QHEAD"], geo + s + "q").format(t=g["tema"]),
                "intro": C["q_intro"],
                "list_label": C["q_list_label"],
                "questions": g["questions"],
                "chips": [
                    {
                        "icon": icon(x["tema"]),
                        "label": x["tema"],
                        "url": x["url"],
                        "soon": False,
                    }
                    for x in q_sibs
                    if x["slug"] != s
                ][:6],
            }
            write(f"{lang}_{geo}_q_{s}.json", page)
            n += 1
        qtiles = [
            {
                "icon": icon(g["tema"]),
                "title": g["tema"],
                "blurb": blurb(C, "q", len(g["questions"])),
                "url": f"/{lang}/{geo}/q/{addr(g, 'tema')}/",
            }
            for g in qgroups
        ]
        write(
            f"{lang}_{geo}_q_hub.json",
            {
                "lang": lang,
                "template": "index.html.j2",
                "path": f"/{lang}/{geo}/q/",
                "geo": geo,
                "geo_name": name,
                "title": C["qhub_title"].format(name=name),
                "meta_desc": C["qhub_desc"].format(name=name),
                "h1": C["qhub_h1"],
                "intro": C["qhub_intro"],
                "list_label": C["list_label_topics"],
                "tiles": qtiles,
            },
        )
        n += 1

    # --- ШЕЛФ-КОНТУР (антологии хвоста: полки под /<lang>/<geo>/s/) ---
    # Хвост-курирование: синглы, что раньше терялись фильтром ≥4, живут на широких
    # полках-антологиях. Через ту же укладку, что факты: дедуп-группы → аккордеон
    # page.html.j2 + тег типа (lifehack/reglament/…).
    #
    # ⛔ Здесь стояло `if lang == "ru"`. Условие было честным 11.07: имена полок брались
    # из русской таксономии, а чип типа был русским словом. Обе причины сняты — 27.07
    # (84b0a19) перевод понёс локализованное имя и латинский `key`, а чипы стали
    # языковыми, — но САМО условие осталось и прожило причину на 12 дней. Тот же
    # коммит 84b0a19 правил ЭТУ функцию: добавил `_sk()` с докстрингом про «адрес полки
    # обязан совпадать во всех языках» — внутрь блока, который для не-ru не исполнялся.
    # Цена: ~1100 запросов, потраченных на перевод текстов хвоста в 12 языков, не дали
    # ни одной страницы; 375 полок × 3 сборных языка = 1125 страниц не выкладывались.
    #
    # Поэтому теперь условие привязано к ПРИЧИНЕ, а не к языку: контур строится там, где
    # у языка есть полочный копирайт. Добавили язык без него — полки просто не выйдут,
    # вместо KeyError; появился копирайт — контур включается сам, без правки этой строки.
    s_ok = False
    shelves = []
    if "shelf_title" in C:
        shelves = sorted(
            [
                sv
                for sv in (facts or {}).get("shelves", [])
                if len(sv["items"]) >= SHELF_MIN
            ],
            key=lambda x: -len(x["items"]),
        )
    if shelves:
        s_ok = True

        def _sk(sv):
            """Ключ полки для URL. Приоритет — ключ, НЕСОМЫЙ файлом (`key`): в переводах
            имя полки локализовано, а `SHELF_KEY` смотрит по РУССКОМУ имени и от перевода
            промахнётся, слепив слаг из локализованного текста — разный в каждом языке, а
            на нелатинских и вовсе нечитаемый. Адрес полки обязан совпадать во всех
            языках, иначе hreflang связывает не те страницы."""
            return sv.get("key") or SHELF_KEY.get(sv["shelf"]) or slug(sv["shelf"])

        sh_sibs = [
            {
                "tema": sv["shelf"],
                "slug": _sk(sv),
                "url": f"/{lang}/{geo}/s/{_sk(sv)}/",
            }
            for sv in shelves
        ]
        for sv in shelves:
            sk = _sk(sv)
            page = {
                "lang": lang,
                "path": f"/{lang}/{geo}/s/{sk}/",
                "geo": geo,
                "geo_name": name,
                "intent_name": sv["shelf"],
                "title": C["shelf_title"].format(name=name, tl=tl(sv["shelf"])),
                "meta_desc": C["shelf_desc"].format(
                    name=name, namep=namep, tl=tl(sv["shelf"])
                ),
                "h1": pick(C["FHEAD"], geo + sk).format(
                    t=cap(sv["shelf"]), g=name, gp=namep
                ),
                "intro": C["shelf_intro"].format(
                    name=name, namep=namep, tl=tl(sv["shelf"])
                ),
                "chips": [
                    {
                        "icon": icon(x["tema"]),
                        "label": x["tema"],
                        "url": x["url"],
                        "soon": False,
                    }
                    for x in sh_sibs
                    if x["slug"] != sk
                ][:6],
            }
            if sv.get("subshelves"):  # полка-гигант ВЕТВИТСЯ: хаб + под-страницы
                n_before = n
                subtiles, rest = build_branches(
                    sv,
                    url_pref=f"/{lang}/{geo}/s/{sk}/",
                    file_pref=f"{lang}_{geo}_s_{sk}_",
                    C=C,
                    keys={
                        "title": "shelf_title",
                        "desc": "shelf_desc",
                        "intro": "shelf_intro",
                        "list_label": "shelf_list_label",
                        "blurb": "shelf",
                    },
                    ctx={"geo": geo, "name": name, "namep": namep, "tl": tl},
                    lang=lang,
                    write_fn=write,
                )
                n = n_before + len(subtiles)
                # хаб полки: плитки веток + остаток (репы вне веток) аккордеоном внизу
                page["template"] = "index.html.j2"
                page["list_label"] = C["list_label_topics"]
                page["tiles"] = subtiles
                if rest:
                    # ⛔ БЫЛО `rest[:30]` — пункты за тридцатым исчезали БЕЗ СЛЕДА и без
                    # строчки в логе (2026-08-07, юзер: «звучит как нездоровая хрень»).
                    # Замер на момент снятия: у 101 разветвлённой полки остаток пуст
                    # полностью, то есть кэп не срабатывал ни разу — это была мина, а не
                    # рабочее ограничение. Молчаливое усечение запрещено: если ветвление
                    # однажды покроет не всё, хаб станет ВИДИМОЙ простынёй, и её поймает
                    # проверка правил. Видимая простыня лучше невидимой потери.
                    # Теперь поведение совпадает с фактовой ветвью — одно правило, не два.
                    page["faqs"] = groups_to_faqs(
                        {"items": sv["items"], "groups": rest}, lang
                    )
                    page["faqs_label"] = C["shelf_list_label"]
            elif sv.get("groups"):  # укладка как у фактов: аккордеон + счётчики + типы
                page["template"] = "page.html.j2"
                page["list_label"] = C["shelf_list_label"]
                page["faqs"] = groups_to_faqs(sv, lang)
            else:  # полка без дедупа → старый список (не должно случаться после dedup.py)
                page["template"] = "qlist.html.j2"
                page["list_label"] = C["shelf_list_label"]
                page["questions"] = [it["text"] for it in sv["items"]]
            write(f"{lang}_{geo}_s_{sk}.json", page)
            n += 1
        stiles = [
            {
                "icon": icon(sv["shelf"]),
                "title": sv["shelf"],
                "blurb": blurb(C, "shelf", len(sv["items"])),
                "url": f"/{lang}/{geo}/s/{_sk(sv)}/",
            }
            for sv in shelves
        ]
        write(
            f"{lang}_{geo}_s_hub.json",
            {
                "lang": lang,
                "template": "index.html.j2",
                "path": f"/{lang}/{geo}/s/",
                "geo": geo,
                "geo_name": name,
                "title": C["shub_title"].format(name=name),
                "meta_desc": C["shub_desc"].format(name=name),
                "h1": C["shub_h1"],
                "intro": C["shub_intro"],
                "list_label": C["list_label_topics"],
                "tiles": stiles,
            },
        )
        n += 1

    # --- ГЕО-ХАБ (тайлы фактов + мостики вопросов и разделов) ---
    tiles = list(fact_tiles)
    if s_ok:
        tiles.insert(
            0,
            {
                "icon": "📚",
                "title": C["bridge_shelf_title"],
                "blurb": C["bridge_shelf_blurb"],
                "url": f"/{lang}/{geo}/s/",
            },
        )
    if q_ok:
        tiles.insert(
            0,
            {
                "icon": "❓",
                "title": C["bridge_title"],
                "blurb": C["bridge_blurb"],
                "url": f"/{lang}/{geo}/q/",
            },
        )
    write(
        f"{lang}_{geo}_hub.json",
        {
            "lang": lang,
            "template": "index.html.j2",
            "path": f"/{lang}/{geo}/",
            "shared_tail": True,  # хаб страны: хвост = код гео, одинаков везде
            "geo": geo,
            "geo_name": name,
            "title": C["hub_title"].format(name=name),
            "meta_desc": C["hub_desc"].format(name=name),
            "h1": name,
            "intro": C["hub_intro"],
            "list_label": C["list_label_topics"],
            "tiles": tiles,
        },
    )
    n += 1
    return (
        n,
        len(fact_tiles),
        len(qgroups) if q_ok else 0,
        len(shelves) if s_ok else 0,
    )


def langs_for(geo):
    """Языки, у которых есть built-данные фактов для гео."""
    out = []
    for lang in COPY:
        if os.path.exists(f"{_facet_dir(lang)}/{geo}.json"):
            out.append(lang)
    return out


def build_home(lang, geos, counts=None):
    """Главная /<lang>/ — портал-вход: поиск + популярные (образный блёрб) + регионы.
    counts: {geo: число тем} для ранжирования «популярных» (из build_geo)."""
    HA = HOME_ABOUT[lang]
    popular, regions, search_index = home_data(lang, geos, counts or {})
    write(
        f"{lang}_home.json",
        {
            "lang": lang,
            "template": "home.html.j2",
            "path": f"/{lang}/",
            "shared_tail": True,  # адрес главной языко-независим по построению
            "crumb_label": None,
            "title": HA["home_title"],
            "meta_desc": HA["home_desc"],
            "h1": HA["home_h1"],
            "intro": HA["home_intro"],
            "popular": popular,
            "regions": regions,
            "search_index": search_index,
        },
    )


def build_about(lang):
    HA = HOME_ABOUT[lang]
    write(
        f"{lang}_about.json",
        {
            "lang": lang,
            "template": "index.html.j2",
            "path": f"/{lang}/about/",
            "shared_tail": True,
            "crumb_label": HA["about_crumb"],
            "title": HA["about_title"],
            "meta_desc": HA["about_desc"],
            "h1": HA["about_h1"],
            "body": HA["about_body"],
        },
    )


if __name__ == "__main__":
    geos = sys.argv[1:]
    if not geos or geos == ["--all"]:
        geos = sorted(
            {os.path.basename(f)[:-5] for f in glob.glob(f"{BUILT}/out_facet/*.json")}
            | {
                os.path.basename(f)[:-5]
                for f in glob.glob(f"{BUILT}/out_facet_*/*.json")
            }
        )
    total = 0
    built = {}  # lang -> [geos]
    counts = {}  # lang -> {geo: число факт-тем} (ранжир «популярных» на home)
    for g in geos:
        for lang in langs_for(g):
            n, nf, nq, ns = build_geo(g, lang)
            total += n
            built.setdefault(lang, []).append(g)
            counts.setdefault(lang, {})[g] = nf
            print(
                f"{g}/{lang}: страниц-data {n} (факт-тем {nf}, вопрос-тем {nq}, полок {ns})"
            )
    # home — портал-вход для ВСЕХ языков (включая ru: pages владеет home, wire делегирует сюда).
    # about — только не-ru (ru_about живой, не трогаем).
    for lang, gl in built.items():
        build_home(lang, sorted(gl), counts.get(lang, {}))
        if lang != "ru":
            build_about(lang)
        print(f"{lang}: home{'' if lang == 'ru' else ' + about'} ({len(gl)} стран)")
    print(f"ИТОГО data-страниц: {total} (дальше render.py --all)")
