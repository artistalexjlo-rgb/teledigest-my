"""pages.py — из BUILT-данных гео (out_facet[_<lang>]/<geo>.json + out_questions[_<lang>]/<geo>.json)
собирает portal-схему data/ (октагон-шаблон): гео-хаб + факт-тема-страницы + вопрос-хаб/темы.
Оба контура, единообразно. Дизайн гарантирован шаблоном (page/qlist/index.html.j2).

Мультиязык: lang="ru" читает out_facet/, любой другой — out_facet_<lang>/ (структура ×1, текст из
оригинала). Копия страниц — из COPY[lang]. render.py сам подхватит i18n/<lang>.json по page.lang.

Запуск: python pages.py <geo> [<geo2> ...]   (или --all по out_facet/*.json; строит ВСЕ языки, у кого
есть built-данные). Дальше — render.py --all + валидация (readycheck).
"""

import glob
import hashlib
import json
import os
import re
import sys

import tail_taxonomy as _tax

# полка → стабильный латинский ключ для URL (/ru/<geo>/s/finance/), не транслит-slug
SHELF_KEY = {name: key for key, name, _ in _tax.SHELVES}
# тип абзаца → латинский ключ (css-класс тега на карточке/аккордеоне)
TYPE_KEY = {name: key for key, name, _ in _tax.TYPES}
# короткий ярлык тега (полное имя типа громоздко для чипа в аккордеоне)
TYPE_SHORT = {
    "lifehack": "лайфхак",
    "reglament": "регламент",
    "howto": "инструкция",
    "risk": "риск",
    "case": "кейс",
    "service": "сервис",
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
        "fact_blurb": "{n} tips from chats",
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
        "fact_blurb": "{n} consejos de los chats",
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
        "fact_blurb": "{n} dicas dos chats",
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
            f["type"] = TYPE_SHORT.get(key, typ)
            f["type_key"] = key
        faqs.append(f)
    return faqs


# RU→latin транслит для слагов: URL латиницей (money-качество), не кириллический %-суп
_TR = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def slug(t):
    t = "".join(_TR.get(ch, ch) for ch in t.lower())
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40] or "tema"


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


def write(name, obj):
    json.dump(
        obj, open(f"{DATA}/{name}", "w", encoding="utf-8"), ensure_ascii=False, indent=1
    )


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
        s = slug(v["zadacha"])
        fact_sibs.append(
            {"tema": v["zadacha"], "slug": s, "url": f"/{lang}/{geo}/{s}/"}
        )
    for v in fviews:
        tema = v["zadacha"]
        s = slug(tema)
        items = [it["text"] for it in v["items"]]
        page = {
            "lang": lang,
            "path": f"/{lang}/{geo}/{s}/",
            "geo": geo,
            "geo_name": name,
            "intent_name": tema,
            "updated": "07.2026",
            "title": C["fact_title"].format(name=name, tl=tl(tema)),
            "meta_desc": C["fact_desc"].format(name=name, namep=namep, tl=tl(tema)),
            "h1": pick(C["FHEAD"], geo + s).format(t=cap(tema), g=name, gp=namep),
            "intro": C["fact_intro"].format(name=name, namep=namep, tl=tl(tema)),
            "chips": chips_for(s, fact_sibs),
        }
        if v.get("groups"):  # дедуп прошёл (dedup.py) → компактная страница-аккордеон
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
                "slug": slug(g["tema"]),
                "url": f"/{lang}/{geo}/q/{slug(g['tema'])}/",
            }
            for g in qgroups
        ]
        for g in qgroups:
            s = slug(g["tema"])
            page = {
                "lang": lang,
                "template": "qlist.html.j2",
                "path": f"/{lang}/{geo}/q/{s}/",
                "geo": geo,
                "geo_name": name,
                "intent_name": g["tema"],
                "updated": "07.2026",
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
                "url": f"/{lang}/{geo}/q/{slug(g['tema'])}/",
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
                "updated": "07.2026",
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
    # page.html.j2 + тег типа (lifehack/reglament/…). Имена полок = русская
    # таксономия → пока только lang=="ru" (i18n имён — отдельный шаг).
    s_ok = False
    shelves = []
    if lang == "ru":
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
        sh_sibs = [
            {
                "tema": sv["shelf"],
                "slug": SHELF_KEY.get(sv["shelf"], slug(sv["shelf"])),
                "url": f"/{lang}/{geo}/s/{SHELF_KEY.get(sv['shelf'], slug(sv['shelf']))}/",
            }
            for sv in shelves
        ]
        for sv in shelves:
            sk = SHELF_KEY.get(sv["shelf"], slug(sv["shelf"]))
            page = {
                "lang": lang,
                "path": f"/{lang}/{geo}/s/{sk}/",
                "geo": geo,
                "geo_name": name,
                "intent_name": sv["shelf"],
                "updated": "07.2026",
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
                by_rep = {g["rep"]: g for g in sv["groups"]}
                sub_sibs = [
                    {"name": sub["name"], "slug": slug(sub["name"])}
                    for sub in sv["subshelves"]
                ]
                subtiles = []
                for sub in sv["subshelves"]:
                    ss = slug(sub["name"])
                    sub_groups = [by_rep[r] for r in sub["reps"] if r in by_rep]
                    sub_view = {
                        "items": sv["items"],
                        "groups": sub_groups,
                    }  # для groups_to_faqs
                    spage = {
                        "lang": lang,
                        "template": "page.html.j2",
                        "path": f"/{lang}/{geo}/s/{sk}/{ss}/",
                        "geo": geo,
                        "geo_name": name,
                        "intent_name": sub["name"],
                        "updated": "07.2026",
                        "title": C["shelf_title"].format(name=name, tl=tl(sub["name"])),
                        "meta_desc": C["shelf_desc"].format(
                            name=name, namep=namep, tl=tl(sub["name"])
                        ),
                        "h1": pick(C["FHEAD"], geo + sk + ss).format(
                            t=cap(sub["name"]), g=name, gp=namep
                        ),
                        "intro": C["shelf_intro"].format(
                            name=name, namep=namep, tl=tl(sub["name"])
                        ),
                        "list_label": C["shelf_list_label"],
                        "faqs": groups_to_faqs(sub_view, lang),
                        "chips": [
                            {
                                "icon": icon(x["name"]),
                                "label": x["name"],
                                "url": f"/{lang}/{geo}/s/{sk}/{x['slug']}/",
                                "soon": False,
                            }
                            for x in sub_sibs
                            if x["slug"] != ss
                        ][:6],
                    }
                    write(f"{lang}_{geo}_s_{sk}_{ss}.json", spage)
                    n += 1
                    subtiles.append(
                        {
                            "icon": icon(sub["name"]),
                            "title": sub["name"],
                            "blurb": C["shelf_blurb"].format(n=len(sub_groups)),
                            "url": f"/{lang}/{geo}/s/{sk}/{ss}/",
                        }
                    )
                # хаб полки: плитки веток + остаток (репы вне веток) аккордеоном внизу
                covered = {r for sub in sv["subshelves"] for r in sub["reps"]}
                rest = [g for g in sv["groups"] if g["rep"] not in covered]
                page["template"] = "index.html.j2"
                page["list_label"] = C["list_label_topics"]
                page["tiles"] = subtiles
                if rest:
                    page["faqs"] = groups_to_faqs(
                        {"items": sv["items"], "groups": rest[:30]}, lang
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
                "url": f"/{lang}/{geo}/s/{SHELF_KEY.get(sv['shelf'], slug(sv['shelf']))}/",
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
                "updated": "07.2026",
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
            "geo": geo,
            "geo_name": name,
            "updated": "07.2026",
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
            "updated": "07.2026",
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
            "updated": "07.2026",
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
