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
from slugs import slug, slug_or_none  # ЕДИНСТВЕННОЕ определение хвоста адреса

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
    "zh": {
        "lifehack": "窍门",
        "reglament": "规定",
        "howto": "步骤",
        "risk": "风险",
        "case": "案例",
        "service": "服务",
    },
    "ja": {
        "lifehack": "コツ",
        "reglament": "ルール",
        "howto": "手順",
        "risk": "リスク",
        "case": "事例",
        "service": "サービス",
    },
    "ko": {
        "lifehack": "팁",
        "reglament": "규정",
        "howto": "방법",
        "risk": "위험",
        "case": "사례",
        "service": "서비스",
    },
    "ar": {
        "lifehack": "نصيحة",
        "reglament": "قاعدة",
        "howto": "دليل",
        "risk": "خطر",
        "case": "حالة",
        "service": "خدمة",
    },
    "hi": {
        "lifehack": "तरीका",
        "reglament": "नियम",
        "howto": "गाइड",
        "risk": "जोखिम",
        "case": "मामला",
        "service": "सेवा",
    },
    "th": {
        "lifehack": "เคล็ดลับ",
        "reglament": "กฎ",
        "howto": "วิธีทำ",
        "risk": "ความเสี่ยง",
        "case": "กรณี",
        "service": "บริการ",
    },
}
SHELF_MIN = 3  # полка становится страницей от 3 абзацев (мельче — тонковато)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pseo
DATA = f"{BASE}/data"
# built-данные лежат либо локально (pull с VPS), либо укажи путь
BUILT = os.environ.get("BUILT_DIR", f"{BASE}/builder")

# ⭐ ИМЕНА И ФЛАГИ СТРАН — ИЗ СПРАВОЧНИКА, а не из своей таблички (2026-08-11).
# Справочник знает 249 стран с русскими именами и флагами плюс 124 английских. А 11.07 я
# завёл рядом свои 35 имён и 35 флагов, не посмотрев, что он есть, — и 214 стран остались
# безымянными: сайт печатал сырые коды (`ae`, `al`, `bo`), 55 позиций валились в «Другие»,
# часть чипов вела в 404. `CLAUDE.md` это прямо запрещал: «Страны: ISO коды из
# country_codes.py, хардкод больше не нужен».
#
# Файл лежит КОПИЕЙ здесь, а не читается из `src/teledigest/`: `pseo` — самодостаточное
# дерево (свои дубли ртов, свои импорты), и бегать за данными в пакет бота ему незачем.
# Копия ПОЛНАЯ и дословная; за расхождением следит `test_country_codes_copy.py`, а не моя
# память. Замер, оправдывающий копию: справочник правили 5 раз за 4 месяца, все правки в мае.
from country_codes import COUNTRIES as REF  # noqa: E402
from country_codes import COUNTRY_NAMES_EN as REF_EN  # noqa: E402

# ⚠️ GEO_NAMES остаётся, но как СЛОЙ ПЕРЕОПРЕДЕЛЕНИЯ, а не источник: в нём лежат имена на
# все 14 языков для 35 стран, а справочник знает только русский и английский. Плюс два
# осознанных расхождения по-русски: «Южная Корея» вместо «Республика Корея» и «Киргизия»
# вместо «Кыргызстан» — так говорит аудитория.
GEO_NAMES = {
    "ru": {
        "any": "Везде",
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
        "any": "Anywhere",
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
        "any": "Cualquier país",
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
        "any": "Qualquer país",
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
        "any": "Überall",
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
        "any": "Partout",
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
        "any": "Ovunque",
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
        "any": "Her yerde",
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
    "zh": {
        "any": "任何地方",
        "br": "巴西",
        "vn": "越南",
        "me": "黑山",
        "id": "印度尼西亚",
        "gr": "希腊",
        "kr": "韩国",
        "ph": "菲律宾",
        "de": "德国",
        "gb": "英国",
        "bg": "保加利亚",
        "jp": "日本",
        "by": "白俄罗斯",
        "fr": "法国",
        "au": "澳大利亚",
        "ar": "阿根廷",
        "hu": "匈牙利",
        "at": "奥地利",
        "ru": "俄罗斯",
        "cl": "智利",
        "fi": "芬兰",
        "ge": "格鲁吉亚",
        "cz": "捷克",
        "mu": "毛里求斯",
        "lk": "斯里兰卡",
        "be": "比利时",
        "ch": "瑞士",
        "cn": "中国",
        "cu": "古巴",
        "eg": "埃及",
        "hr": "克罗地亚",
        "il": "以色列",
        "in": "印度",
        "kz": "哈萨克斯坦",
        "tr": "土耳其",
        "kg": "吉尔吉斯斯坦",
    },
    "ja": {
        "any": "どこでも",
        "br": "ブラジル",
        "vn": "ベトナム",
        "me": "モンテネグロ",
        "id": "インドネシア",
        "gr": "ギリシャ",
        "kr": "韓国",
        "ph": "フィリピン",
        "de": "ドイツ",
        "gb": "イギリス",
        "bg": "ブルガリア",
        "jp": "日本",
        "by": "ベラルーシ",
        "fr": "フランス",
        "au": "オーストラリア",
        "ar": "アルゼンチン",
        "hu": "ハンガリー",
        "at": "オーストリア",
        "ru": "ロシア",
        "cl": "チリ",
        "fi": "フィンランド",
        "ge": "ジョージア",
        "cz": "チェコ",
        "mu": "モーリシャス",
        "lk": "スリランカ",
        "be": "ベルギー",
        "ch": "スイス",
        "cn": "中国",
        "cu": "キューバ",
        "eg": "エジプト",
        "hr": "クロアチア",
        "il": "イスラエル",
        "in": "インド",
        "kz": "カザフスタン",
        "tr": "トルコ",
        "kg": "キルギス",
    },
    "ko": {
        "any": "어디서나",
        "br": "브라질",
        "vn": "베트남",
        "me": "몬테네그로",
        "id": "인도네시아",
        "gr": "그리스",
        "kr": "대한민국",
        "ph": "필리핀",
        "de": "독일",
        "gb": "영국",
        "bg": "불가리아",
        "jp": "일본",
        "by": "벨라루스",
        "fr": "프랑스",
        "au": "호주",
        "ar": "아르헨티나",
        "hu": "헝가리",
        "at": "오스트리아",
        "ru": "러시아",
        "cl": "칠레",
        "fi": "핀란드",
        "ge": "조지아",
        "cz": "체코",
        "mu": "모리셔스",
        "lk": "스리랑카",
        "be": "벨기에",
        "ch": "스위스",
        "cn": "중국",
        "cu": "쿠바",
        "eg": "이집트",
        "hr": "크로아티아",
        "il": "이스라엘",
        "in": "인도",
        "kz": "카자흐스탄",
        "tr": "튀르키예",
        "kg": "키르기스스탄",
    },
    "ar": {
        "any": "في أي مكان",
        "br": "البرازيل",
        "vn": "فيتنام",
        "me": "الجبل الأسود",
        "id": "إندونيسيا",
        "gr": "اليونان",
        "kr": "كوريا الجنوبية",
        "ph": "الفلبين",
        "de": "ألمانيا",
        "gb": "المملكة المتحدة",
        "bg": "بلغاريا",
        "jp": "اليابان",
        "by": "بيلاروسيا",
        "fr": "فرنسا",
        "au": "أستراليا",
        "ar": "الأرجنتين",
        "hu": "هنغاريا",
        "at": "النمسا",
        "ru": "روسيا",
        "cl": "تشيلي",
        "fi": "فنلندا",
        "ge": "جورجيا",
        "cz": "التشيك",
        "mu": "موريشيوس",
        "lk": "سريلانكا",
        "be": "بلجيكا",
        "ch": "سويسرا",
        "cn": "الصين",
        "cu": "كوبا",
        "eg": "مصر",
        "hr": "كرواتيا",
        "il": "إسرائيل",
        "in": "الهند",
        "kz": "كازاخستان",
        "tr": "تركيا",
        "kg": "قيرغيزستان",
    },
    "hi": {
        "any": "कहीं भी",
        "br": "ब्राज़ील",
        "vn": "वियतनाम",
        "me": "मोंटेनेग्रो",
        "id": "इंडोनेशिया",
        "gr": "ग्रीस",
        "kr": "दक्षिण कोरिया",
        "ph": "फ़िलीपींस",
        "de": "जर्मनी",
        "gb": "यूनाइटेड किंगडम",
        "bg": "बुल्गारिया",
        "jp": "जापान",
        "by": "बेलारूस",
        "fr": "फ़्रांस",
        "au": "ऑस्ट्रेलिया",
        "ar": "अर्जेंटीना",
        "hu": "हंगरी",
        "at": "ऑस्ट्रिया",
        "ru": "रूस",
        "cl": "चिली",
        "fi": "फ़िनलैंड",
        "ge": "जॉर्जिया",
        "cz": "चेकिया",
        "mu": "मॉरिशस",
        "lk": "श्रीलंका",
        "be": "बेल्जियम",
        "ch": "स्विट्ज़रलैंड",
        "cn": "चीन",
        "cu": "क्यूबा",
        "eg": "मिस्र",
        "hr": "क्रोएशिया",
        "il": "इज़राइल",
        "in": "भारत",
        "kz": "कज़ाख़िस्तान",
        "tr": "तुर्किये",
        "kg": "किर्गिज़स्तान",
    },
    "th": {
        "any": "ทุกที่",
        "br": "บราซิล",
        "vn": "เวียดนาม",
        "me": "มอนเตเนโกร",
        "id": "อินโดนีเซีย",
        "gr": "กรีซ",
        "kr": "เกาหลีใต้",
        "ph": "ฟิลิปปินส์",
        "de": "เยอรมนี",
        "gb": "สหราชอาณาจักร",
        "bg": "บัลแกเรีย",
        "jp": "ญี่ปุ่น",
        "by": "เบลารุส",
        "fr": "ฝรั่งเศส",
        "au": "ออสเตรเลีย",
        "ar": "อาร์เจนตินา",
        "hu": "ฮังการี",
        "at": "ออสเตรีย",
        "ru": "รัสเซีย",
        "cl": "ชิลี",
        "fi": "ฟินแลนด์",
        "ge": "จอร์เจีย",
        "cz": "เช็กเกีย",
        "mu": "มอริเชียส",
        "lk": "ศรีลังกา",
        "be": "เบลเยียม",
        "ch": "สวิตเซอร์แลนด์",
        "cn": "จีน",
        "cu": "คิวบา",
        "eg": "อียิปต์",
        "hr": "โครเอเชีย",
        "il": "อิสราเอล",
        "in": "อินเดีย",
        "kz": "คาซัคสถาน",
        "tr": "ตุรกี",
        "kg": "คีร์กีซสถาน",
    },
}
# ru: «где» с предлогом (в/на + предложный падеж) — «в {name}» даёт «в Бразилия»
GEO_LOC = {
    # ⛔ Не «в Везде»: русские строки шаблона подставляют падежную форму, и для
    # псевдо-гео её надо задать явно.
    "any": "везде",
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


def geo_name(geo, lang="ru"):
    """Имя страны для языка. Порядок: наше переопределение → справочник → код.

    ⛔ Код как имя — последнее средство и признак дефекта: на сайте это выглядело как
    «ae», «al», «bo» в списке стран. Для языков, кроме ru и en, справочник даёт английское
    имя — это хуже локализованного, но несравнимо лучше кода. Сторож на это есть.
    """
    own = GEO_NAMES.get(lang, {}).get(geo)
    if own:
        return own
    if lang == "ru" and geo in REF:
        return REF[geo][0]
    return REF_EN.get(geo) or (REF[geo][0] if geo in REF else geo)


# Знаки для ПСЕВДО-гео: они не страны, и флага у них в справочнике нет по определению.
# ⛔ Сюда нельзя добавлять настоящие страны — это снова начало второй таблицы флагов
# (сторож `test_geo_names.py` падает, если тут появится код из справочника).
PSEUDO_FLAG = {"any": "🌍"}


def geo_flag(geo):
    """Знак гео: справочник → псевдо-гео → заглушка. Своей таблицы флагов для СТРАН нет —
    она совпадала со справочником один в один на всех 35, то есть была чистым дублем."""
    if geo in REF:
        return REF[geo][1]
    return PSEUDO_FLAG.get(geo, "•")


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
    "zh": {
        "FHEAD": [
            "{g}的{t}：来自聊天群的真实经验",
            "{t}：在{g}实际是怎么运作的 — 第一手",
            "{g}的{t}：真正需要知道的事",
        ],
        "QHEAD": [
            "{t}：大家在聊天群里都在问什么",
            "{t}：来自真实聊天群的常见问题",
            "{t}：经常被问却很少有人回答的问题",
        ],
        "fact_title": "{name}：{tl} — 真实经验 · Luky",
        "fact_desc": "来自聊天群的关于{name}「{tl}」的真实经验：原样呈现，第一手。你的具体情况 — 问 Luky。",
        "fact_intro": "聊天群里的人关于{name}「{tl}」的真实经验 — 原样呈现，不掺水。你的具体情况 — <a href='#luky'>问 Luky</a>。",
        "fact_list_label": "来自真实经验",
        "shelf_title": "{name}：{tl} — 来自聊天群的真实经验 · Luky",
        "shelf_desc": "关于{name}「{tl}」的真实经验：来自聊天群的实用技巧、案例和规定。你的具体情况 — 问 Luky。",
        "shelf_intro": "从真实经验中整理：{name}「{tl}」 — 技巧、案例和规定，原样呈现。你的具体情况 — <a href='#luky'>问 Luky</a>。",
        "shelf_list_label": "来自真实经验",
        "fact_blurb": "来自聊天群的 {n} 条建议",
        "shelf_blurb": "来自聊天群的 {n} 条记录",
        "bridge_shelf_blurb": "按主题整理的真实记录 — 你的情况请问 Luky",
        "bridge_shelf_title": "真实经验分区",
        "shub_title": "{name}：真实经验分区 — 全部来自聊天群 · Luky",
        "shub_desc": "{name}的真实经验，按分区整理：签证、钱、交通、证件、安全等等。你的具体情况 — 问 Luky。",
        "shub_h1": "真实经验分区",
        "shub_intro": "人们亲身经历过的一切 — 按分区整理。挑一个，具体情况就<a href='#luky'>问 Luky</a>。",
        "q_title": "{name}：{tl} — 大家在问什么 · Luky",
        "q_desc": "来自真实聊天群的关于{name}「{tl}」的真实问题。针对你的情况的回答 — 问 Luky。",
        "q_intro": "来自社区聊天群的鲜活问题 — 人们真正遇到的事。有同感吗？针对你情况的回答 — <a href='#luky'>问 Luky</a>。",
        "q_list_label": "来自聊天群的问题",
        "q_blurb": "来自聊天群的 {n} 个问题",
        "qhub_title": "{name}：大家在聊天群里问什么 — 真实问题 · Luky",
        "qhub_desc": "来自真实聊天群的关于{name}的问题：签证、钱、住房、安全。针对你情况的回答 — 问 Luky。",
        "qhub_h1": "大家在聊天群里问什么",
        "qhub_intro": "成百上千的人 — 同样的困惑。挑一个主题，看看真实的问题。别人慢慢找答案…… 而 <a href='#luky'>Luky</a> — 立刻就有。",
        "bridge_title": "大家在聊天群里问什么",
        "bridge_blurb": "来自真人的真实问题 — 你的情况请问 Luky",
        "hub_title": "{name}：证件、钱、住房 — 来自聊天群的真实经验 · Luky",
        "hub_desc": "来自社区聊天群的{name}真实经验：证件、钱、住房、安全、交通。不掺水，针对你的情况。",
        "hub_intro": "真正亲身经历过的人的经验 — 直奔要点，不掺水。挑一个主题，你的情况就<a href='#luky'>问 Luky</a>。",
        "list_label_topics": "主题",
        "lower": False,
    },
    "ja": {
        "FHEAD": [
            "{g}の{t}：チャットからの実際の経験",
            "{t}：{g}では実際どう動いているか — 一次情報",
            "{g}の{t}：本当に知っておくべきこと",
        ],
        "QHEAD": [
            "{t}：チャットで何が聞かれているか",
            "{t}：生きたチャットからのよくある質問",
            "{t}：よく聞かれるのに答えが返ってこないこと",
        ],
        "fact_title": "{name}：{tl} — 実際の経験 · Luky",
        "fact_desc": "{name}の「{tl}」についてチャットからの実際の経験：ありのまま、一次情報で。あなたのケースは — Luky に聞いて。",
        "fact_intro": "{name}の「{tl}」についてチャットにいる人たちの実際の経験 — ありのまま、水増しなし。あなたのケースは — <a href='#luky'>Luky に聞いて</a>。",
        "fact_list_label": "実際の経験から",
        "shelf_title": "{name}：{tl} — チャットからの実際の経験 · Luky",
        "shelf_desc": "{name}の「{tl}」についての実際の経験：チャットからの実用的なコツ、事例、ルール。あなたのケースは — Luky に聞いて。",
        "shelf_intro": "実際の経験からまとめました：{name}の「{tl}」 — コツ、事例、ルールをありのまま。あなたのケースは — <a href='#luky'>Luky に聞いて</a>。",
        "shelf_list_label": "実際の経験から",
        "fact_blurb": "チャットからの{n}件のコツ",
        "shelf_blurb": "チャットからの{n}件のメモ",
        "bridge_shelf_blurb": "テーマ別の実際のメモ — あなたのケースは Luky に",
        "bridge_shelf_title": "実際の経験のセクション",
        "shub_title": "{name}：実際の経験のセクション — すべてチャットから · Luky",
        "shub_desc": "{name}の実際の経験をセクション別に：ビザ、お金、交通、書類、安全など。あなたのケースは — Luky に聞いて。",
        "shub_h1": "実際の経験のセクション",
        "shub_intro": "人が自分で通ってきたことすべて — セクション別に。自分に合うものを選んで、具体的なケースは<a href='#luky'>Luky に聞いて</a>。",
        "q_title": "{name}：{tl} — 何が聞かれているか · Luky",
        "q_desc": "{name}の「{tl}」について生きたチャットからの実際の質問。あなたのケースへの答えは — Luky に聞いて。",
        "q_intro": "コミュニティのチャットからの生きた質問 — 人が本当に直面すること。心当たりある？あなたのケースへの答えは — <a href='#luky'>Luky に聞いて</a>。",
        "q_list_label": "チャットからの質問",
        "q_blurb": "チャットからの{n}件の質問",
        "qhub_title": "{name}：チャットで何が聞かれているか — 実際の質問 · Luky",
        "qhub_desc": "{name}について生きたチャットからの実際の質問：ビザ、お金、住まい、安全。あなたのケースへの答えは — Luky に聞いて。",
        "qhub_h1": "チャットで何が聞かれているか",
        "qhub_intro": "何百人もの人 — 同じつまずき。テーマを選んで実際の質問を見てみて。人は答えを見つけるのが遅い…… でも <a href='#luky'>Luky</a> なら — すぐ。",
        "bridge_title": "チャットで何が聞かれているか",
        "bridge_blurb": "本物の人の実際の質問 — あなたのケースは Luky に",
        "hub_title": "{name}：書類、お金、住まい — チャットからの実際の経験 · Luky",
        "hub_desc": "コミュニティのチャットからの{name}の実際の経験：書類、お金、住まい、安全、交通。水増しなし、あなたのケースに。",
        "hub_intro": "実際に自分で通ってきた人たちの経験 — 要点だけ、水増しなし。テーマを選んで、あなたのケースは<a href='#luky'>Luky に聞いて</a>。",
        "list_label_topics": "テーマ",
        "lower": False,
    },
    "ko": {
        "FHEAD": [
            "{g} {t}: 채팅방에서 나온 실제 경험",
            "{t}: {g}에서 실제로 어떻게 돌아가는지 — 직접 들은 이야기",
            "{g} {t}: 정말 알아야 할 것들",
        ],
        "QHEAD": [
            "{t}: 채팅방에서 무엇을 묻는가",
            "{t}: 살아 있는 채팅방의 자주 나오는 질문",
            "{t}: 자주 묻지만 답을 못 받는 것들",
        ],
        "fact_title": "{name}: {tl} — 실제 경험 · Luky",
        "fact_desc": "{name}의 「{tl}」에 대한 채팅방의 실제 경험: 있는 그대로, 직접 들은 이야기. 당신의 상황은 — Luky에게 물어보세요.",
        "fact_intro": "{name}의 「{tl}」에 대해 채팅방 사람들의 실제 경험 — 있는 그대로, 군더더기 없이. 당신의 상황은 — <a href='#luky'>Luky에게 물어보세요</a>.",
        "fact_list_label": "실제 경험에서",
        "shelf_title": "{name}: {tl} — 채팅방에서 나온 실제 경험 · Luky",
        "shelf_desc": "{name}의 「{tl}」에 대한 실제 경험: 채팅방에서 나온 실용적인 팁, 사례, 규정. 당신의 상황은 — Luky에게 물어보세요.",
        "shelf_intro": "실제 경험에서 모았습니다: {name}의 「{tl}」 — 팁, 사례, 규정을 있는 그대로. 당신의 상황은 — <a href='#luky'>Luky에게 물어보세요</a>.",
        "shelf_list_label": "실제 경험에서",
        "fact_blurb": "채팅방에서 나온 팁 {n}건",
        "shelf_blurb": "채팅방에서 나온 메모 {n}건",
        "bridge_shelf_blurb": "주제별 실제 메모 — 당신의 상황은 Luky에게",
        "bridge_shelf_title": "실제 경험 섹션",
        "shub_title": "{name}: 실제 경험 섹션 — 모두 채팅방에서 · Luky",
        "shub_desc": "{name}의 실제 경험을 섹션별로: 비자, 돈, 교통, 서류, 안전 등. 당신의 상황은 — Luky에게 물어보세요.",
        "shub_h1": "실제 경험 섹션",
        "shub_intro": "사람들이 직접 겪은 모든 것 — 섹션별로. 자신에게 맞는 것을 골라보고, 구체적인 상황은 <a href='#luky'>Luky에게 물어보세요</a>.",
        "q_title": "{name}: {tl} — 무엇을 묻는가 · Luky",
        "q_desc": "{name}의 「{tl}」에 대해 살아 있는 채팅방의 실제 질문들. 당신의 상황에 대한 답은 — Luky에게 물어보세요.",
        "q_intro": "커뮤니티 채팅방의 살아 있는 질문들 — 사람들이 실제로 마주치는 것. 익숙한가요? 당신의 상황에 대한 답은 — <a href='#luky'>Luky에게 물어보세요</a>.",
        "q_list_label": "채팅방의 질문",
        "q_blurb": "채팅방에서 나온 질문 {n}건",
        "qhub_title": "{name}: 채팅방에서 무엇을 묻는가 — 실제 질문 · Luky",
        "qhub_desc": "{name}에 대해 살아 있는 채팅방의 실제 질문: 비자, 돈, 주거, 안전. 당신의 상황에 대한 답은 — Luky에게 물어보세요.",
        "qhub_h1": "채팅방에서 무엇을 묻는가",
        "qhub_intro": "수백 명의 사람 — 똑같은 막막함. 주제를 골라 실제 질문을 보세요. 사람들은 답을 천천히 찾지만…… <a href='#luky'>Luky</a>는 — 바로.",
        "bridge_title": "채팅방에서 무엇을 묻는가",
        "bridge_blurb": "실제 사람들의 실제 질문 — 당신의 상황은 Luky에게",
        "hub_title": "{name}: 서류, 돈, 주거 — 채팅방에서 나온 실제 경험 · Luky",
        "hub_desc": "커뮤니티 채팅방에서 나온 {name} 실제 경험: 서류, 돈, 주거, 안전, 교통. 군더더기 없이, 당신의 상황에.",
        "hub_intro": "직접 겪어본 사람들의 실제 경험 — 요점만, 군더더기 없이. 주제를 고르고, 당신의 상황은 <a href='#luky'>Luky에게 물어보세요</a>.",
        "list_label_topics": "주제",
        "lower": False,
    },
    "ar": {
        "FHEAD": [
            "{t} في {g}: تجربة حقيقية من المحادثات",
            "{t}: كيف يعمل الأمر فعلاً في {g} — من مصدر أول",
            "{t} في {g}: ما يجب معرفته فعلاً",
        ],
        "QHEAD": [
            "{t}: ما يسأل عنه الناس في المحادثات",
            "{t}: أسئلة متكرّرة من محادثات حيّة",
            "{t}: ما يُسأل كثيراً ويبقى بلا جواب",
        ],
        "fact_title": "{name}: {tl} — تجربة حقيقية · Luky",
        "fact_desc": "تجربة حقيقية من المحادثات عن «{tl}» في {name}: كما هي، من مصدر أول. لحالتك — اسأل Luky.",
        "fact_intro": "تجربة الناس الحقيقية من المحادثات حول «{tl}» في {name} — كما هي، بلا حشو. لحالتك — <a href='#luky'>اسأل Luky</a>.",
        "fact_list_label": "من تجربة حقيقية",
        "shelf_title": "{name}: {tl} — تجربة حقيقية من المحادثات · Luky",
        "shelf_desc": "تجربة حقيقية حول «{tl}» في {name}: نصائح عملية وحالات وقواعد من المحادثات. لحالتك — اسأل Luky.",
        "shelf_intro": "مجموعة من تجربة حقيقية: «{tl}» في {name} — نصائح وحالات وقواعد كما هي. لحالتك — <a href='#luky'>اسأل Luky</a>.",
        "shelf_list_label": "من تجربة حقيقية",
        "fact_blurb": "{n} نصيحة من المحادثات",
        "shelf_blurb": "{n} ملاحظة من المحادثات",
        "bridge_shelf_blurb": "ملاحظات حقيقية حسب الموضوع — لحالتك اسأل Luky",
        "bridge_shelf_title": "أقسام التجربة الحقيقية",
        "shub_title": "{name}: أقسام التجربة الحقيقية — كلها من المحادثات · Luky",
        "shub_desc": "تجربة حقيقية في {name} حسب الأقسام: التأشيرات، المال، النقل، الأوراق، الأمان وغيرها. لحالتك — اسأل Luky.",
        "shub_h1": "أقسام التجربة الحقيقية",
        "shub_intro": "كل ما مرّ به الناس بأنفسهم — حسب الأقسام. اختر قسمك، ولحالتك تحديداً <a href='#luky'>اسأل Luky</a>.",
        "q_title": "{name}: {tl} — ما يسأل عنه الناس · Luky",
        "q_desc": "أسئلة حقيقية عن «{tl}» في {name} من محادثات حيّة. جواب لحالتك — اسأل Luky.",
        "q_intro": "أسئلة حيّة من محادثات المجتمع — ما يواجهه الناس فعلاً. يبدو مألوفاً؟ جواب لحالتك — <a href='#luky'>اسأل Luky</a>.",
        "q_list_label": "أسئلة من المحادثات",
        "q_blurb": "{n} سؤال من المحادثات",
        "qhub_title": "{name}: ما يسأل عنه الناس في المحادثات — أسئلة حقيقية · Luky",
        "qhub_desc": "أسئلة حقيقية عن {name} من محادثات حيّة: التأشيرات، المال، السكن، الأمان. جواب لحالتك — اسأل Luky.",
        "qhub_h1": "ما يسأل عنه الناس في المحادثات",
        "qhub_intro": "مئات الأشخاص — الحيرة نفسها. اختر موضوعاً وانظر الأسئلة الحقيقية. الناس يجدون الجواب ببطء… أما <a href='#luky'>Luky</a> — فوراً.",
        "bridge_title": "ما يسأل عنه الناس في المحادثات",
        "bridge_blurb": "أسئلة حقيقية من أشخاص حقيقيين — لحالتك اسأل Luky",
        "hub_title": "{name}: أوراق، مال، سكن — تجربة حقيقية من المحادثات · Luky",
        "hub_desc": "تجربة حقيقية عن {name} من محادثات المجتمع: أوراق، مال، سكن، أمان، نقل. بلا حشو، لحالتك.",
        "hub_intro": "تجربة حقيقية لمن مرّ بها بنفسه — إلى الهدف مباشرة، بلا حشو. اختر موضوعاً، ولحالتك <a href='#luky'>اسأل Luky</a>.",
        "list_label_topics": "المواضيع",
        "lower": False,
    },
    "hi": {
        "FHEAD": [
            "{g} में {t}: चैट से असली अनुभव",
            "{t}: {g} में यह असल में कैसे चलता है — पहले हाथ से",
            "{g} में {t}: जो असल में जानना ज़रूरी है",
        ],
        "QHEAD": [
            "{t}: चैट में लोग क्या पूछते हैं",
            "{t}: ज़िंदा चैट से आम सवाल",
            "{t}: जो बार-बार पूछा जाता है और जवाब नहीं मिलता",
        ],
        "fact_title": "{name}: {tl} — असली अनुभव · Luky",
        "fact_desc": "{name} में «{tl}» पर चैट से असली अनुभव: जैसा है वैसा, पहले हाथ से। अपने मामले के लिए — Luky से पूछो।",
        "fact_intro": "{name} में «{tl}» पर चैट में लोगों का असली अनुभव — जैसा है वैसा, बिना लफ़्फ़ाज़ी। अपने मामले के लिए — <a href='#luky'>Luky से पूछो</a>।",
        "fact_list_label": "असली अनुभव से",
        "shelf_title": "{name}: {tl} — चैट से असली अनुभव · Luky",
        "shelf_desc": "{name} में «{tl}» पर असली अनुभव: चैट से काम के तरीके, मामले और नियम। अपने मामले के लिए — Luky से पूछो।",
        "shelf_intro": "असली अनुभव से जुटाया: {name} में «{tl}» — तरीके, मामले और नियम जैसे हैं वैसे। अपने मामले के लिए — <a href='#luky'>Luky से पूछो</a>।",
        "shelf_list_label": "असली अनुभव से",
        "fact_blurb": "चैट से {n} तरीके",
        "shelf_blurb": "चैट से {n} नोट",
        "bridge_shelf_blurb": "विषय के हिसाब से असली नोट — अपने मामले के लिए Luky से पूछो",
        "bridge_shelf_title": "असली अनुभव के हिस्से",
        "shub_title": "{name}: असली अनुभव के हिस्से — सब चैट से · Luky",
        "shub_desc": "{name} का असली अनुभव हिस्सों में: वीज़ा, पैसा, आवाजाही, दस्तावेज़, सुरक्षा और बाकी। अपने मामले के लिए — Luky से पूछो।",
        "shub_h1": "असली अनुभव के हिस्से",
        "shub_intro": "वह सब जो लोग ख़ुद झेल चुके हैं — हिस्सों में। अपना चुनो, और ख़ास अपने मामले के लिए <a href='#luky'>Luky से पूछो</a>।",
        "q_title": "{name}: {tl} — लोग क्या पूछते हैं · Luky",
        "q_desc": "{name} में «{tl}» पर ज़िंदा चैट से असली सवाल। अपने मामले का जवाब — Luky से पूछो।",
        "q_intro": "कम्युनिटी चैट से ज़िंदा सवाल — जो लोग असल में झेलते हैं। पहचाना? अपने मामले का जवाब — <a href='#luky'>Luky से पूछो</a>।",
        "q_list_label": "चैट से सवाल",
        "q_blurb": "चैट से {n} सवाल",
        "qhub_title": "{name}: चैट में लोग क्या पूछते हैं — असली सवाल · Luky",
        "qhub_desc": "{name} पर ज़िंदा चैट से असली सवाल: वीज़ा, पैसा, रहने की जगह, सुरक्षा। अपने मामले का जवाब — Luky से पूछो।",
        "qhub_h1": "चैट में लोग क्या पूछते हैं",
        "qhub_intro": "सैकड़ों लोग — वही उलझनें। एक विषय चुनो और असली सवाल देखो। लोग जवाब धीरे ढूँढते हैं… पर <a href='#luky'>Luky</a> — तुरंत।",
        "bridge_title": "चैट में लोग क्या पूछते हैं",
        "bridge_blurb": "असली लोगों के असली सवाल — अपने मामले के लिए Luky से पूछो",
        "hub_title": "{name}: दस्तावेज़, पैसा, रहने की जगह — चैट से असली अनुभव · Luky",
        "hub_desc": "कम्युनिटी चैट से {name} का असली अनुभव: दस्तावेज़, पैसा, रहने की जगह, सुरक्षा, आवाजाही। बिना लफ़्फ़ाज़ी, अपने मामले के लिए।",
        "hub_intro": "जिन्होंने ख़ुद झेला उनका असली अनुभव — सीधे मुद्दे पर, बिना लफ़्फ़ाज़ी। एक विषय चुनो, और अपने मामले के लिए <a href='#luky'>Luky से पूछो</a>।",
        "list_label_topics": "विषय",
        "lower": False,
    },
    "th": {
        "FHEAD": [
            "{t} ใน{g}: ประสบการณ์จริงจากแชท",
            "{t}: ใน{g}เอาจริงแล้วเป็นอย่างไร — จากปากคนที่เจอเอง",
            "{t} ใน{g}: สิ่งที่ควรรู้จริง ๆ",
        ],
        "QHEAD": [
            "{t}: ในแชทคนถามอะไรกัน",
            "{t}: คำถามที่เจอบ่อยจากแชทจริง",
            "{t}: สิ่งที่ถามกันบ่อยแต่ไม่ค่อยมีคำตอบ",
        ],
        "fact_title": "{name}: {tl} — ประสบการณ์จริง · Luky",
        "fact_desc": "ประสบการณ์จริงจากแชทเรื่อง «{tl}» ใน{name}: ตามที่เป็น จากปากคนที่เจอเอง สำหรับกรณีของคุณ — ถาม Luky",
        "fact_intro": "ประสบการณ์จริงของคนในแชทเรื่อง «{tl}» ใน{name} — ตามที่เป็น ไม่มีน้ำ สำหรับกรณีของคุณ — <a href='#luky'>ถาม Luky</a>",
        "fact_list_label": "จากประสบการณ์จริง",
        "shelf_title": "{name}: {tl} — ประสบการณ์จริงจากแชท · Luky",
        "shelf_desc": "ประสบการณ์จริงเรื่อง «{tl}» ใน{name}: เคล็ดลับที่ใช้ได้ กรณีจริง และกฎจากแชท สำหรับกรณีของคุณ — ถาม Luky",
        "shelf_intro": "รวบรวมจากประสบการณ์จริง: «{tl}» ใน{name} — เคล็ดลับ กรณีจริง และกฎตามที่เป็น สำหรับกรณีของคุณ — <a href='#luky'>ถาม Luky</a>",
        "shelf_list_label": "จากประสบการณ์จริง",
        "fact_blurb": "เคล็ดลับ {n} ข้อจากแชท",
        "shelf_blurb": "บันทึก {n} ข้อจากแชท",
        "bridge_shelf_blurb": "บันทึกจริงแยกตามหัวข้อ — กรณีของคุณถาม Luky",
        "bridge_shelf_title": "หมวดประสบการณ์จริง",
        "shub_title": "{name}: หมวดประสบการณ์จริง — ทั้งหมดจากแชท · Luky",
        "shub_desc": "ประสบการณ์จริงใน{name}แยกเป็นหมวด: วีซ่า เงิน การเดินทาง เอกสาร ความปลอดภัย และอื่น ๆ สำหรับกรณีของคุณ — ถาม Luky",
        "shub_h1": "หมวดประสบการณ์จริง",
        "shub_intro": "ทุกอย่างที่คนผ่านมาด้วยตัวเอง — แยกเป็นหมวด เลือกหมวดของคุณ และสำหรับกรณีเฉพาะของคุณ <a href='#luky'>ถาม Luky</a>",
        "q_title": "{name}: {tl} — คนถามอะไรกัน · Luky",
        "q_desc": "คำถามจริงเรื่อง «{tl}» ใน{name} จากแชทจริง คำตอบสำหรับกรณีของคุณ — ถาม Luky",
        "q_intro": "คำถามสด ๆ จากแชทของคอมมูนิตี้ — สิ่งที่คนเจอจริง คุ้น ๆ ไหม? คำตอบสำหรับกรณีของคุณ — <a href='#luky'>ถาม Luky</a>",
        "q_list_label": "คำถามจากแชท",
        "q_blurb": "คำถาม {n} ข้อจากแชท",
        "qhub_title": "{name}: ในแชทคนถามอะไรกัน — คำถามจริง · Luky",
        "qhub_desc": "คำถามจริงเรื่อง{name}จากแชทจริง: วีซ่า เงิน ที่อยู่ ความปลอดภัย คำตอบสำหรับกรณีของคุณ — ถาม Luky",
        "qhub_h1": "ในแชทคนถามอะไรกัน",
        "qhub_intro": "คนหลายร้อย — สับสนเรื่องเดียวกัน เลือกหัวข้อแล้วดูคำถามจริง คนหาคำตอบได้ช้า… แต่ <a href='#luky'>Luky</a> — ทันที",
        "bridge_title": "ในแชทคนถามอะไรกัน",
        "bridge_blurb": "คำถามจริงจากคนจริง — กรณีของคุณถาม Luky",
        "hub_title": "{name}: เอกสาร เงิน ที่อยู่ — ประสบการณ์จริงจากแชท · Luky",
        "hub_desc": "ประสบการณ์จริงเรื่อง{name}จากแชทของคอมมูนิตี้: เอกสาร เงิน ที่อยู่ ความปลอดภัย การเดินทาง ไม่มีน้ำ ตรงกรณีของคุณ",
        "hub_intro": "ประสบการณ์จริงของคนที่ผ่านมาเองจริง ๆ — ตรงประเด็น ไม่มีน้ำ เลือกหัวข้อ และสำหรับกรณีของคุณ <a href='#luky'>ถาม Luky</a>",
        "list_label_topics": "หัวข้อ",
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
    "zh": {
        "home_title": "Luky — 按国家整理的真实经验：钱、证件、住房",
        "home_desc": "Luky 信息门户：按国家整理来自社区聊天群的真实经验 — 钱、证件、住房、安全。不掺水，针对你的情况。",
        "home_h1": "你要去哪儿？",
        "home_intro": "已经亲身走过当地那些麻烦的人的真实经验 — 直奔要点，不掺水。挑一个国家，你的情况就<a href='#luky'>问 Luky</a>。",
        "home_list_label": "国家",
        "geo_blurb": "真实经验",
        "about_crumb": "关于",
        "about_title": "关于 · Luky",
        "about_desc": "Luky 按国家收集来自公开社区聊天群的真实经验 — 直奔要点，不掺水。",
        "about_h1": "关于",
        "about_body": "<p><a href='#luky'>Luky</a> 是真实的人的经验，不是干巴巴的理论。我们从公开的社区聊天群里收集真实建议：当地实际怎么运作、要避开什么、现在什么方法有效。</p><p>这个门户是这些经验按主题和国家的橱窗。至于你的具体情况，可以<a href='#luky'>问 Luky</a> — 它根据人们最近的反馈来回答。</p>",
    },
    "ja": {
        "home_title": "Luky — 国ごとの実際の経験：お金、書類、住まい",
        "home_desc": "Luky 情報ポータル：国ごとにコミュニティのチャットからの実際の経験 — お金、書類、住まい、安全。水増しなし、あなたのケースに。",
        "home_h1": "どこへ行くの？",
        "home_intro": "現地のややこしさをすでに自分で通り抜けた人たちの実際の経験 — 要点だけ、水増しなし。国を選んで、あなたのケースは<a href='#luky'>Luky に聞いて</a>。",
        "home_list_label": "国",
        "geo_blurb": "実際の経験",
        "about_crumb": "概要",
        "about_title": "概要 · Luky",
        "about_desc": "Luky は国ごとに公開コミュニティのチャットから実際の経験を集めます — 要点だけ、水増しなし。",
        "about_h1": "概要",
        "about_body": "<p><a href='#luky'>Luky</a> は乾いた理論ではなく、実際の人の経験です。公開されたコミュニティのチャットから本物のコツを集めています：現地で実際どう動くのか、何を避けるべきか、いま何が使えるのか。</p><p>このポータルはその経験をテーマと国ごとに並べたショーケースです。具体的なケースは<a href='#luky'>Luky に聞けます</a> — 人々の最近の報告から答えます。</p>",
    },
    "ko": {
        "home_title": "Luky — 나라별 실제 경험: 돈, 서류, 주거",
        "home_desc": "Luky 정보 포털: 나라별로 커뮤니티 채팅방에서 모은 실제 경험 — 돈, 서류, 주거, 안전. 군더더기 없이, 당신의 상황에.",
        "home_h1": "어디로 가세요?",
        "home_intro": "현지의 막막함을 이미 직접 겪어본 사람들의 실제 경험 — 요점만, 군더더기 없이. 나라를 고르고, 당신의 상황은 <a href='#luky'>Luky에게 물어보세요</a>.",
        "home_list_label": "나라",
        "geo_blurb": "실제 경험",
        "about_crumb": "소개",
        "about_title": "소개 · Luky",
        "about_desc": "Luky는 나라별로 공개 커뮤니티 채팅방에서 실제 경험을 모읍니다 — 요점만, 군더더기 없이.",
        "about_h1": "소개",
        "about_body": "<p><a href='#luky'>Luky</a>는 마른 이론이 아니라 실제 사람들의 경험입니다. 공개된 커뮤니티 채팅방에서 진짜 팁을 모읍니다: 현지에서 실제로 어떻게 돌아가는지, 무엇을 피해야 하는지, 지금 무엇이 통하는지.</p><p>이 포털은 그 경험을 주제와 나라별로 진열한 곳입니다. 구체적인 상황은 <a href='#luky'>Luky에게 물어볼 수 있습니다</a> — 사람들의 최근 후기를 바탕으로 답합니다.</p>",
    },
    "ar": {
        "home_title": "Luky — تجربة حقيقية حسب البلد: مال، أوراق، سكن",
        "home_desc": "بوابة Luky المعلوماتية: تجربة حقيقية من محادثات المجتمع حسب البلد — مال، أوراق، سكن، أمان. بلا حشو، لحالتك.",
        "home_h1": "إلى أين أنت ذاهب؟",
        "home_intro": "تجربة حقيقية لمن تجاوز بنفسه متاهات المكان — إلى الهدف مباشرة، بلا حشو. اختر بلداً، ولحالتك <a href='#luky'>اسأل Luky</a>.",
        "home_list_label": "البلدان",
        "geo_blurb": "تجربة حقيقية",
        "about_crumb": "عن المشروع",
        "about_title": "عن المشروع · Luky",
        "about_desc": "يجمع Luky تجربة حقيقية من محادثات المجتمع المفتوحة حسب البلد — إلى الهدف مباشرة، بلا حشو.",
        "about_h1": "عن المشروع",
        "about_body": "<p><a href='#luky'>Luky</a> هو تجربة أشخاص حقيقيين، لا نظرية جافة. نجمع نصائح حقيقية من محادثات المجتمع المفتوحة: كيف تسير الأمور فعلاً على الأرض، ما يجب تجنّبه، وما ينفع الآن.</p><p>البوابة واجهة لهذه التجربة حسب الموضوع والبلد. ولحالتك تحديداً يمكنك أن <a href='#luky'>تسأل Luky</a> — يجيب من تقارير الناس الأخيرة.</p>",
    },
    "hi": {
        "home_title": "Luky — देश के हिसाब से असली अनुभव: पैसा, दस्तावेज़, रहने की जगह",
        "home_desc": "Luky जानकारी पोर्टल: देश के हिसाब से कम्युनिटी चैट से असली अनुभव — पैसा, दस्तावेज़, रहने की जगह, सुरक्षा। बिना लफ़्फ़ाज़ी, अपने मामले के लिए।",
        "home_h1": "कहाँ जा रहे हो?",
        "home_intro": "जो वहाँ की उलझनें ख़ुद झेल चुके हैं उनका असली अनुभव — सीधे मुद्दे पर, बिना लफ़्फ़ाज़ी। एक देश चुनो, और अपने मामले के लिए <a href='#luky'>Luky से पूछो</a>।",
        "home_list_label": "देश",
        "geo_blurb": "असली अनुभव",
        "about_crumb": "बारे में",
        "about_title": "बारे में · Luky",
        "about_desc": "Luky देश के हिसाब से खुली कम्युनिटी चैट से असली अनुभव जुटाता है — सीधे मुद्दे पर, बिना लफ़्फ़ाज़ी।",
        "about_h1": "बारे में",
        "about_body": "<p><a href='#luky'>Luky</a> असली लोगों का अनुभव है, सूखी थ्योरी नहीं। हम खुली कम्युनिटी चैट से सच्चे तरीके जुटाते हैं: ज़मीन पर चीज़ें असल में कैसे चलती हैं, किससे बचना है, अभी क्या काम करता है।</p><p>पोर्टल इस अनुभव की विषय और देश के हिसाब से झलक है। और ख़ास अपने मामले के लिए तुम <a href='#luky'>Luky से पूछ सकते हो</a> — यह लोगों की हाल की रिपोर्टों से जवाब देता है।</p>",
    },
    "th": {
        "home_title": "Luky — ประสบการณ์จริงแยกตามประเทศ: เงิน เอกสาร ที่อยู่",
        "home_desc": "พอร์ทัลข้อมูล Luky: ประสบการณ์จริงจากแชทของคอมมูนิตี้แยกตามประเทศ — เงิน เอกสาร ที่อยู่ ความปลอดภัย ไม่มีน้ำ ตรงกรณีของคุณ",
        "home_h1": "จะไปไหนดี?",
        "home_intro": "ประสบการณ์จริงของคนที่ผ่านความสับสนของที่นั่นมาแล้วด้วยตัวเอง — ตรงประเด็น ไม่มีน้ำ เลือกประเทศ และสำหรับกรณีของคุณ <a href='#luky'>ถาม Luky</a>",
        "home_list_label": "ประเทศ",
        "geo_blurb": "ประสบการณ์จริง",
        "about_crumb": "เกี่ยวกับ",
        "about_title": "เกี่ยวกับ · Luky",
        "about_desc": "Luky รวบรวมประสบการณ์จริงจากแชทเปิดของคอมมูนิตี้แยกตามประเทศ — ตรงประเด็น ไม่มีน้ำ",
        "about_h1": "เกี่ยวกับ",
        "about_body": "<p><a href='#luky'>Luky</a> คือประสบการณ์ของคนจริง ไม่ใช่ทฤษฎีแห้ง ๆ เรารวบรวมเคล็ดลับจริงจากแชทเปิดของคอมมูนิตี้: หน้างานเอาจริงแล้วเป็นอย่างไร ควรเลี่ยงอะไร อะไรใช้ได้อยู่ตอนนี้</p><p>พอร์ทัลนี้คือหน้าร้านของประสบการณ์นั้น แยกตามหัวข้อและประเทศ และสำหรับกรณีเฉพาะของคุณ <a href='#luky'>ถาม Luky ได้</a> — ตอบจากรายงานล่าสุดของผู้คน</p>",
    },
}


# ── Портал-home: регионы + образные вайбы стран ──
# ── Регионы: ПОЛНОЕ покрытие справочника, а не «наши гео» ──
# ⛔ 2026-08-11: регион был у 35 кодов из 249, всё остальное валилось в «Другие» — 55 позиций
# сырыми кодами на главной. Юзер: «звучит как опять урезанная версия на сейчас». Поэтому
# раскладка покрывает справочник ЦЕЛИКОМ, а сторож падает, если появилась страна без региона.
# Восемь регионов вместо шести: Африка отделена от Ближнего Востока, добавлена Северная
# Америка — до этого Канада и США жили в «Других».
# ⚠️ Ключи — СЛОВАМИ, не двухбуквенные: `me`, `na`, `af`, `la` заняты реальными странами
# (Черногория, Намибия, Афганистан, Лаос), и совпадение ключа с кодом — мина для читателя.
REGION_CODES = {
    "europe": {
        "ad",
        "al",
        "at",
        "ax",
        "ba",
        "be",
        "bg",
        "ch",
        "cy",
        "cz",
        "de",
        "dk",
        "ee",
        "es",
        "fi",
        "fo",
        "fr",
        "gb",
        "gg",
        "gi",
        "gr",
        "hr",
        "hu",
        "ie",
        "im",
        "is",
        "it",
        "je",
        "li",
        "lt",
        "lu",
        "lv",
        "mc",
        "me",
        "mk",
        "mt",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "rs",
        "se",
        "si",
        "sj",
        "sk",
        "sm",
        "va",
        "xk",
    },
    "cis": {
        "am",
        "az",
        "by",
        "ge",
        "kg",
        "kz",
        "md",
        "ru",
        "tj",
        "tm",
        "ua",
        "uz",
    },
    "asia": {
        "af",
        "bd",
        "bn",
        "bt",
        "cn",
        "hk",
        "id",
        "in",
        "jp",
        "kh",
        "kp",
        "kr",
        "la",
        "lk",
        "mm",
        "mn",
        "mo",
        "mv",
        "my",
        "np",
        "ph",
        "pk",
        "sg",
        "th",
        "tl",
        "tw",
        "vn",
    },
    "mideast": {
        "ae",
        "bh",
        "il",
        "iq",
        "ir",
        "jo",
        "kw",
        "lb",
        "om",
        "ps",
        "qa",
        "sa",
        "sy",
        "tr",
        "ye",
    },
    "africa": {
        "ao",
        "bf",
        "bi",
        "bj",
        "bw",
        "cd",
        "cf",
        "cg",
        "ci",
        "cm",
        "cv",
        "dj",
        "dz",
        "eg",
        "er",
        "et",
        "ga",
        "gh",
        "gm",
        "gn",
        "gq",
        "gw",
        "ke",
        "km",
        "lr",
        "ls",
        "ly",
        "ma",
        "mg",
        "ml",
        "mr",
        "mu",
        "mw",
        "mz",
        "na",
        "ne",
        "ng",
        "re",
        "rw",
        "sc",
        "sd",
        "sh",
        "sl",
        "sn",
        "so",
        "ss",
        "st",
        "sz",
        "td",
        "tg",
        "tn",
        "tz",
        "ug",
        "yt",
        "za",
        "zm",
        "zw",
    },
    "namerica": {
        "bm",
        "ca",
        "gl",
        "pm",
        "us",
    },
    "latam": {
        "ag",
        "ai",
        "ar",
        "aw",
        "bb",
        "bl",
        "bo",
        "bq",
        "br",
        "bs",
        "bz",
        "cl",
        "co",
        "cr",
        "cu",
        "cw",
        "dm",
        "do",
        "ec",
        "fk",
        "gd",
        "gf",
        "gp",
        "gs",
        "gt",
        "gy",
        "hn",
        "ht",
        "jm",
        "kn",
        "ky",
        "lc",
        "mf",
        "mq",
        "ms",
        "mx",
        "ni",
        "pa",
        "pe",
        "pr",
        "py",
        "sr",
        "sv",
        "sx",
        "tc",
        "tt",
        "uy",
        "vc",
        "ve",
        "vg",
        "vi",
    },
    "oceania": {
        "as",
        "au",
        "cc",
        "ck",
        "cx",
        "fj",
        "fm",
        "gu",
        "ki",
        "mh",
        "mp",
        "nc",
        "nf",
        "nr",
        "nu",
        "nz",
        "pf",
        "pg",
        "pn",
        "pw",
        "sb",
        "tk",
        "to",
        "tv",
        "um",
        "vu",
        "wf",
        "ws",
    },
    "other": {
        "aq",
        "bv",
        "hm",
        "io",
        "tf",
    },
}
# Нежилые и приполярные территории лежат в "other": страниц у них не бывает, но регион
# иметь обязаны, иначе сторож полноты бессмыслен.
REGION_ORDER = [
    "europe",
    "cis",
    "asia",
    "mideast",
    "africa",
    "namerica",
    "latam",
    "oceania",
]
CODE2REGION = {c: r for r, cs in REGION_CODES.items() for c in cs}
OTHER_REGION = "other"  # единственное место, где живёт этот ключ

REGION_NAMES = {
    "ru": {
        "europe": "Европа",
        "cis": "СНГ",
        "asia": "Азия",
        "mideast": "Ближний Восток",
        "africa": "Африка",
        "namerica": "Северная Америка",
        "latam": "Латинская Америка",
        "oceania": "Океания",
        "other": "Другие",
    },
    "en": {
        "europe": "Europe",
        "cis": "CIS",
        "asia": "Asia",
        "mideast": "Middle East",
        "africa": "Africa",
        "namerica": "North America",
        "latam": "Latin America",
        "oceania": "Oceania",
        "other": "Other",
    },
    "es": {
        "europe": "Europa",
        "cis": "CEI",
        "asia": "Asia",
        "mideast": "Oriente Medio",
        "africa": "África",
        "namerica": "América del Norte",
        "latam": "América Latina",
        "oceania": "Oceanía",
        "other": "Otros",
    },
    "pt": {
        "europe": "Europa",
        "cis": "CEI",
        "asia": "Ásia",
        "mideast": "Oriente Médio",
        "africa": "África",
        "namerica": "América do Norte",
        "latam": "América Latina",
        "oceania": "Oceania",
        "other": "Outros",
    },
    "de": {
        "europe": "Europa",
        "cis": "GUS",
        "asia": "Asien",
        "mideast": "Naher Osten",
        "africa": "Afrika",
        "namerica": "Nordamerika",
        "latam": "Lateinamerika",
        "oceania": "Ozeanien",
        "other": "Andere",
    },
    "fr": {
        "europe": "Europe",
        "cis": "CEI",
        "asia": "Asie",
        "mideast": "Moyen-Orient",
        "africa": "Afrique",
        "namerica": "Amérique du Nord",
        "latam": "Amérique latine",
        "oceania": "Océanie",
        "other": "Autres",
    },
    "it": {
        "europe": "Europa",
        "cis": "CSI",
        "asia": "Asia",
        "mideast": "Medio Oriente",
        "africa": "Africa",
        "namerica": "America del Nord",
        "latam": "America Latina",
        "oceania": "Oceania",
        "other": "Altri",
    },
    "tr": {
        "europe": "Avrupa",
        "cis": "BDT",
        "asia": "Asya",
        "mideast": "Orta Doğu",
        "africa": "Afrika",
        "namerica": "Kuzey Amerika",
        "latam": "Latin Amerika",
        "oceania": "Okyanusya",
        "other": "Diğer",
    },
    "ar": {
        "europe": "أوروبا",
        "cis": "رابطة الدول المستقلة",
        "asia": "آسيا",
        "mideast": "الشرق الأوسط",
        "africa": "أفريقيا",
        "namerica": "أمريكا الشمالية",
        "latam": "أمريكا اللاتينية",
        "oceania": "أوقيانوسيا",
        "other": "أخرى",
    },
    "hi": {
        "europe": "यूरोप",
        "cis": "सीआईएस",
        "asia": "एशिया",
        "mideast": "मध्य पूर्व",
        "africa": "अफ़्रीका",
        "namerica": "उत्तरी अमेरिका",
        "latam": "लैटिन अमेरिका",
        "oceania": "ओशिनिया",
        "other": "अन्य",
    },
    "ja": {
        "europe": "ヨーロッパ",
        "cis": "CIS",
        "asia": "アジア",
        "mideast": "中東",
        "africa": "アフリカ",
        "namerica": "北米",
        "latam": "ラテンアメリカ",
        "oceania": "オセアニア",
        "other": "その他",
    },
    "ko": {
        "europe": "유럽",
        "cis": "독립국가연합",
        "asia": "아시아",
        "mideast": "중동",
        "africa": "아프리카",
        "namerica": "북아메리카",
        "latam": "라틴아메리카",
        "oceania": "오세아니아",
        "other": "기타",
    },
    "th": {
        "europe": "ยุโรป",
        "cis": "เครือรัฐเอกราช",
        "asia": "เอเชีย",
        "mideast": "ตะวันออกกลาง",
        "africa": "แอฟริกา",
        "namerica": "อเมริกาเหนือ",
        "latam": "ละตินอเมริกา",
        "oceania": "โอเชียเนีย",
        "other": "อื่น ๆ",
    },
    "zh": {
        "europe": "欧洲",
        "cis": "独联体",
        "asia": "亚洲",
        "mideast": "中东",
        "africa": "非洲",
        "namerica": "北美洲",
        "latam": "拉丁美洲",
        "oceania": "大洋洲",
        "other": "其他",
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

    def nm(g):
        return geo_name(g, lang)

    def tile(g):
        return {"flag": geo_flag(g), "name": nm(g), "url": f"/{lang}/{g}/"}

    gs = sorted(geos, key=nm)
    search_index = [tile(g) for g in gs]
    pop_codes = sorted(geos, key=lambda g: (-counts.get(g, 0), nm(g)))[:12]
    vibe = VIBE.get(lang, {})
    popular = [{**tile(g), "vibe": vibe.get(g, "")} for g in pop_codes]
    rn = REGION_NAMES.get(lang, REGION_NAMES["en"])
    # ⛔ Ключ региона не зашивать литералом: строка "oth" тут пережила переименование ключей
    # и печаталась на главной как есть — «oth» вместо названия. Один источник — REGION_CODES.
    groups = {}
    for g in geos:
        if g not in CODE2REGION:
            continue  # не страна (`any` = «везде») — у неё своё место, не регион
        groups.setdefault(CODE2REGION[g], []).append(g)
    regions = []
    for rk in REGION_ORDER + [OTHER_REGION]:
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
    "zh": ("条反馈", "条反馈"),
    "ja": ("件の報告", "件の報告"),
    "ko": ("건의 후기", "건의 후기"),
    "ar": ("تقرير", "تقارير"),
    "hi": ("रिपोर्ट", "रिपोर्टें"),
    "th": ("รายงาน", "รายงาน"),
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

    ⛔ Возвращает None, когда адреса НЕТ: ключа нет, а из метки не осталось ни одного
    пригодного символа (zh ja ko ar hi th — `_TR` знает только кириллицу). Вызывающий
    ОБЯЗАН пропустить такую страницу. Выдумать ей имя нельзя: адрес вышел бы один и тот
    же у всех страниц гео, а уникализации адресов тут нет — страницы затёрли бы друг
    друга молча, оставив ~90 на язык вместо ~1843.
    """
    return obj.get("key") or slug_or_none(obj[label_field])


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
    subs_ok = [x for x in sv["subshelves"] if addr(x, "name")]
    sub_sibs = [{"name": sub["name"], "slug": addr(sub, "name")} for sub in subs_ok]
    subtiles = []
    for sub in subs_ok:
        ss = addr(sub, "name")
        sub_groups = [by_rep[r] for r in sub["reps"] if r in by_rep]
        sub_view = {"items": sv["items"], "groups": sub_groups}
        tl_ = ctx["tl"]
        spage = {
            "lang": lang,
            "template": "page.html.j2",
            "path": f"{url_pref}{ss}/",
            "shared_tail": bool(sub.get("key")),
            # раздел ветка наследует от своей страницы: на нём стоят и довод CTA, и шлюз
            # клика. Без этого ветвлённые страницы (202 в корпусе) остались бы без адресности.
            "shelf_key": ctx.get("shelf_key"),
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


# ── ПЛИТКИ РАЗДЕЛОВ НА ХАБЕ ГЕО (канон §0.12: «хаб страны = плитки полок со счётчиком;
#    адреса живут ВНУТРИ плитки, хаб их не перечисляет»).
#    Замер до правила: `/ru/gr/` — 63 ссылки плоским списком, `/ru/any/` — 87. После
#    раскладки разборов по разделам (13.08): у `gr` 6 разделов, у `me` 10, у `br` 13 —
#    потолок хаба 13 плиток + мостики, а не «сколько собралось страниц».
#
# ⛔ ВТОРАЯ ПОЛОВИНА ПРАВИЛА, которую я 13.08 сначала не исполнил: «хаб их НЕ ПЕРЕЧИСЛЯЕТ».
# Первый заход я сделал плитку-аккордеон — адреса лежали в HTML хаба под кликом, то есть
# правило нарушалось незаметно глазами. Плитка обязана быть ССЫЛКОЙ на страницу раздела,
# а сами адреса — перечислены уже там. Страница раздела не новая: полочный контур ниже
# строит её из хвоста, имя раздела то же самое (таксономия).
THEME_ICON = {  # по КЛЮЧУ полки, а не по имени: имя своё в каждом из 14 языков, ключ один
    "border": "🛬",  # 🛂 занят визами: два одинаковых значка на одном хабе — видно глазом
    "visa": "🛂",
    "finance": "💰",
    "transport": "🚕",
    "docs": "📄",
    "safety": "🛡",
    "customs": "📦",
    "digital": "📶",
    "tourism": "🗺",
    "housing": "🏠",
    "shopping": "🛒",
    "work": "💼",
    "health": "🩺",
}
_RU_THEMES = {}  # geo → {key разбора: латинский ключ его раздела}
_THEME_NAMES = {}  # lang → {латинский ключ раздела: имя на этом языке}


def ru_themes(geo):
    """Раздел разбора лежит в РУССКОМ корпусе: его пишет рот `assign`, перевод не несёт.

    Соединяем по `key` разбора — хвост адреса, одинаковый во всех языках по построению
    (штамповка адресов). Замер 13.08: штамп есть у 1889 разборов из 1889 в `ru` и у 1851
    из 1851 в `de`, так что соединение полное, а не выборочное.
    """
    if geo not in _RU_THEMES:
        d = load(f"{_facet_dir('ru')}/{geo}.json") or {}
        _RU_THEMES[geo] = {
            v["key"]: SHELF_KEY[v["shelf"]]
            for v in d.get("views_by_task") or []
            if v.get("key") and SHELF_KEY.get(v.get("shelf") or "")
        }
    return _RU_THEMES[geo]


def theme_names(lang):
    """Имя раздела на языке — из корпуса ЭТОГО языка (union по всем гео).

    Перевод уже несёт локализованное имя полки и латинский `key` рядом с ним, и этими
    же именами подписаны полочные страницы. Своей таблицы имён не заводим: она бы
    разъехалась с тем, что на сайте уже написано.
    """
    if lang == "ru":
        return {k: name for k, name, _ in _tax.SHELVES}
    if lang not in _THEME_NAMES:
        out = {}
        for p in sorted(glob.glob(f"{_facet_dir(lang)}/*.json")):
            for s in (load(p) or {}).get("shelves") or []:
                if s.get("key") and s.get("shelf"):
                    out.setdefault(s["key"], s["shelf"])
        _THEME_NAMES[lang] = out
    return _THEME_NAMES[lang]


def view_theme(v, geo, lang):
    """Латинский ключ раздела разбора: из самого разбора (ru) либо из русского по `key`."""
    k = SHELF_KEY.get(v.get("shelf") or "")
    if not k and lang != "ru":
        k = ru_themes(geo).get(v.get("key") or "")
    return k


def theme_tiles(cards, lang, geo, urls):
    """Плитки разделов для хаба гео. Возвращает (плитки, несгруппированное).

    `urls` — {ключ раздела: адрес его страницы}: плитка это ССЫЛКА, а не раскрытие, и
    вести ей некуда, если страницы раздела в этом гео нет. Так бывает: страница раздела
    строится из хвоста при трёх и более заметках. Замер 13.08: таких пар «гео × раздел»
    12 из 249 в 9 гео, за ними 23 разбора — они остаются обычными карточками.

    ⛔ ПОЛОВИНЧАТОГО НЕ ВЫПУСКАЕМ: если у гео есть раздел, у которого на этом языке нет
    имени, хаб остаётся плоским ЦЕЛИКОМ (как был) и причина печатается. Полу-состояния в
    этом проекте уже стоили прода: язык, полный наполовину, давал три разных исхода —
    тихий скип, KeyError и код страны вместо имени. Возврат `(None, …)` и значит «этот
    язык ещё не готов группировать»: языковые корпуса стоят на таксономии v0, и пяти
    новых имён (`tourism`, `housing`, `shopping`, `work`, `health`) там нет.

    Разбор без раздела плиткой не становится, но и не теряется: остаётся карточкой. Такие
    метки сборные («Прочее», «Общие советы») — брак нарезки, он лечится в карве, а до тех
    пор адрес обязан быть достижим с хаба.
    """
    names = theme_names(lang)
    groups, loose = {}, []
    for c in cards:
        k = c.pop("theme", None)
        if k:
            groups.setdefault(k, []).append(c)
        else:
            loose.append(c)
    # ⛔ ДВЕ ПРИЧИНЫ, И ИХ НЕЛЬЗЯ СЛИВАТЬ (поймано сторожем 13.08). Сначала я фильтровал по
    # `urls`, и раздел без ИМЕНИ на языке уходил тем же путём, что раздел без страницы, —
    # то есть язык, не готовый группировать, выглядел как «просто нет страниц» и молчал.
    # Нет имени → хаб плоский целиком и причина в логе. Нет страницы → карточки, тоже в лог.
    missing = sorted(k for k in groups if k not in names)
    if missing:
        print(
            f"{geo}/{lang}: хаб плоский — на этом языке нет имени разделов: "
            + ",".join(missing),
            flush=True,
        )
        return None, cards
    for k in [k for k in groups if k not in urls]:
        loose.extend(groups.pop(k))
    # порядок: крупный раздел выше, дальше по имени. Из словаря порядок пришёл бы от
    # порядка данных, и хаб перетряхивался бы каждый прогон — лишний дифф в репо страниц.
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), names[kv[0]]))
    tiles = [
        {
            "icon": THEME_ICON.get(k, "•"),
            "title": names[k],
            "blurb": blurb(COPY[lang], "fact", sum(x["n"] for x in cs)),
            "url": urls[k],
        }
        for k, cs in order
    ]
    return tiles, loose


def build_geo(geo, lang="ru"):
    C = COPY[lang]
    name = geo_name(geo, lang)
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

    # (отсев непереведённых меток переехал в `_skip` — одно место на обе причины)
    # ⛔ Безадресные виды выбрасываем ЗДЕСЬ, до сиблингов: иначе в «похожие темы» уехали
    # бы ссылки на страницы, которых не будет. Молчаливое усечение запрещено — считаем и
    # печатаем, см. `no_addr` ниже.
    # ⛔ ОТСЕВ ОДИН И ДО СИБЛИНГОВ. Причин выпасть у страницы две — нет адреса и метка не
    # перевелась, — и решать надо ОДНИМ местом. Раньше проверка кириллицы стояла НИЖЕ,
    # внутри цикла записи: страница не писалась, а в «похожие темы» ссылка на неё уже
    # уехала. Замер 08.08: 118 битых ссылок в 13 языках.
    def _skip(v):
        if not addr(v, "zadacha"):
            return "без адреса"
        if lang != "ru" and re.search("[а-яёА-ЯЁ]", v["zadacha"]):
            return "метка не перевелась"
        return None

    dropped = [r for r in (_skip(v) for v in fviews) if r]
    if dropped:
        from collections import Counter as _C

        print(
            f"{geo}/{lang}: пропущено видов {len(dropped)} — "
            + ", ".join(f"{k}: {n}" for k, n in _C(dropped).items()),
            flush=True,
        )
    fviews = [v for v in fviews if not _skip(v)]
    for v in fviews:
        s = addr(v, "zadacha")
        fact_sibs.append(
            {"tema": v["zadacha"], "slug": s, "url": f"/{lang}/{geo}/{s}/"}
        )
    for v in fviews:
        tema = v["zadacha"]
        s = addr(v, "zadacha")
        items = [it["text"] for it in v["items"]]
        vkey = view_theme(v, geo, lang)  # раздел разбора: и плитка, и довод CTA, и шлюз
        page = {
            "lang": lang,
            "path": f"/{lang}/{geo}/{s}/",
            # хвост адреса общий для всех языков? от этого зависят свитчер и hreflang
            "shared_tail": bool(v.get("key")),
            # ⭐ ШАГ 6: раздел на самой странице. Довод CTA выбирался по хешу пути, поэтому
            # на странице про сроки визы обещали «и официант не перепутает заказ». Ещё этот
            # ключ уезжает в шлюз клика, чтобы переход считался ПО РАЗДЕЛУ.
            "shelf_key": vkey,
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
                ctx={
                    "geo": geo,
                    "name": name,
                    "namep": namep,
                    "tl": tl,
                    "shelf_key": vkey,
                },
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
                "n": len(items),  # масса: счётчик раздела и порядок плиток внутри него
                "theme": vkey,
            }
        )

    # --- ВОПРОС-КОНТУР (хаб + темы под /<lang>/<geo>/q/) ---
    q_ok = False
    qgroups = [g for g in (ques or {}).get("groups", []) if len(g["questions"]) >= 4]
    if qgroups:
        q_ok = True
        qgroups = [g for g in qgroups if addr(g, "tema")]  # безадресные — не страницы
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

    # Разборы по разделам — для плиток НА СТРАНИЦЕ РАЗДЕЛА (ниже) и для плиток раздела на
    # хабе (в самом конце). Считаем ДО того, как `theme_tiles` разберёт карточки.
    by_theme = {}
    for c in fact_tiles:
        if c.get("theme"):
            by_theme.setdefault(c["theme"], []).append(c)
    theme_urls = {}  # ключ раздела → адрес его страницы; наполняется в шелф-контуре

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
            # ⭐ ШАГ 5: РАЗБОРЫ ЭТОГО РАЗДЕЛА — плитками СВЕРХУ страницы раздела. До этого
            # раздел показывал только свой хвост, а разборы висели плоским списком на хабе
            # страны (у `gr` 63 ссылки, у `any` 87). Своей страницы разделу не завожу —
            # это она и есть; новых адресов шаг не создаёт.
            tkey = sv.get("key") or SHELF_KEY.get(sv["shelf"]) or ""
            own = by_theme.get(tkey) or []
            theme_urls[tkey] = f"/{lang}/{geo}/s/{sk}/"  # куда ведёт плитка на хабе
            page = {
                "lang": lang,
                "path": f"/{lang}/{geo}/s/{sk}/",
                # хвост полки ОБЩИЙ по построению: ключ латинский, из таксономии, один во
                # всех языках. Признака не было — и 5616 страниц шли без hreflang.
                "shared_tail": True,
                "shelf_key": tkey,  # шаг 6: довод CTA и шлюз клика — по разделу
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
                    ctx={
                        "geo": geo,
                        "name": name,
                        "namep": namep,
                        "tl": tl,
                        "shelf_key": tkey,
                    },
                    lang=lang,
                    write_fn=write,
                )
                n = n_before + len(subtiles)
                # хаб полки: плитки веток + остаток (репы вне веток) аккордеоном внизу
                page["template"] = "index.html.j2"
                page["list_label"] = C["list_label_topics"]
                page["tiles"] = own + subtiles
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
            elif own:  # разборы сверху плитками, хвост раздела — ниже, как лежит
                page["template"] = "index.html.j2"
                page["list_label"] = C["list_label_topics"]
                page["tiles"] = own
                page["faqs_label"] = C["shelf_list_label"]
                if sv.get("groups"):
                    page["faqs"] = groups_to_faqs(sv, lang)
                else:  # хвост без дедупа: пунктами, чтобы заметки не пропали молча
                    page["questions"] = [it["text"] for it in sv["items"]]
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

    # --- ГЕО-ХАБ (плитки разделов + мостики вопросов и разделов) ---
    themed, loose = theme_tiles(fact_tiles, lang, geo, theme_urls)
    tiles = list(fact_tiles) if themed is None else themed + loose
    if themed is not None:
        print(
            f"{geo}/{lang}: хаб — {len(themed)} плиток разделов "
            f"на {len(fact_tiles)} адресов"
            + (f", карточками {len(loose)}" if loose else ""),
            flush=True,
        )
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
    # ⛔ НЕЧЕГО ПОКАЗАТЬ — НЕТ ХАБА (2026-08-12). Хаб писался безусловно, и гео с пустым
    # корпусом получало страницу из одной обвязки: 804 символа интро, CTA и подвала, ноль
    # ссылок внутрь. Замер по живому зеркалу: 33 таких хаба, у всех НОЛЬ видов и 1–8 мух на
    # всё гео (`al`, `ao`, `nl`, `eu`, `uk`, `ua` …); на `.online` те же гео отдавали 404
    # прямо из навигации, потому что главная их перечисляла, а пуш отсеивал.
    # ⚠️ `readycheck` этого поймать не мог: его правило «пустая» — текста меньше 400 символов,
    # а одна обвязка даёт 804. Порог ниже веса шаблона, то есть по пустоте гейт не срабатывал
    # НИКОГДА. Правило переносится сюда, к моменту сборки, где видно СОДЕРЖИМОЕ.
    if not tiles:
        print(f"{geo}/{lang}: пропущено — ни одной собранной страницы, хаб не пишем")
        return (0, 0, 0, 0)
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


def build_gateway(lang):
    """⭐ ШАГ 6: ШЛЮЗ КЛИКА `/<язык>/go/luky/` — чтобы переход в продукт был ПОСЧИТАН.

    Замер до шлюза: сколько людей уходит со страниц в Luky, мы не знали вовсе — ни одной
    цифры. Все двери (кнопка CTA и маркеры `#luky` в текстах) ведут теперь сюда, а сюда
    приходит `?geo=&shelf=`, то есть видно, какая страна и какой РАЗДЕЛ отдаёт переходы.

    ⛔ Своего бэкенда не заводим: nginx уже пишет строку запроса и Referer в access.log —
    счёт бесплатный. Страница только пересылает дальше.
    ⛔ `noindex`: это не контент, в карту сайта ей нельзя (её отсекает `_indexable`).
    """
    HA = HOME_ABOUT[lang]
    write(
        f"{lang}_go_luky.json",
        {
            "lang": lang,
            "template": "go.html.j2",
            "path": f"/{lang}/go/luky/",
            "noindex": True,
            "crumb_label": None,
            "title": HA["home_title"],
            "meta_desc": HA["home_desc"],
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
            if not n:
                continue  # гео без страниц не попадает ни в главную, ни в свитчер, ни в карту
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
        build_gateway(lang)  # шаг 6: без этой страницы все двери сайта отдают 404
        print(
            f"{lang}: home{'' if lang == 'ru' else ' + about'} + шлюз ({len(gl)} стран)"
        )
    print(f"ИТОГО data-страниц: {total} (дальше render.py --all)")
