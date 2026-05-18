"""
country_codes.py — ISO 3166-1 alpha-2 country codes with Russian names.
Single source of truth for country code resolution.
"""

import logging

log = logging.getLogger("teledigest")

# code -> (russian_name, flag_emoji)
COUNTRIES: dict[str, tuple[str, str]] = {
    "af": ("Афганистан", "🇦🇫"),
    "al": ("Албания", "🇦🇱"),
    "dz": ("Алжир", "🇩🇿"),
    "ad": ("Андорра", "🇦🇩"),
    "ao": ("Ангола", "🇦🇴"),
    "ag": ("Антигуа и Барбуда", "🇦🇬"),
    "ar": ("Аргентина", "🇦🇷"),
    "am": ("Армения", "🇦🇲"),
    "au": ("Австралия", "🇦🇺"),
    "at": ("Австрия", "🇦🇹"),
    "az": ("Азербайджан", "🇦🇿"),
    "bs": ("Багамы", "🇧🇸"),
    "bh": ("Бахрейн", "🇧🇭"),
    "bd": ("Бангладеш", "🇧🇩"),
    "bb": ("Барбадос", "🇧🇧"),
    "by": ("Беларусь", "🇧🇾"),
    "be": ("Бельгия", "🇧🇪"),
    "bz": ("Белиз", "🇧🇿"),
    "bj": ("Бенин", "🇧🇯"),
    "bt": ("Бутан", "🇧🇹"),
    "bo": ("Боливия", "🇧🇴"),
    "ba": ("Босния и Герцеговина", "🇧🇦"),
    "bw": ("Ботсвана", "🇧🇼"),
    "br": ("Бразилия", "🇧🇷"),
    "bn": ("Бруней", "🇧🇳"),
    "bg": ("Болгария", "🇧🇬"),
    "bf": ("Буркина-Фасо", "🇧🇫"),
    "bi": ("Бурунди", "🇧🇮"),
    "cv": ("Кабо-Верде", "🇨🇻"),
    "kh": ("Камбоджа", "🇰🇭"),
    "cm": ("Камерун", "🇨🇲"),
    "ca": ("Канада", "🇨🇦"),
    "cf": ("ЦАР", "🇨🇫"),
    "td": ("Чад", "🇹🇩"),
    "cl": ("Чили", "🇨🇱"),
    "cn": ("Китай", "🇨🇳"),
    "co": ("Колумбия", "🇨🇴"),
    "km": ("Коморы", "🇰🇲"),
    "cg": ("Конго", "🇨🇬"),
    "cr": ("Коста-Рика", "🇨🇷"),
    "hr": ("Хорватия", "🇭🇷"),
    "cu": ("Куба", "🇨🇺"),
    "cy": ("Кипр", "🇨🇾"),
    "cz": ("Чехия", "🇨🇿"),
    "dk": ("Дания", "🇩🇰"),
    "dj": ("Джибути", "🇩🇯"),
    "dm": ("Доминика", "🇩🇲"),
    "do": ("Доминикана", "🇩🇴"),
    "ec": ("Эквадор", "🇪🇨"),
    "eg": ("Египет", "🇪🇬"),
    "sv": ("Сальвадор", "🇸🇻"),
    "ee": ("Эстония", "🇪🇪"),
    "et": ("Эфиопия", "🇪🇹"),
    "fi": ("Финляндия", "🇫🇮"),
    "fr": ("Франция", "🇫🇷"),
    "ga": ("Габон", "🇬🇦"),
    "gm": ("Гамбия", "🇬🇲"),
    "ge": ("Грузия", "🇬🇪"),
    "de": ("Германия", "🇩🇪"),
    "gh": ("Гана", "🇬🇭"),
    "gr": ("Греция", "🇬🇷"),
    "gd": ("Гренада", "🇬🇩"),
    "gt": ("Гватемала", "🇬🇹"),
    "gn": ("Гвинея", "🇬🇳"),
    "gy": ("Гайана", "🇬🇾"),
    "ht": ("Гаити", "🇭🇹"),
    "hn": ("Гондурас", "🇭🇳"),
    "hu": ("Венгрия", "🇭🇺"),
    "is": ("Исландия", "🇮🇸"),
    "in": ("Индия", "🇮🇳"),
    "id": ("Индонезия", "🇮🇩"),
    "ir": ("Иран", "🇮🇷"),
    "iq": ("Ирак", "🇮🇶"),
    "ie": ("Ирландия", "🇮🇪"),
    "il": ("Израиль", "🇮🇱"),
    "it": ("Италия", "🇮🇹"),
    "jm": ("Ямайка", "🇯🇲"),
    "jp": ("Япония", "🇯🇵"),
    "jo": ("Иордания", "🇯🇴"),
    "kz": ("Казахстан", "🇰🇿"),
    "ke": ("Кения", "🇰🇪"),
    "kg": ("Кыргызстан", "🇰🇬"),
    "la": ("Лаос", "🇱🇦"),
    "lv": ("Латвия", "🇱🇻"),
    "lb": ("Ливан", "🇱🇧"),
    "ls": ("Лесото", "🇱🇸"),
    "lr": ("Либерия", "🇱🇷"),
    "ly": ("Ливия", "🇱🇾"),
    "li": ("Лихтенштейн", "🇱🇮"),
    "lt": ("Литва", "🇱🇹"),
    "lu": ("Люксембург", "🇱🇺"),
    "mg": ("Мадагаскар", "🇲🇬"),
    "mw": ("Малави", "🇲🇼"),
    "my": ("Малайзия", "🇲🇾"),
    "mv": ("Мальдивы", "🇲🇻"),
    "ml": ("Мали", "🇲🇱"),
    "mt": ("Мальта", "🇲🇹"),
    "mu": ("Маврикий", "🇲🇺"),
    "mx": ("Мексика", "🇲🇽"),
    "md": ("Молдова", "🇲🇩"),
    "mc": ("Монако", "🇲🇨"),
    "mn": ("Монголия", "🇲🇳"),
    "me": ("Черногория", "🇲🇪"),
    "ma": ("Марокко", "🇲🇦"),
    "mz": ("Мозамбик", "🇲🇿"),
    "mm": ("Мьянма", "🇲🇲"),
    "na": ("Намибия", "🇳🇦"),
    "np": ("Непал", "🇳🇵"),
    "nl": ("Нидерланды", "🇳🇱"),
    "nz": ("Новая Зеландия", "🇳🇿"),
    "ni": ("Никарагуа", "🇳🇮"),
    "ne": ("Нигер", "🇳🇪"),
    "ng": ("Нигерия", "🇳🇬"),
    "mk": ("Северная Македония", "🇲🇰"),
    "no": ("Норвегия", "🇳🇴"),
    "om": ("Оман", "🇴🇲"),
    "pk": ("Пакистан", "🇵🇰"),
    "pa": ("Панама", "🇵🇦"),
    "pg": ("Папуа Новая Гвинея", "🇵🇬"),
    "py": ("Парагвай", "🇵🇾"),
    "pe": ("Перу", "🇵🇪"),
    "ph": ("Филиппины", "🇵🇭"),
    "pl": ("Польша", "🇵🇱"),
    "pt": ("Португалия", "🇵🇹"),
    "qa": ("Катар", "🇶🇦"),
    "ro": ("Румыния", "🇷🇴"),
    "rw": ("Руанда", "🇷🇼"),
    "sa": ("Саудовская Аравия", "🇸🇦"),
    "sn": ("Сенегал", "🇸🇳"),
    "rs": ("Сербия", "🇷🇸"),
    "sc": ("Сейшелы", "🇸🇨"),
    "sl": ("Сьерра-Леоне", "🇸🇱"),
    "sg": ("Сингапур", "🇸🇬"),
    "sk": ("Словакия", "🇸🇰"),
    "si": ("Словения", "🇸🇮"),
    "so": ("Сомали", "🇸🇴"),
    "za": ("ЮАР", "🇿🇦"),
    "ss": ("Южный Судан", "🇸🇸"),
    "es": ("Испания", "🇪🇸"),
    "lk": ("Шри-Ланка", "🇱🇰"),
    "sd": ("Судан", "🇸🇩"),
    "sr": ("Суринам", "🇸🇷"),
    "se": ("Швеция", "🇸🇪"),
    "ch": ("Швейцария", "🇨🇭"),
    "sy": ("Сирия", "🇸🇾"),
    "tw": ("Тайвань", "🇹🇼"),
    "tj": ("Таджикистан", "🇹🇯"),
    "tz": ("Танзания", "🇹🇿"),
    "th": ("Таиланд", "🇹🇭"),
    "tl": ("Тимор-Лесте", "🇹🇱"),
    "tg": ("Того", "🇹🇬"),
    "tt": ("Тринидад и Тобаго", "🇹🇹"),
    "tn": ("Тунис", "🇹🇳"),
    "tr": ("Турция", "🇹🇷"),
    "tm": ("Туркменистан", "🇹🇲"),
    "ug": ("Уганда", "🇺🇬"),
    "ua": ("Украина", "🇺🇦"),
    "ae": ("ОАЭ", "🇦🇪"),
    "gb": ("Великобритания", "🇬🇧"),
    "us": ("США", "🇺🇸"),
    "uy": ("Уругвай", "🇺🇾"),
    "uz": ("Узбекистан", "🇺🇿"),
    "ve": ("Венесуэла", "🇻🇪"),
    "vn": ("Вьетнам", "🇻🇳"),
    "ye": ("Йемен", "🇾🇪"),
    "zm": ("Замбия", "🇿🇲"),
    "zw": ("Зимбабве", "🇿🇼"),
    # --- Additions to complete ISO 3166-1 alpha-2 coverage ---
    # Major omissions first
    "ru": ("Россия", "🇷🇺"),
    "kr": ("Республика Корея", "🇰🇷"),
    "kp": ("КНДР", "🇰🇵"),
    "hk": ("Гонконг", "🇭🇰"),
    "mo": ("Макао", "🇲🇴"),
    "ps": ("Палестина", "🇵🇸"),
    "xk": ("Косово", "🇽🇰"),  # de-facto, not official ISO
    "fj": ("Фиджи", "🇫🇯"),
    "kw": ("Кувейт", "🇰🇼"),
    "mr": ("Мавритания", "🇲🇷"),
    "gl": ("Гренландия", "🇬🇱"),
    "pr": ("Пуэрто-Рико", "🇵🇷"),
    # Africa missing
    "ci": ("Кот-д'Ивуар", "🇨🇮"),
    "cd": ("ДР Конго", "🇨🇩"),
    "er": ("Эритрея", "🇪🇷"),
    "gq": ("Экваториальная Гвинея", "🇬🇶"),
    "gw": ("Гвинея-Бисау", "🇬🇼"),
    "sz": ("Эсватини", "🇸🇿"),
    "st": ("Сан-Томе и Принсипи", "🇸🇹"),
    # Asia / Pacific
    "ki": ("Кирибати", "🇰🇮"),
    "mh": ("Маршалловы Острова", "🇲🇭"),
    "fm": ("Микронезия", "🇫🇲"),
    "nr": ("Науру", "🇳🇷"),
    "pw": ("Палау", "🇵🇼"),
    "ws": ("Самоа", "🇼🇸"),
    "sb": ("Соломоновы Острова", "🇸🇧"),
    "to": ("Тонга", "🇹🇴"),
    "tv": ("Тувалу", "🇹🇻"),
    "vu": ("Вануату", "🇻🇺"),
    # Americas
    "kn": ("Сент-Китс и Невис", "🇰🇳"),
    "lc": ("Сент-Люсия", "🇱🇨"),
    "vc": ("Сент-Винсент и Гренадины", "🇻🇨"),
    # Europe / micro states
    "sm": ("Сан-Марино", "🇸🇲"),
    "va": ("Ватикан", "🇻🇦"),
    "ax": ("Аландские острова", "🇦🇽"),
    "gi": ("Гибралтар", "🇬🇮"),
    "gg": ("Гернси", "🇬🇬"),
    "je": ("Джерси", "🇯🇪"),
    "im": ("Остров Мэн", "🇮🇲"),
    "fo": ("Фарерские острова", "🇫🇴"),
    "sj": ("Шпицберген и Ян-Майен", "🇸🇯"),
    # Caribbean / Atlantic territories
    "aw": ("Аруба", "🇦🇼"),
    "bq": ("Карибские Нидерланды", "🇧🇶"),
    "cw": ("Кюрасао", "🇨🇼"),
    "sx": ("Синт-Мартен", "🇸🇽"),
    "mf": ("Сен-Мартен", "🇲🇫"),
    "bl": ("Сен-Бартелеми", "🇧🇱"),
    "ai": ("Ангилья", "🇦🇮"),
    "bm": ("Бермуды", "🇧🇲"),
    "ky": ("Каймановы острова", "🇰🇾"),
    "ms": ("Монтсеррат", "🇲🇸"),
    "tc": ("Тёркс и Кайкос", "🇹🇨"),
    "vg": ("Британские Виргинские острова", "🇻🇬"),
    "vi": ("Виргинские острова США", "🇻🇮"),
    # Pacific territories
    "as": ("Американское Самоа", "🇦🇸"),
    "gu": ("Гуам", "🇬🇺"),
    "mp": ("Северные Марианские острова", "🇲🇵"),
    "nc": ("Новая Каледония", "🇳🇨"),
    "pf": ("Французская Полинезия", "🇵🇫"),
    "ck": ("Острова Кука", "🇨🇰"),
    "nu": ("Ниуэ", "🇳🇺"),
    "tk": ("Токелау", "🇹🇰"),
    "wf": ("Уоллис и Футуна", "🇼🇫"),
    "fk": ("Фолклендские острова", "🇫🇰"),
    # French overseas
    "gf": ("Французская Гвиана", "🇬🇫"),
    "gp": ("Гваделупа", "🇬🇵"),
    "mq": ("Мартиника", "🇲🇶"),
    "re": ("Реюньон", "🇷🇪"),
    "yt": ("Майотта", "🇾🇹"),
    "pm": ("Сен-Пьер и Микелон", "🇵🇲"),
    "tf": ("Французские Южные территории", "🇹🇫"),
    # Uninhabited / remote
    "aq": ("Антарктида", "🇦🇶"),
    "bv": ("Остров Буве", "🇧🇻"),
    "cc": ("Кокосовые острова", "🇨🇨"),
    "cx": ("Остров Рождества", "🇨🇽"),
    "hm": ("Остров Херд и острова Макдональд", "🇭🇲"),
    "io": ("Британская территория в Индийском океане", "🇮🇴"),
    "nf": ("Остров Норфолк", "🇳🇫"),
    "pn": ("Острова Питкэрн", "🇵🇳"),
    "sh": ("Остров Святой Елены", "🇸🇭"),
    "gs": ("Южная Георгия", "🇬🇸"),
    "um": ("Внешние малые острова США", "🇺🇲"),
}

# Russian name (lowercase) -> code
_NAME_TO_CODE: dict[str, str] = {v[0].lower(): k for k, v in COUNTRIES.items()}

# Variants with spaces/dashes normalized
_NORMALIZED: dict[str, str] = {}
for name, code in _NAME_TO_CODE.items():
    _NORMALIZED[name] = code
    _NORMALIZED[name.replace("-", " ")] = code
    _NORMALIZED[name.replace(" ", "-")] = code
    _NORMALIZED[name.replace("-", "").replace(" ", "")] = code


def resolve_country(text: str) -> tuple[str, str, str] | None:
    """
    Resolve user input to (code, russian_name, flag).

    Accepts: Russian name, ISO code, partial match.
    Returns None if not recognized.
    """
    text = text.strip().lower()

    # Direct ISO code
    if text in COUNTRIES:
        name, flag = COUNTRIES[text]
        return text, name, flag

    # Full or normalized name
    if text in _NORMALIZED:
        code = _NORMALIZED[text]
        name, flag = COUNTRIES[code]
        return code, name, flag

    # Prefix match on Russian names and ISO codes
    candidates = {v for k, v in _NORMALIZED.items() if k.startswith(text)}
    candidates |= {k for k in COUNTRIES if k.startswith(text)}
    if len(candidates) == 1:
        code = candidates.pop()
        name, flag = COUNTRIES[code]
        return code, name, flag

    return None


def display_name(code: str) -> str:
    """Return 'flag RussianName' for a country code."""
    if code in COUNTRIES:
        name, flag = COUNTRIES[code]
        return f"{flag} {name}"
    return code.upper()


# ---------------------------------------------------------------------------
# English country names — for embedding text and other places where we need
# a stable English label (Apps Script, wiki import, migration). Single source
# of truth: import from here, do NOT redefine locally.
#
# SYNC WITH apps_script/Code.gs::COUNTRY_NAMES — keep the two lists identical.
# When you add a row here, mirror it there.
# ---------------------------------------------------------------------------
COUNTRY_NAMES_EN: dict[str, str] = {
    # Tier 0 — active user chats
    "ar": "Argentina",
    "at": "Austria",
    "be": "Belgium",
    "bg": "Bulgaria",
    "br": "Brazil",
    "de": "Germany",
    "fr": "France",
    "id": "Indonesia",
    "lk": "Sri Lanka",
    "mu": "Mauritius",
    "ph": "Philippines",
    "th": "Thailand",
    "tr": "Turkey",
    "vn": "Vietnam",
    # Tier 1 — popular expat/digital-nomad destinations
    "ae": "United Arab Emirates",
    "am": "Armenia",
    "az": "Azerbaijan",
    "ba": "Bosnia and Herzegovina",
    "by": "Belarus",
    "ca": "Canada",
    "cl": "Chile",
    "cn": "China",
    "co": "Colombia",
    "cr": "Costa Rica",
    "cy": "Cyprus",
    "cz": "Czech Republic",
    "dk": "Denmark",
    "ec": "Ecuador",
    "ee": "Estonia",
    "eg": "Egypt",
    "es": "Spain",
    "fi": "Finland",
    "gb": "United Kingdom",
    "ge": "Georgia",
    "gr": "Greece",
    "hr": "Croatia",
    "hu": "Hungary",
    "ie": "Ireland",
    "il": "Israel",
    "in": "India",
    "it": "Italy",
    "jo": "Jordan",
    "jp": "Japan",
    "ke": "Kenya",
    "kg": "Kyrgyzstan",
    "kh": "Cambodia",
    "kr": "South Korea",
    "kz": "Kazakhstan",
    "la": "Laos",
    "lb": "Lebanon",
    "lt": "Lithuania",
    "lv": "Latvia",
    "ma": "Morocco",
    "md": "Moldova",
    "me": "Montenegro",
    "mk": "North Macedonia",
    "mm": "Myanmar",
    "mn": "Mongolia",
    "mx": "Mexico",
    "my": "Malaysia",
    "nl": "Netherlands",
    "no": "Norway",
    "np": "Nepal",
    "nz": "New Zealand",
    "pe": "Peru",
    "pk": "Pakistan",
    "pl": "Poland",
    "pt": "Portugal",
    "py": "Paraguay",
    "ro": "Romania",
    "rs": "Serbia",
    "ru": "Russia",
    "sa": "Saudi Arabia",
    "se": "Sweden",
    "sg": "Singapore",
    "si": "Slovenia",
    "sk": "Slovakia",
    "tn": "Tunisia",
    "tw": "Taiwan",
    "ua": "Ukraine",
    "us": "United States of America",
    "uy": "Uruguay",
    "uz": "Uzbekistan",
    "za": "South Africa",
    # Tier 2 — extended coverage
    "bd": "Bangladesh",
    "bo": "Bolivia",
    "cd": "Democratic Republic of Congo",
    "ci": "Ivory Coast",
    "cm": "Cameroon",
    "cu": "Cuba",
    "do": "Dominican Republic",
    "dz": "Algeria",
    "et": "Ethiopia",
    "gh": "Ghana",
    "gt": "Guatemala",
    "hn": "Honduras",
    "ht": "Haiti",
    "li": "Liechtenstein",
    "lu": "Luxembourg",
    "ly": "Libya",
    "mg": "Madagascar",
    "ml": "Mali",
    "mw": "Malawi",
    "mz": "Mozambique",
    "na": "Namibia",
    "ng": "Nigeria",
    "ni": "Nicaragua",
    "om": "Oman",
    "pa": "Panama",
    "qa": "Qatar",
    "rw": "Rwanda",
    "sd": "Sudan",
    "sn": "Senegal",
    "sv": "El Salvador",
    "sy": "Syria",
    "tz": "Tanzania",
    "ug": "Uganda",
    "ve": "Venezuela",
    "xk": "Kosovo",
    "ye": "Yemen",
    "zm": "Zambia",
    "zw": "Zimbabwe",
}


def country_full_name_en(code: str) -> str:
    """Return English country name for an ISO code, or uppercase ISO if missing.

    Use this everywhere we build embedding text or any English label —
    keeps `wisdom_base`/`wikivoyage_base` consistent across writers
    (Apps Script, wiki import, migration).
    """
    c = (code or "").lower()
    name = COUNTRY_NAMES_EN.get(c)
    if name:
        return name
    # Fall back to uppercase ISO and log so missing countries surface.
    if c:
        log.warning(
            "country_full_name_en: missing English name for ISO code %r — "
            "falling back to uppercase. Add it to COUNTRY_NAMES_EN.",
            c,
        )
    return c.upper()
