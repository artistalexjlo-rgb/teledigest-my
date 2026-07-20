"""Единая точка для всех внешних ссылок / доменов / языко-нейтральных
бренд-констант. Меняешь здесь одно значение → re-render → обновляется на
ВСЕХ страницах сразу (чистый template-fill, без Gemini/квот).

Переводимые лейблы — НЕ здесь, а в i18n/{lang}.json.

PSEO_DOMAIN (env) — оверрайд домена для зеркал: рендер зеркала (ship --mirror) шьёт
canonical/sitemap/hreflang на СВОЙ домен (info.multyspeak.ru), иначе Яндекс видит
канониклы на чужой .online и зеркало не индексирует."""

import os

SITE = {
    # Бренд-имя = Luky (звучное имя). MultySpeak = суть/смысл (мульти-язык, любая
    # страна) — живёт в ДОМЕНЕ, не как второе видимое имя на странице.
    "brand": "Luky",
    "domain": os.environ.get(
        "PSEO_DOMAIN", "https://info.multyspeak.online"
    ),  # сабдомен pSEO-портала; env — для рендера зеркала
    "mirror_domain": "https://info.multyspeak.ru",  # зеркало (RU без VPN)
    # CTA-дверь в Luky. База продукта = .online (.ru — локализация для РФ, редиректит
    # на .online). Интерим: продуктовый лендинг. TODO: диплинк в комнату в РЕЖИМЕ
    # ПОМОЩНИКА, когда появится URL-параметр режима (см. model_luky_funnel_cta).
    "cta_luky_url": "https://multyspeak.online",
    "telegram_url": "https://t.me/luky_channel",  # footer: канал + чат
    "languages": ["ru", "en", "es", "pt"],
    "year": 2026,
    # draft=True → на всех страницах <meta robots noindex>. Пока строим — True.
    # На лаунче (реальное наполнение прошло гейт) → False + sitemap + GSC.
    "draft": False,
}
