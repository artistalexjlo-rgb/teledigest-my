"""Конфиг для зеркала .ru (info.multyspeak.ru, РФ/Яндекс-аудитория, PLAN.md §3.1/§3.2).

⛔ Домен и CTA у `.ru` СВОИ (проверено фактом: `templates/base.html.j2` вшивает `site.domain`
прямо в canonical/hreflang каждой страницы — одно дерево на оба домена не годится). Всё
остальное — бренд, языки, telegram, год, draft — берём из `config/site.py` как ОДИН
источник: два домена одного продукта, а не два разных конфига, которые могут разъехаться.

CTA — юзер 30.08: multyspeak.ru, а не multyspeak.online (свой вход для РФ-аудитории).
"""

from config.site import SITE as _BASE

SITE = {
    **_BASE,
    "domain": "https://info.multyspeak.ru",
    "cta_luky_url": "https://multyspeak.ru",
}
