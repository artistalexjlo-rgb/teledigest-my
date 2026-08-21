"""Сторож: поле страны бывает СПИСКОМ, и в текст эмбеддинга должны идти нормальные имена.

⛔ Повод (20.08, лог бота): `country_full_name_en: missing English name for ISO code 'de, ru'`
шло пачками каждую минуту. Функция искала весь список как ОДИН ISO-код, не находила и
подставляла заглушку — в Qdrant улетало «DE, RU» вместо «Germany, Russia».
"""

from __future__ import annotations

from teledigest.country_codes import country_full_name_en
from teledigest.embed_pump import _build_embed_text


def test_multi_country_becomes_names():
    """Список кодов → список имён, в том же порядке."""
    assert country_full_name_en("de, ru") == "Germany, Russia"
    assert country_full_name_en("tr,bg") == "Turkey, Bulgaria"
    assert country_full_name_en("eg, tr, ru") == "Egypt, Turkey, Russia"


def test_single_code_unchanged():
    """Одиночный код работает как прежде — правка не должна ломать обычный случай."""
    assert country_full_name_en("me") == "Montenegro"
    assert country_full_name_en("") == ""


def test_unknown_code_warns_by_code_not_by_list(caplog):
    """Неизвестный код всё ещё виден в логе, но ПО КОДУ: список в справочник не добавишь."""
    with caplog.at_level("WARNING"):
        assert country_full_name_en("xx, de") == "XX, Germany"
    assert "'xx'" in caplog.text, caplog.text
    assert "'xx, de'" not in caplog.text, caplog.text


def test_embedding_text_gets_real_names():
    """То, ради чего правка: в текст эмбеддинга уходят имена, а не заглушка."""
    txt = _build_embed_text("de, ru", "Bank account", "Finance", "Совет про счёт.")
    assert txt.startswith("Germany, Russia. "), txt
