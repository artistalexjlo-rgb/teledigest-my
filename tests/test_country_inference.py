"""Tests for country_inference."""

from __future__ import annotations

import pytest

from teledigest.country_inference import infer_country, infer_country_with_log


# --- Brazilian terms ---------------------------------------------------------

@pytest.mark.parametrize("query", [
    "как получить CPF",
    "Cpf для иностранца",
    "получить цпф",
    "СПФ оформление",
    "что такое PIX",
    "перевод через пикс",
    "ifood доставка",
    "поехал в Сан-Паулу",
    "rio de janeiro карнавал",
    "Curitiba стоит ли ехать",
    "куда сходить в Curitiba",
])
def test_infer_br(query):
    assert infer_country(query) == "br"


# --- Argentinian terms -------------------------------------------------------

@pytest.mark.parametrize("query", [
    "оформление CUIT",
    "куит как получить",
    "что в Буэнос-Айресе делать",
    "buenos aires или mendoza?",
    "поехать в Барилоче",
    "цены в Mercado Pago",
])
def test_infer_ar(query):
    assert infer_country(query) == "ar"


# --- Indonesian terms --------------------------------------------------------

@pytest.mark.parametrize("query", [
    "kitas процесс",
    "получить китас",
    "что на Bali смотреть",
    "Ubud где жить",
    "Canggu или Сeminyak",
    "Telkomsel SIM",
])
def test_infer_id(query):
    assert infer_country(query) == "id"


# --- Vietnamese terms --------------------------------------------------------

@pytest.mark.parametrize("query", [
    "Zalo не работает",
    "перевод Vietcombank",
    "Нячанг сейчас как?",
    "Phu Quoc отдых",
    "из Hanoi в Ho Chi Minh",
])
def test_infer_vn(query):
    assert infer_country(query) == "vn"


# --- Sri Lanka --------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "виза eta срок",  # eta is in lk terms
    "Galle стоит?",
    "Mirissa и Hikkaduwa",
    "из Коломбо куда?",
])
def test_infer_lk(query):
    assert infer_country(query) == "lk"


# --- Mauritius --------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "MUR курс",
    "Flic-en-Flac пляжи",
    "Casela парк",
    "MCB банк",
])
def test_infer_mu(query):
    assert infer_country(query) == "mu"


# --- No match -> None --------------------------------------------------------

@pytest.mark.parametrize("query", [
    "виза какая нужна",
    "паспорт",
    "что нужно для въезда",
    "",
    "   ",
    "просто общий вопрос",
])
def test_infer_no_match(query):
    assert infer_country(query) is None


# --- Override logic ----------------------------------------------------------

def test_override_when_term_differs_from_chat():
    """Chat says AR, query says CPF (BR) -> override to BR."""
    final, overridden = infer_country_with_log("как получить CPF", chat_country="ar")
    assert final == "br"
    assert overridden is True


def test_no_override_when_chat_and_query_agree():
    """Chat says BR, query has CPF (BR) -> keep BR, no override marker."""
    final, overridden = infer_country_with_log("CPF вопрос", chat_country="br")
    assert final == "br"
    assert overridden is False


def test_no_override_when_query_has_no_terms():
    """Chat says AR, query has no specific terms -> stay with AR."""
    final, overridden = infer_country_with_log("какой совет", chat_country="ar")
    assert final == "ar"
    assert overridden is False


# --- Case sensitivity --------------------------------------------------------

def test_case_insensitive():
    assert infer_country("CPF") == "br"
    assert infer_country("cpf") == "br"
    assert infer_country("Cpf") == "br"
    assert infer_country("СПФ") == "br"
    assert infer_country("спф") == "br"


# --- Word boundary -----------------------------------------------------------

def test_word_boundary_avoids_substring_matches():
    """'sus' is BR (SUS healthcare), 'suspended' should NOT match."""
    # 'suspended account' must NOT trigger BR
    assert infer_country("My suspended account") is None
    # 'CUIT' must NOT match inside 'circuit'
    assert infer_country("design a new circuit") is None


# --- Conflict resolution -----------------------------------------------------

def test_more_terms_wins():
    """Query with both CPF (br) and Mendoza (ar) — terms count decides."""
    # 1 br + 1 ar = tie; first-found wins. Verify deterministically: more BR.
    final = infer_country("CPF, PIX и SUS — это всё про Mendoza?")
    assert final == "br"  # 3 br terms vs 1 ar term
