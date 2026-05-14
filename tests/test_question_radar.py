from teledigest.question_radar import _message_link, is_translator_question


def test_positive_ru_recommend():
    assert is_translator_question("Посоветуйте хороший переводчик для Турции")


def test_positive_ru_question_mark():
    assert is_translator_question("Каким переводчиком пользуетесь?")


def test_positive_en():
    assert is_translator_question("Which translator app do you recommend?")


def test_negative_just_mention():
    assert not is_translator_question("Я пользуюсь Google переводчиком, очень удобно")


def test_negative_unrelated():
    assert not is_translator_question("Как дела?")


def test_negative_too_short():
    assert not is_translator_question("?")


def test_link_public():
    link = _message_link(-1001234567890, "balichat", 42)
    assert link == "https://t.me/balichat/42"


def test_link_private():
    link = _message_link(-1001631614451, "-1001631614451", 99)
    assert link == "https://t.me/c/1631614451/99"
