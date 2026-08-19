"""Сторож семени «что знает мир»: адреса из Search Console невосстановимы.

⛔ Окно Search Console — 16 месяцев, и 448 показов на уже мёртвых адресах истекут сами.
Пока миграция адресов не сделана, это единственный источник ответа «куда обязан вести старый
адрес, чтобы человек попал на ОТВЕТ, а не в оглавление» (июльская ошибка: 983 из 992 правил
ведут в хаб страны). Поэтому выгрузка лежит в репо, а не в загрузках у юзера.
"""

import pathlib

SEED = pathlib.Path(__file__).resolve().parent.parent / "site" / "seed"
KNOWN = SEED / "known_urls.tsv"
REDIR = SEED / "redirects_2026_08_19.txt"


def _rows():
    out = []
    for line in KNOWN.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        out.append(line.split("\t"))
    return out


def test_seed_files_are_in_the_repo():
    assert KNOWN.exists(), f"нет выгрузки Search Console: {KNOWN}"
    assert REDIR.exists(), f"нет снимка нынешних правил редиректа: {REDIR}"


def test_known_urls_are_paths_with_demand():
    rows = _rows()
    assert len(rows) >= 188, f"адресов меньше снятого 19.08: {len(rows)}"
    for r in rows:
        assert len(r) == 3, f"строка не «путь<TAB>показы<TAB>клики»: {r}"
        path, imps, clicks = r
        assert path.startswith("/"), f"не путь, а что-то ещё: {path}"
        assert int(imps) >= 0 and int(clicks) >= 0, r
    # ⛔ Порядок = приоритет цели редиректа. Сбитый порядок молча обесценивает семя.
    imps = [int(r[1]) for r in rows]
    assert imps == sorted(imps, reverse=True), "семя не отсортировано по показам"


def test_redirect_snapshot_is_whole():
    lines = [x for x in REDIR.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) >= 992, f"снимок правил короче выложенного: {len(lines)}"
