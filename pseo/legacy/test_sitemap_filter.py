"""SITEMAP = ТОЛЬКО ТО, ЧТО ЛЕЖИТ НА САЙТЕ.

Повод (2026-08-07): карта копировалась из `out/` целиком, а гейт отсекал часть гео ПОЗЖЕ.
Итог — 188 русских адресов задержанных гео стояли в sitemap, страниц по ним не было. Мы
своей же картой звали Google на несуществующее; а после появления настоящего 404 он бы
честно получал на них 404 и записывал нам «Not found» пачками.

Проверяем инвариант: адрес попадает в карту ⇔ рядом лежит файл. Это верно для ЛЮБОГО
языка — фильтр по файлам, а не по списку уехавших гео (гейт русский, языковые деревья едут
целиком, правила у них разные). Поэтому +10 языков тут ничего не меняют.

Ни сети, ни боевого репо: песочница. Запуск:
  python test_sitemap_filter.py
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ship  # noqa: E402

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://info.multyspeak.online/ru/aa/</loc>
    <lastmod>2026-01-15</lastmod>
  </url>
  <url>
    <loc>https://info.multyspeak.online/ru/bb/</loc>
    <lastmod>2026-02-20</lastmod>
  </url>
  <url>
    <loc>https://info.multyspeak.online/zz/aa/</loc>
    <lastmod>2026-03-25</lastmod>
  </url>
  <url>
    <loc>https://info.multyspeak.online/</loc>
  </url>
</urlset>
"""


def ok(cond, what, got=""):
    print("%-58s %-22s %s" % (what, got, "OK" if cond else "← ПРОВАЛ"))
    return cond


if __name__ == "__main__":
    out, repo = tempfile.mkdtemp(), tempfile.mkdtemp()
    ship.OUT, ship.PAGES_REPO = out, repo
    open(f"{out}/sitemap.xml", "w", encoding="utf-8").write(SITEMAP)
    # На «сайте» есть только ru/aa, язык zz/aa и корень. ru/bb не уехало.
    for p in ("ru/aa", "zz/aa"):
        os.makedirs(f"{repo}/{p}", exist_ok=True)
        open(f"{repo}/{p}/index.html", "w").close()
    open(f"{repo}/index.html", "w").close()

    ship._write_filtered_sitemap()
    got = open(f"{repo}/sitemap.xml", encoding="utf-8").read()
    locs = re.findall(r"<loc>([^<]+)</loc>", got)
    good = True

    good &= ok(
        any(x.endswith("/ru/aa/") for x in locs),
        "1. страница есть на сайте → в карте",
        str(len(locs)) + " адресов",
    )
    good &= ok(
        not any(x.endswith("/ru/bb/") for x in locs),
        "2. ⭐ страницы НЕТ на сайте → из карты выкинута",
        (
            "ru/bb отсутствует"
            if not any("bb" in x for x in locs)
            else "ru/bb просочилось"
        ),
    )
    good &= ok(
        any(x.endswith("/zz/aa/") for x in locs),
        "3. фильтр языко-независим (zz прошёл как ru)",
        "zz/aa в карте",
    )
    good &= ok(
        locs and locs[-1].rstrip("/").endswith("multyspeak.online"),
        "4. корень '/' сопоставлен с index.html",
        "корень в карте" if any(x.count("/") == 3 for x in locs) else str(locs),
    )
    good &= ok(
        "<lastmod>2026-01-15</lastmod>" in got,
        "5. дата страницы сохранена, не перезаписана",
        "2026-01-15 на месте",
    )
    good &= ok(
        got.startswith("<?xml") and got.rstrip().endswith("</urlset>"),
        "6. карта осталась валидным XML",
    )

    print("\nVERDICT:", "OK — карта не врёт про состав сайта" if good else "FAIL")
    sys.exit(0 if good else 1)
