#!/usr/bin/env bash
# deploy.sh — деплой «по вызову» с ДЕСКТОПА (push-кред живёт только здесь, VPS безкредовый).
# Детерминированно, без Claude-интеллекта: забрать собранное с VPS → навигация → рендер → push.
# Запуск: bash pseo/deploy.sh   (из корня репо _Grab)
set -euo pipefail

VPS=root@199.195.252.114
PSEO="$(cd "$(dirname "$0")" && pwd)"      # .../pseo
PAGES="$PSEO/../../multyspeak-pages"

echo "== 1. забрать собранные страницы с VPS =="
for f in $(ssh -o ConnectTimeout=25 "$VPS" 'ls /root/pseo_builder/out/*.json 2>/dev/null | xargs -n1 basename'); do
  # <geo>_<slug>.json → data/auto_<geo>_<slug>.json
  ssh -o ConnectTimeout=20 "$VPS" "cat /root/pseo_builder/out/$f" > "$PSEO/data/auto_${f}"
done
echo "   страниц в data/: $(ls "$PSEO"/data/auto_*.json 2>/dev/null | wc -l)"

echo "== 2. матрица спроса (для data-backed СКОРО) =="
python "$PSEO/demand.py"

echo "== 3. навигация из манифеста (хабы/главная/чипы) =="
python "$PSEO/wire.py"

echo "== 4. рендер всего + sitemap/robots =="
rm -rf "$PSEO/out"
python "$PSEO/render.py" --all "$(date -u +%Y-%m-%d)"

echo "== 5. в репо pages =="
cp -r "$PSEO/out/ru/"* "$PAGES/ru/"
cp "$PSEO/out/sitemap.xml" "$PSEO/out/robots.txt" "$PAGES/"

echo "== 6. коммит + push =="
cd "$PAGES"
if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "   нет изменений — деплоить нечего."
  exit 0
fi
git add -A
git commit -q -m "pSEO deploy: $(ls "$PSEO"/data/auto_*.json | wc -l) auto pages ($(date -u +%F))

Automated deploy via pseo/deploy.sh (build on VPS -> pull -> wire -> render -> push).
Push credential stays on desktop only; VPS holds none.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin main
echo "== ГОТОВО =="
