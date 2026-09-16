#!/usr/bin/env bash
# Окно на BuyVM для звена 8 (PLAN.md §3.2, шаг 3, 12.09). Запускать root'ом на VPS; идемпотентно —
# повторный запуск только перезаписывает site.conf и делает reload (16.09: location помощника).
# Dokploy не трогается, редеплоя нет: маунт site/online → /usr/share/nginx/html остаётся,
# версии и симлинк current живут ВНУТРИ online/. Идемпотентно: если current уже есть — пропуск.
set -euo pipefail
cd /root/pseo_builder/site
C=$(docker ps --format "{{.Names}}" | grep pseosite | head -1)
[ -n "$C" ] || { echo "контейнер pseosite не найден"; exit 1; }

echo "=== 1. старый снимок -> online/online_v0 (inode папки online не меняется) ==="
if [ ! -e online/current ]; then
  mkdir -p online/.v0_tmp
  find online -mindepth 1 -maxdepth 1 ! -name .v0_tmp -exec mv -t online/.v0_tmp/ {} +
  mv online/.v0_tmp online/online_v0
  ln -s online_v0 online/current
fi
echo "current -> $(readlink online/current)"

echo "=== 2. nginx-конфиг: root current, без карты редиректов (решение 20.08) ==="
cp -n nginxconf/site.conf "nginxconf/site.conf.bak_$(date +%s)" || true
cat > nginxconf/site.conf <<'NGX'
# Живой сайт = симлинк current внутри примонтированной папки (PLAN.md §3.2, шаг 3, 12.09).
# Публикация переставляет current; nginx подхватывает без рестарта. Редиректов со старых
# адресов НЕТ (решение юзера 20.08: дерево меняется целиком).
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html/current;
    index index.html;
    absolute_redirect off;

    gzip on;
    gzip_comp_level 5;
    gzip_min_length 512;
    gzip_types text/css application/javascript application/xml image/svg+xml;

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ $uri/index.html =404;
    }

    # Помощник Luky на страницах (16.09): виджет ходит на СВОЙ домен, nginx проксирует
    # в приложение Luky по dokploy-network (имя сервиса стабильно, порт приложения 3000).
    # Host подменяем на домен продукта — приложение различает .online/.ru по нему.
    location /api/assistant/ {
        proxy_pass http://bots-luky3-muoa7b:3000/api/assistant/;
        proxy_set_header Host multyspeak.online;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_read_timeout 90s;
        client_max_body_size 64k;
    }

    error_page 404 /404.html;
    location = /404.html {
        internal;
    }
}
NGX

echo "=== 3. проверка конфига и reload ==="
docker exec "$C" nginx -t
docker exec "$C" nginx -s reload
sleep 1
echo "=== 4. отдаёт? ==="
curl -s -o /dev/null -w "https://info.multyspeak.online/ru/ -> %{http_code}\n" https://info.multyspeak.online/ru/
