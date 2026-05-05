#!/bin/bash
# vps-backup.sh — Daily VPS backup → Telegram document.
#
# Read token+chat from /etc/vps-backup.env, tar everything important,
# split into ≤45MB parts if needed (Telegram bot API limit ~50MB),
# send each part as document, rotate local cache to keep last 7 days.
#
# Run via systemd timer (see vps-backup.timer / .service).

set -euo pipefail

# --- 0. Config ---
ENV_FILE=/etc/vps-backup.env
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Create it with TG_TOKEN= and TG_CHAT_ID=" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

if [ -z "${TG_TOKEN:-}" ] || [ -z "${TG_CHAT_ID:-}" ]; then
  echo "ERROR: TG_TOKEN or TG_CHAT_ID empty in $ENV_FILE" >&2
  exit 1
fi

DATE=$(date +%Y-%m-%d)
HOSTNAME=$(hostname -s)
BACKUP_DIR=/root/backups
WORK_DIR=$(mktemp -d /tmp/vps-backup-XXXXXX)
ARCHIVE="$BACKUP_DIR/vps-backup-$DATE.tar.gz"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$BACKUP_DIR"

echo "=== VPS backup $DATE start ==="

# --- 1. Coolify postgres dump (logical, version-portable) ---
if docker ps --format '{{.Names}}' | grep -q '^coolify-db$'; then
  echo "Dumping coolify-db..."
  docker exec coolify-db pg_dumpall -U coolify > "$WORK_DIR/coolify-db.sql" \
    || echo "WARNING: coolify-db dump failed (continuing)"
fi

# --- 2. Tar everything important ---
echo "Creating tar archive..."
tar czf "$ARCHIVE" \
  --warning=no-file-changed \
  --exclude='*.db-shm' \
  --exclude='*.db-wal' \
  --exclude='*/__pycache__/*' \
  --ignore-failed-read \
  /data/ \
  /home/ \
  /root/3x-ui/ \
  /usr/local/etc/xray/ \
  /etc/systemd/system/xray*.service \
  /etc/systemd/system/vps-backup.* \
  -C "$WORK_DIR" coolify-db.sql \
  2>/dev/null || echo "tar finished with warnings"

SIZE_BYTES=$(stat -c '%s' "$ARCHIVE")
SIZE_HUMAN=$(du -h "$ARCHIVE" | cut -f1)
echo "Archive: $ARCHIVE ($SIZE_HUMAN)"

# --- 3. Send to Telegram ---
MAX_PART=$((45 * 1024 * 1024))  # 45MB (TG bot limit ~50MB, headroom for caption)

send_doc() {
  local file="$1" caption="$2"
  echo "Uploading: $file ($(du -h "$file" | cut -f1))"
  local response
  response=$(curl -sS --max-time 600 \
    -F "chat_id=$TG_CHAT_ID" \
    -F "caption=$caption" \
    -F "document=@$file" \
    "https://api.telegram.org/bot$TG_TOKEN/sendDocument") || {
      echo "ERROR: curl failed for $file"
      return 1
    }
  echo "$response" | head -c 200
  echo ""
  if echo "$response" | grep -q '"ok":true'; then
    return 0
  else
    echo "ERROR: Telegram API rejected upload"
    return 1
  fi
}

if [ "$SIZE_BYTES" -le "$MAX_PART" ]; then
  send_doc "$ARCHIVE" "VPS $HOSTNAME — backup $DATE ($SIZE_HUMAN)"
else
  echo "Archive larger than $MAX_PART bytes, splitting..."
  split -b 45M -d -a 3 "$ARCHIVE" "$WORK_DIR/part."
  PART_COUNT=$(ls "$WORK_DIR"/part.* | wc -l)
  i=1
  for part in "$WORK_DIR"/part.*; do
    send_doc "$part" "VPS $HOSTNAME — backup $DATE — part $i/$PART_COUNT" \
      || echo "Part $i upload failed"
    i=$((i + 1))
  done
fi

# --- 4. Rotation: keep last 7 daily archives locally ---
echo "Rotating local cache (keeping last 7)..."
# shellcheck disable=SC2012
ls -1tr "$BACKUP_DIR"/vps-backup-*.tar.gz 2>/dev/null \
  | head -n -7 \
  | xargs -r rm -f

echo "=== VPS backup $DATE done ==="
