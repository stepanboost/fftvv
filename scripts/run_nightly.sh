#!/bin/bash
# Вечерний отчёт за сегодня: data/<день>/report/masters.html (+ Telegram, если задан токен).
cd "$(dirname "$0")/.."
set -a; source .env; set +a
EXTRA=""
[ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] && EXTRA="--telegram"
exec .venv/bin/python -m salon_analytics nightly --config config/salon.yaml --data data $EXTRA "$@"
