#!/bin/bash
# Установка всего на Mac mini. Запуск из папки проекта:  bash scripts/install_mac.sh
# Повторный запуск безопасен — ставит только то, чего нет.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== Папка проекта: $ROOT"

if ! command -v brew >/dev/null; then
  echo "== Ставлю Homebrew (попросит пароль от Mac)"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
fi
brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
PY="$(brew --prefix python@3.12)/bin/python3.12"

if [ ! -d .venv ]; then
  echo "== Создаю окружение Python"
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt pytest
pip install -q --no-build-isolation filterpy       # зависимость BoxMOT, иначе не собирается
pip install -q -r requirements-cv.txt

echo "== Модели (YOLO, YuNet, SFace; OSNet BoxMOT скачает при первом запуске)"
python -m salon_analytics models --out models

echo "== Проверка программы"
python -m pytest -q

mkdir -p data masters logs
if [ ! -f .env ]; then
  cat > .env <<'ENV'
# Настройки салона. Заполнить и сохранить. Этот файл в git не попадает.
NVR_HOST=192.168.1.64
NVR_USER=admin
NVR_PASSWORD=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENV
  echo "== Создан файл .env — впишите в него IP, логин и пароль регистратора"
fi
chmod 600 .env

if [ "${1:-}" = "--autostart" ]; then
  echo "== Включаю автозапуск (днём — запись, в 21:00 — отчёт)"
  mkdir -p ~/Library/LaunchAgents
  for name in live nightly; do
    plist=~/Library/LaunchAgents/ru.nikolskaya.salon-$name.plist
    sed "s#__ROOT__#$ROOT#g" deploy/ru.nikolskaya.salon-$name.plist > "$plist"
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
  done
  launchctl list | grep ru.nikolskaya || true
fi
echo "== Готово"
