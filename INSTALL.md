# Установка на компьютер в салоне

> Полная установка на Mac mini с автозапуском и распознаванием — в [MACMINI.md](MACMINI.md).
> Здесь — только проверка камер с любого компьютера.

Компьютер должен быть в той же сети, что и регистратор Hikvision (тот же роутер
или Wi-Fi). Подойдёт Mac mini, любой Mac или Windows.

## 1. Скачать программу

**Вариант А — архивом (проще).** Открыть в браузере (нужен вход в GitHub с
доступом к репозиторию `stepanboost/fftvv`):

https://github.com/stepanboost/fftvv/archive/refs/heads/claude/brave-faraday-e4lrpa.zip

Распаковать, папку переименовать в `fftvv` и положить, например, в домашнюю папку.

**Вариант Б — через git** (тогда обновления — одной командой `git pull`):

```bash
git clone -b claude/brave-faraday-e4lrpa https://github.com/stepanboost/fftvv.git
```

## 2. Поставить Python и ffmpeg

**Mac** (Терминал):

```bash
# если нет Homebrew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 ffmpeg
```

**Windows:**
- Python 3.12 с https://www.python.org/downloads/ — при установке поставить
  галочку **Add python.exe to PATH**.
- ffmpeg: в PowerShell `winget install Gyan.FFmpeg`, затем закрыть и заново
  открыть окно.

Проверка: `python3 --version` (на Windows `python --version`) и `ffmpeg -version`.

## 3. Установить зависимости

**Mac:**

```bash
cd ~/fftvv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q          # должно быть «53 passed»
```

**Windows** (PowerShell):

```powershell
cd $HOME\fftvv
python -m venv .venv
.venv\Scripts\Activate.ps1   # если ругается: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt pytest
python -m pytest -q
```

В каждом новом окне терминала перед запуском снова активировать окружение:
`source .venv/bin/activate` (Mac) или `.venv\Scripts\Activate.ps1` (Windows).

## 4. Найти регистратор

Нужны IP-адрес регистратора, логин и пароль (те же, что в приложении Hik-Connect
или iVMS-4200).

- IP: утилита Hikvision **SADP** (https://www.hikvision.com/en/support/tools/hitools/)
  показывает все устройства в сети; или список устройств в роутере.
- Проверить, что порт видео открыт:
  - Mac: `nc -vz 192.168.1.64 554` → `succeeded`
  - Windows: `Test-NetConnection 192.168.1.64 -Port 554` → `TcpTestSucceeded : True`

## 5. Проверить камеры

**Mac:**

```bash
NVR_PASSWORD='пароль' python -m salon_analytics probe --nvr 192.168.1.64 --out probe
```

**Windows:**

```powershell
$env:NVR_PASSWORD = "пароль"
python -m salon_analytics probe --nvr 192.168.1.64 --out probe
```

Если логин не `admin`: добавить `--user ЛОГИН`. Если пароль не задан, программа
спросит его сама.

Результат в папке `probe/`: `report.txt` (итог по 8 каналам) и `cam1.png` …
`cam8.png` (кадр с наложенной разметкой). Эти файлы и нужно прислать.

## Частые ошибки

| Сообщение | Что делать |
|---|---|
| `401 Unauthorized` | неверный логин/пароль; после нескольких ошибок регистратор блокирует вход на 30 мин |
| `Connection refused` / таймаут | не тот IP, компьютер в другой сети, или RTSP выключен в настройках регистратора (Сеть → Дополнительно → порт RTSP 554) |
| `404 Not Found` на одном канале | камера на этом канале не подключена |
| `ffprobe: command not found` | не установлен ffmpeg (шаг 2) или не перезапущен терминал |
| `No module named numpy` | не активировано окружение `.venv` (шаг 3) |

## Обновление

- через git: `cd ~/fftvv && git pull`
- архивом: скачать заново по ссылке из шага 1 и заменить папку (папку `.venv`
  можно перенести из старой).
