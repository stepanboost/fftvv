"""Отправка вечернего отчёта в Telegram (Bot API, без сторонних библиотек).

Нужны переменные окружения TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path


def _post(method: str, fields: dict, file: tuple[str, Path] | None = None) -> dict:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if file:
        name, path = file
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def send_report(text: str, html_file: Path) -> None:
    chat = os.environ["TELEGRAM_CHAT_ID"]
    _post("sendMessage", {"chat_id": chat, "text": text[:4000]})
    _post("sendDocument", {"chat_id": chat}, ("document", html_file))
