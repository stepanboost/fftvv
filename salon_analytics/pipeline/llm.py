"""Локальная LLM через Ollama: только ночью или по запросу.

LLM ничего не считает: VLM разбирает спорные визиты по кадрам и даёт подсказку
типа услуги, текстовая модель пишет SQL к базе для вопросов в боте.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Optional

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VLM_MODEL = "qwen2.5vl:7b"
TEXT_MODEL = "qwen2.5:7b"

VISIT_PROMPT = """Ты смотришь на кадры из парикмахерской с потолочной камеры.
Ответь строго JSON без пояснений:
{"client_in_cape": true|false,
 "master_hands_on_client": true|false,
 "service": "стрижка"|"окрашивание"|"фольга"|"борода"|"мытьё"|"укладка"|"неясно",
 "confidence": 0..1}"""

SQL_PROMPT = """Напиши один SQLite SELECT по таблицам:
visits(visit_id, day 'YYYY-MM-DD', master, second_master, client, start_utc, end_utc,
       seated_min, chairs, service_hint, confidence, status, staff_client, photo, flags)
master_days(day, master, clients, work_min, on_site_min, exits, exits_min, intervals, load)
Только SQL, без пояснений. Вопрос: {question}"""

SERVICE_HINTS = {"стрижка": "похоже на стрижку", "окрашивание": "похоже на окрашивание",
                 "фольга": "похоже на мелирование (фольга)", "борода": "похоже на бороду",
                 "мытьё": "похоже на мытьё", "укладка": "похоже на укладку"}


def _generate(model: str, prompt: str, images: Optional[list[str]] = None,
              fmt: Optional[str] = None, url: str = OLLAMA_URL, timeout: float = 300) -> str:
    body: dict = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0}
    if images:
        body["images"] = images
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def classify_visit(frame_paths: list[str], **kw) -> dict:
    images = [base64.b64encode(Path(p).read_bytes()).decode() for p in frame_paths[:4]]
    raw = _generate(VLM_MODEL, VISIT_PROMPT, images=images, fmt="json", **kw)
    return parse_visit_answer(raw)


def parse_visit_answer(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"service": "неясно", "confidence": 0.0}
    data["hint"] = SERVICE_HINTS.get(data.get("service", ""), "")
    return data


def question_to_sql(question: str, **kw) -> str:
    sql = _generate(TEXT_MODEL, SQL_PROMPT.format(question=question), **kw).strip()
    return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
