"""SQLite-база результатов: визиты, мастера по дням, журнал треков, исправления."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .config import SalonConfig
from .engine import DayResult
from .report import salon_tz

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    visit_id TEXT, day TEXT, master TEXT, second_master TEXT, client TEXT,
    start_utc REAL, end_utc REAL, seated_min REAL, chairs TEXT, service_hint TEXT,
    confidence REAL, status TEXT, staff_client INTEGER, photo TEXT, flags TEXT,
    PRIMARY KEY (day, visit_id)
);
CREATE TABLE IF NOT EXISTS master_days (
    day TEXT, master TEXT, clients INTEGER, work_min REAL, on_site_min REAL,
    exits INTEGER, exits_min REAL, intervals TEXT, load REAL,
    PRIMARY KEY (day, master)
);
CREATE TABLE IF NOT EXISTS track_events (
    day TEXT, t_utc REAL, gid TEXT, kind TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS corrections (
    day TEXT, visit_id TEXT, action TEXT, value TEXT, by_user TEXT, t_utc REAL
);
"""

READABLE_TABLES = {"visits", "master_days"}


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def save_day(con: sqlite3.Connection, day: str, r: DayResult, cfg: SalonConfig) -> None:
    con.execute("DELETE FROM visits WHERE day = ?", (day,))
    con.execute("DELETE FROM master_days WHERE day = ?", (day,))
    con.execute("DELETE FROM track_events WHERE day = ?", (day,))
    for v in r.visits:
        con.execute(
            "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (v.visit_id, day, v.master, v.second_master, v.client_label, v.start, v.end,
             round(v.seated_s / 60, 1), " → ".join(v.zones_path), v.service_hint,
             v.confidence, v.status, int(v.staff_client),
             v.photo.best.path if v.photo and v.photo.best else None,
             json.dumps(v.flags + v.review_reasons, ensure_ascii=False)))
    for name, d in r.masters.items():
        con.execute(
            "INSERT INTO master_days VALUES (?,?,?,?,?,?,?,?,?)",
            (day, name, d.clients, round(d.work_s / 60, 1), round(d.on_site_s / 60, 1),
             len(d.exits), round(sum(b - a for a, b in d.exits) / 60, 1),
             json.dumps(d.intervals), round(d.load, 3)))
    con.executemany("INSERT INTO track_events VALUES (?,?,?,?,?)",
                    [(day, e.t, e.gid, e.kind, e.detail) for e in r.journal])
    con.commit()


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def run_readonly_query(db_path: str | Path, sql: str, limit: int = 200) -> list[tuple]:
    """Выполняет SELECT, который написала LLM для вопроса в боте.

    LLM ничего не считает сама: она только пишет запрос, а выполняет его этот код,
    в режиме только чтения и только по разрешённым таблицам.
    """
    q = sql.strip().rstrip(";")
    if ";" in q or not re.match(r"(?is)^\s*(select|with)\b", q) or _FORBIDDEN.search(q):
        raise ValueError("разрешён только один SELECT")
    tables = {t.lower() for t in re.findall(r"(?i)\b(?:from|join)\s+([a-z_]+)", q)}
    if not tables <= READABLE_TABLES:
        raise ValueError(f"недоступные таблицы: {sorted(tables - READABLE_TABLES)}")
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        return con.execute(q).fetchmany(limit)
    finally:
        con.close()
