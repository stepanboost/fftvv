"""Раздел 8: плоская таблица «одна строка = один визит», шапки, CSV, Telegram."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone, tzinfo

from .config import SalonConfig
from .engine import DayResult
from .models import Visit

COLUMNS = ["Дата", "Мастер", "Второй мастер", "Клиент", "Фото", "Визит", "В кресле",
           "Кресла / зоны", "Тип (подсказка)", "Уверенность", "Статус", "Пометки"]


def salon_tz(cfg: SalonConfig) -> tzinfo:
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(cfg.timezone)
    except Exception:   # нет tzdata в системе
        return timezone(timedelta(hours=cfg.utc_offset_hours))


def _local(ts: float, tz: tzinfo) -> datetime:
    return datetime.fromtimestamp(ts, tz)


def _hm(ts: float, tz: tzinfo) -> str:
    return _local(ts, tz).strftime("%H:%M")


def _mins(sec: float) -> int:
    return int(round(sec / 60))


def plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def visit_row(v: Visit, tz: tzinfo, review_conf: float) -> dict[str, str]:
    approx = "≈" if v.confidence < review_conf or v.status == "на проверке" else ""
    second = ""
    if v.second_master:
        second = v.second_master + (f" ({v.second_master_role})" if v.second_master_role else "")
    client = v.client_label + (f" / «{v.repeat_note}»" if v.repeat_note else "")
    if v.staff_client:
        client += " (сотрудник)"
    photo = v.photo.best.path if v.photo and v.photo.best else ""
    return {
        "Дата": _local(v.start, tz).strftime("%d.%m.%Y"),
        "Мастер": v.master or "?",
        "Второй мастер": second,
        "Клиент": client,
        "Фото": photo,
        "Визит": f"{_hm(v.start, tz)}–{_hm(v.end, tz)} ({_mins(v.duration_s)} мин)",
        "В кресле": f"{_mins(v.seated_s)} мин",
        "Кресла / зоны": " → ".join(v.zones_path),
        "Тип (подсказка)": v.service_hint,
        "Уверенность": f"{approx}{v.confidence:.2f}",
        "Статус": v.status,
        "Пометки": "; ".join(dict.fromkeys(v.flags + v.review_reasons)),
    }


def rows(day: DayResult, cfg: SalonConfig) -> list[dict[str, str]]:
    tz = salon_tz(cfg)
    ordered = sorted(day.visits, key=lambda v: (v.master or "яяя", v.start))
    return [visit_row(v, tz, cfg.params.review_conf) for v in ordered]


def to_csv(day: DayResult, cfg: SalonConfig) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter=";")
    w.writeheader()
    w.writerows(rows(day, cfg))
    return buf.getvalue()


def day_header(day: DayResult) -> dict[str, str]:
    h = {
        "Всего визитов": str(len(day.visits)),
        "Покрытие камер": f"{day.coverage * 100:.0f}%",
        "Ждут проверки": str(len(day.review_queue)),
        "Нарушений самопроверки": str(len(day.violations)),
    }
    if day.crm_checks is not None:
        h["Сверка с CRM"] = f"по камерам {len(day.visits)}, в CRM {day.crm_checks} чеков"
    return h


def master_summary(day: DayResult, cfg: SalonConfig) -> list[dict[str, str]]:
    tz = salon_tz(cfg)
    out = []
    for name, d in sorted(day.masters.items()):
        intervals = ", ".join(f"{_hm(a, tz)}–{_hm(b, tz)}" for a, b in d.intervals)
        if d.departure_approx:
            intervals += " (уход ≈)"
        out.append({
            "Мастер": name,
            "Клиентов": str(d.clients),
            "Минут с клиентами": str(_mins(d.work_s)),
            "На месте": intervals,
            "Выходов": f"{len(d.exits)} ({_mins(sum(b - a for a, b in d.exits))} мин)",
            "Загрузка": f"{d.load * 100:.0f}%",
        })
    return out


def telegram_summary(day: DayResult, cfg: SalonConfig, site_url: str = "") -> str:
    """Шаблонная сводка; ночью LLM может переписать её «по-человечески»."""
    tz = salon_tz(cfg)
    lines = []
    if day.visits:
        lines.append(f"Сводка за {_local(day.visits[0].start, tz).strftime('%d.%m.%Y')}")
    for k, v in day_header(day).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    for m in master_summary(day, cfg):
        d = day.masters[m["Мастер"]]
        exits = ""
        if d.exits:
            a, b = max(d.exits, key=lambda e: e[1] - e[0])
            exits = f", дольше всего отсутствовал(а) {_mins(b - a)} мин с {_hm(a, tz)}"
        clients = plural(d.clients, "клиент", "клиента", "клиентов")
        lines.append(f"• {m['Мастер']}: {clients}, загрузка {m['Загрузка']}{exits}")
    if site_url:
        lines += ["", site_url]
    return "\n".join(lines)


def review_cards(day: DayResult, cfg: SalonConfig) -> list[dict[str, object]]:
    tz = salon_tz(cfg)
    cards = []
    for v in day.review_queue:
        frames = []
        if v.photo:
            frames = [c.path for c in ([v.photo.best] if v.photo.best else []) + v.photo.spares][:3]
        cards.append({
            "visit_id": v.visit_id,
            "время": f"{_hm(v.start, tz)}–{_hm(v.end, tz)}",
            "мастер": v.master,
            "причины": v.review_reasons,
            "кадры": frames,
            # 15-секундный клип из архива регистратора вокруг первой посадки
            "клип_utc": [v.segments[0].start, v.segments[0].start + 15] if v.segments else None,
            "кнопки": ["верно", "другой мастер", "склеить с…", "разделить", "не клиент"],
        })
    return cards
