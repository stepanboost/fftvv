"""Раздел 5.2: правила, которые в жизни не нарушаются. Нарушение = сбой трекинга."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .config import SalonConfig
from .models import ZONE_ENTRANCE, MasterDay, Person, Role, Tick, Visit


@dataclass(frozen=True)
class Violation:
    rule: str
    t: Optional[float]
    subject: str
    detail: str = ""


@dataclass(frozen=True)
class HeadCount:
    """Число людей в зоне по детекциям (без трекинга), раз в 5 мин."""
    t: float
    zone: str
    count: int


def one_place_at_a_time(ticks: list[Tick]) -> list[Violation]:
    seen: dict[tuple[str, float], str] = {}
    out = []
    for t in ticks:
        key = (t.gid, round(t.t))
        if key in seen and seen[key] != t.zone:
            out.append(Violation("один ID в двух местах", t.t, t.gid, f"{seen[key]} и {t.zone}"))
        seen[key] = t.zone
    return out


def headcount_matches(ticks: list[Tick], counts: list[HeadCount]) -> list[Violation]:
    active: dict[tuple[float, str], set[str]] = defaultdict(set)
    for t in ticks:
        active[(round(t.t), t.zone)].add(t.gid)
    out = []
    for c in counts:
        n = len(active.get((round(c.t), c.zone), set()))
        if n != c.count:
            out.append(Violation("люди ≠ активные ID", c.t, c.zone, f"детекций {c.count}, ID {n}"))
    return out


def clients_left(ticks: list[Tick], persons: dict[str, Person]) -> list[Violation]:
    last: dict[str, Tick] = {}
    for t in ticks:
        if t.present and (t.gid not in last or t.t > last[t.gid].t):
            last[t.gid] = t
    return [Violation("клиент «завис» к закрытию", t.t, gid, t.zone)
            for gid, t in last.items()
            if persons.get(gid, Person(gid)).role == Role.CLIENT and t.zone != ZONE_ENTRANCE]


def visits_have_bounds(visits: list[Visit]) -> list[Violation]:
    out = []
    for v in visits:
        if (not v.entered_via_entrance or not v.exited_via_entrance) and not v.flags:
            out.append(Violation("визит без входа/выхода и без причины", v.start, v.visit_id))
    return out


def master_one_client_per_minute(ticks: list[Tick], persons: dict[str, Person],
                                 min_s: float = 20.0) -> list[Violation]:
    """Мастер не работает руками с двумя клиентами в одну минуту.

    Переход от одного кресла к соседнему внутри минуты — норма, поэтому
    нарушение, только если у двух и более кресел набралось ≥ min_s секунд.
    """
    secs: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in ticks:
        pr = persons.get(t.gid)
        if pr and pr.role == Role.MASTER and t.hands_target:
            secs[(pr.name or t.gid, int(t.t // 60))][t.hands_target] += t.dt
    out = []
    for (name, m), per_chair in sorted(secs.items()):
        busy = sorted(c for c, s in per_chair.items() if s >= min_s)
        if len(busy) > 1:
            out.append(Violation("мастер руками у двух клиентов", m * 60.0, name, ", ".join(busy)))
    return out


def master_visits_in_intervals(visits: list[Visit], days: dict[str, MasterDay]) -> list[Violation]:
    out = []
    for v in visits:
        d = days.get(v.master or "")
        if d and not any(a < v.end and v.start < b for a, b in d.intervals):
            out.append(Violation("визит вне интервалов «на месте»", v.start, v.visit_id, v.master or ""))
    return out


def crm_reconcile(visits: list[Visit], crm_checks: Optional[int], tolerance: int) -> list[Violation]:
    if crm_checks is None:
        return []
    n = sum(1 for v in visits if not v.staff_client)
    if abs(n - crm_checks) > tolerance:
        return [Violation("расхождение с CRM", None, "день", f"по камерам {n}, в CRM {crm_checks} чеков")]
    return []


def run_all(ticks: list[Tick], persons: dict[str, Person], visits: list[Visit],
            days: dict[str, MasterDay], cfg: SalonConfig,
            counts: Optional[list[HeadCount]] = None, crm_checks: Optional[int] = None) -> list[Violation]:
    return [
        *one_place_at_a_time(ticks),
        *headcount_matches(ticks, counts or []),
        *clients_left(ticks, persons),
        *visits_have_bounds(visits),
        *master_one_client_per_minute(ticks, persons),
        *master_visits_in_intervals(visits, days),
        *crm_reconcile(visits, crm_checks, cfg.params.crm_tolerance),
    ]
