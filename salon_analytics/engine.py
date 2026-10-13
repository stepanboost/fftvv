"""Вечерний расчёт дня: тики глобальных ID → визиты, мастера, проверки."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .attribution import Booking, Shift, attribute
from .checks import HeadCount, Violation, run_all
from .config import SalonConfig
from .models import FaceCandidate, MasterDay, Outage, Person, Tick, TrackEvent, Visit
from .photos import KnownFace, apply_photos
from .sittings import find_sittings
from .staff import master_days
from .visits import build_visits


@dataclass
class DayInput:
    ticks: list[Tick]
    persons: dict[str, Person]
    faces: dict[str, list[FaceCandidate]] = field(default_factory=dict)
    outages: list[Outage] = field(default_factory=list)
    journal: list[TrackEvent] = field(default_factory=list)
    shifts: Optional[list[Shift]] = None
    bookings: list[Booking] = field(default_factory=list)
    crm_checks: Optional[int] = None
    headcounts: list[HeadCount] = field(default_factory=list)
    face_history: list[KnownFace] = field(default_factory=list)
    service_hints: dict[str, str] = field(default_factory=dict)   # visit_id → подсказка VLM
    cameras: list[str] = field(default_factory=list)


@dataclass
class DayResult:
    visits: list[Visit]
    possible: list[Visit]
    masters: dict[str, MasterDay]
    violations: list[Violation]
    coverage: float
    journal: list[TrackEvent]
    crm_checks: Optional[int] = None

    @property
    def review_queue(self) -> list[Visit]:
        return [v for v in self.visits if v.status == "на проверке"] + self.possible


def camera_coverage(outages: list[Outage], cameras: list[str], t0: float, t1: float) -> float:
    if not cameras or t1 <= t0:
        return 1.0
    lost = 0.0
    for o in outages:
        if o.camera in cameras:
            lost += max(0.0, min(o.end, t1) - max(o.start, t0))
    return max(0.0, 1 - lost / (len(cameras) * (t1 - t0)))


def analyze_day(inp: DayInput, cfg: SalonConfig) -> DayResult:
    p = cfg.params
    journal = list(inp.journal)
    segments = find_sittings(inp.ticks, inp.persons, cfg)
    visits, possible = build_visits(inp.ticks, segments, inp.persons, cfg, inp.outages, journal)
    for v in visits:
        attribute(v, cfg, inp.shifts, inp.bookings)
        v.service_hint = inp.service_hints.get(v.visit_id, "")
    apply_photos(visits, inp.faces, p, inp.face_history)
    days = master_days(inp.ticks, inp.persons, visits, cfg)
    violations = run_all(inp.ticks, inp.persons, visits, days, cfg, inp.headcounts, inp.crm_checks)

    for v in visits:
        if v.confidence < p.review_conf:
            v.review_reasons.append(f"уверенность {v.confidence:.2f}")
        v.review_reasons = list(dict.fromkeys(v.review_reasons))
        if v.review_reasons:
            v.status = "на проверке"

    t0 = min((t.t for t in inp.ticks), default=0.0)
    t1 = max((t.t + t.dt for t in inp.ticks), default=0.0)
    cameras = inp.cameras or list(cfg.cameras)
    return DayResult(visits, possible, days, violations,
                     camera_coverage(inp.outages, cameras, t0, t1), journal, inp.crm_checks)
