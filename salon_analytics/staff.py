"""Раздел 5: учёт мастера за день — приход, уход, выходы, загрузка."""
from __future__ import annotations

from collections import defaultdict

from .config import SalonConfig
from .models import ZONE_ENTRANCE, MasterDay, Person, Role, Tick, Visit


def master_days(ticks: list[Tick], persons: dict[str, Person], visits: list[Visit],
                cfg: SalonConfig) -> dict[str, MasterDay]:
    p = cfg.params
    name_of = {g: (pr.name or g) for g, pr in persons.items() if pr.role == Role.MASTER}
    by_name: dict[str, list[Tick]] = defaultdict(list)
    for t in ticks:
        if t.gid in name_of and t.present:
            by_name[name_of[t.gid]].append(t)

    days: dict[str, MasterDay] = {}
    for name, items in by_name.items():
        items.sort(key=lambda t: t.t)
        d = MasterDay(name=name, first_seen=items[0].t, last_seen=items[-1].t + items[-1].dt)
        d.departure_approx = items[-1].zone != ZONE_ENTRANCE
        start = items[0].t
        last = items[0]
        for t in items[1:]:
            gap = t.t - (last.t + last.dt)
            # выход = прошёл зону входа и не виден > 2 мин; подсобка — «на месте»
            if last.zone == ZONE_ENTRANCE and gap > p.staff_exit_min_s:
                d.exits.append((last.t + last.dt, t.t))
                if gap >= p.staff_interval_split_s:
                    d.intervals.append((start, last.t + last.dt))
                    start = t.t
            last = t
        d.intervals.append((start, d.last_seen))
        span = sum(b - a for a, b in d.intervals)
        short_exits = sum(b - a for a, b in d.exits if b - a < p.staff_interval_split_s)
        d.on_site_s = span - short_exits
        days[name] = d

    for v in visits:
        if v.master in days:
            days[v.master].clients += 1
        for s in v.segments:
            if s.accepted or cfg.chair_kind(s.chair) == "wash":
                for m, sec in s.hands_s.items():
                    if m in days:
                        days[m].work_s += sec
    return days
