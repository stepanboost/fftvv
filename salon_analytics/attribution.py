"""Раздел 5: мастер визита — тот, кто дольше всех работал руками у клиента."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import SalonConfig
from .models import Visit


@dataclass(frozen=True)
class Shift:
    master: str
    start: float
    end: float


@dataclass(frozen=True)
class Booking:
    """Запись из CRM: клиент записан к мастеру на время t."""
    master: str
    t: float
    client: str = ""


def _on_shift(master: str, t0: float, t1: float, shifts: Optional[list[Shift]]) -> bool:
    if shifts is None or master.startswith("сотрудник?"):   # графика нет / ещё не опознан
        return True
    return any(s.master == master and s.start < t1 and t0 < s.end for s in shifts)


def attribute(v: Visit, cfg: SalonConfig, shifts: Optional[list[Shift]] = None,
              bookings: Optional[list[Booking]] = None) -> None:
    p = cfg.params
    hands: dict[str, float] = {}
    wash_hands: dict[str, float] = {}
    near: dict[str, float] = {}
    for s in v.segments:
        is_wash = cfg.chair_kind(s.chair) == "wash"
        if not (s.accepted or is_wash):
            continue
        for m, sec in s.hands_s.items():
            hands[m] = hands.get(m, 0.0) + sec
            if is_wash:
                wash_hands[m] = wash_hands.get(m, 0.0) + sec
        for m, sec in s.near_s.items():
            near[m] = near.get(m, 0.0) + sec

    by_near = not any(hands.values())   # руки не видны — голосуем временем рядом
    votes = near if by_near else hands
    votes = {m: s for m, s in votes.items() if s > 0 and _on_shift(m, v.start, v.end, shifts)}
    if not votes:
        v.master, v.master_conf = None, 0.0
        v.review_reasons.append("мастер не определён")
        return

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    total = sum(votes.values())
    v.master, main_s = ranked[0]
    # второй — с долей ≥ 20% (колорист) или тот, кто мыл голову (ассистент)
    for m, sec in ranked[1:]:
        if wash_hands.get(m, 0) >= 0.5 * sec:
            v.second_master, v.second_master_role = m, "мойка"
            break
        if sec / total >= p.second_master_share:
            v.second_master = m
            break

    dominance = main_s / total
    enough_time = min(1.0, main_s / p.min_sitting_s)
    conf = 0.1 + 0.8 * dominance + 0.1 * enough_time
    if by_near:
        conf -= 0.2
        v.flags.append("руки не видны")
    first_chair = next((s.chair for s in v.segments if s.accepted), None)
    if first_chair and cfg.pinned_master(first_chair) == v.master:
        conf += p.pinned_chair_bonus
    if bookings:
        first = min(s.start for s in v.segments if s.accepted) if any(s.accepted for s in v.segments) else v.start
        if any(b.master == v.master and abs(b.t - first) <= p.booking_window_s for b in bookings):
            conf += p.booking_bonus
            v.flags.append("совпало с записью CRM")
    if v.master.startswith("сотрудник?"):
        conf = min(conf, 0.5)
        v.review_reasons.append("неизвестный сотрудник")
    v.master_conf = round(max(0.0, min(1.0, conf)), 2)
