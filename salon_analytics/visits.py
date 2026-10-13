"""Раздел 4.3: сборка визитов из сегментов, склейка порванных ID."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .config import SalonConfig
from .models import (ZONE_ENTRANCE, ChairSegment, Outage, Person, Role, Tick,
                     TrackEvent, Visit)
from .similarity import same_person


@dataclass
class Fragment:
    """Непрерывное присутствие одного ID в салоне."""
    gid: str
    start: float
    end: float
    entered: bool       # первый тик в зоне входа
    exited: bool        # последний тик в зоне входа
    torn_inside: bool = False   # ID пропал посреди салона дольше, чем ждём
    prev: "Fragment | None" = None
    next: "Fragment | None" = None
    sim: float = 0.0
    segments: list[ChairSegment] = field(default_factory=list)


def presence_fragments(ticks: Iterable[Tick], gids: set[str], cfg: SalonConfig) -> list[Fragment]:
    p = cfg.params
    by_gid: dict[str, list[Tick]] = defaultdict(list)
    for t in ticks:
        if t.gid in gids and t.present:
            by_gid[t.gid].append(t)
    frags: list[Fragment] = []
    for gid, items in by_gid.items():
        items.sort(key=lambda t: t.t)
        first = items[0]
        cur = Fragment(gid, first.t, first.t + first.dt, entered=first.zone == ZONE_ENTRANCE, exited=False)
        last = first
        for t in items[1:]:
            gap = t.t - (last.t + last.dt)
            via_exit = last.zone == ZONE_ENTRANCE
            limit = p.visit_close_absence_s if via_exit else p.lost_id_wait_s
            if gap > limit:
                cur.exited = via_exit
                cur.torn_inside = not via_exit
                frags.append(cur)
                cur = Fragment(gid, t.t, t.t + t.dt, entered=t.zone == ZONE_ENTRANCE, exited=False)
            else:
                cur.end = t.t + t.dt
            last = t
        cur.exited = last.zone == ZONE_ENTRANCE
        frags.append(cur)
    frags.sort(key=lambda f: f.start)
    return frags


def stitch(frags: list[Fragment], persons: dict[str, Person], cfg: SalonConfig,
           journal: list[TrackEvent]) -> None:
    """Склеивает куски разных ID одного человека: разрыв < 30 мин и совпало
    лицо (> 0.5) или одежда (> 0.7). После честного выхода через вход окно
    20 мин: вернулся позже — это повторный визит.

    Когда в салоне одновременно много клиентов, на один оборванный кусок
    претендуют несколько. Пары перебираются по убыванию сходства, и каждый
    кусок склеивается не больше чем с одним — побеждает самый похожий, а не
    тот, кто раньше попался.
    """
    p = cfg.params
    pairs: list[tuple[float, int, int]] = []
    for j, later in enumerate(frags):
        for i, earlier in enumerate(frags):
            if earlier is later or earlier.gid == later.gid:
                continue
            gap = later.start - earlier.end
            # вышел через вход и вернулся: тот же визит только при отсутствии ≤ 20 мин
            limit = p.visit_close_absence_s if earlier.exited else p.stitch_max_gap_s
            if not (0 <= gap < limit):
                continue
            a, b = persons.get(earlier.gid), persons.get(later.gid)
            if a is None or b is None:
                continue
            ok, sim = same_person(a.face, b.face, a.reid, b.reid, p.stitch_face_cos, p.stitch_reid_cos)
            if ok:
                pairs.append((sim, i, j))
    for sim, i, j in sorted(pairs, key=lambda x: -x[0]):
        earlier, later = frags[i], frags[j]
        if earlier.next is not None or later.prev is not None:
            continue
        earlier.next, later.prev, later.sim = later, earlier, sim
        journal.append(TrackEvent(later.start, later.gid, "склеен",
                                  f"с {earlier.gid}, сходство {sim:.2f}"))


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def chairs_path(segments: list[ChairSegment], cfg: SalonConfig) -> list[str]:
    path: list[str] = []
    for s in sorted(segments, key=lambda s: s.start):
        if not (s.accepted or cfg.chair_kind(s.chair) == "wash"):
            continue
        label = "мойка" if cfg.chair_kind(s.chair) == "wash" else s.chair
        if not path or path[-1] != label:
            path.append(label)
    return path


def build_visits(ticks: list[Tick], segments: list[ChairSegment], persons: dict[str, Person],
                 cfg: SalonConfig, outages: list[Outage] | None = None,
                 journal: list[TrackEvent] | None = None) -> tuple[list[Visit], list[Visit]]:
    """Возвращает (визиты для отчёта, возможные визиты только для очереди проверки)."""
    p = cfg.params
    journal = journal if journal is not None else []
    outages = outages or []
    client_gids = {g for g, pr in persons.items() if pr.role == Role.CLIENT}
    client_gids |= {s.gid for s in segments if s.gid not in persons}

    frags = presence_fragments(ticks, client_gids, cfg)
    stitch(frags, persons, cfg, journal)

    seg_by_gid: dict[str, list[ChairSegment]] = defaultdict(list)
    for s in segments:
        seg_by_gid[s.gid].append(s)
    for f in frags:
        f.segments = [s for s in seg_by_gid[f.gid] if _overlaps(s.start, s.end, f.start, f.end)]

    visits: list[Visit] = []
    possible: list[Visit] = []
    n = 0
    for head in frags:
        if head.prev is not None:
            continue
        chain = []
        f: Fragment | None = head
        while f:
            chain.append(f)
            f = f.next
        segs = sorted((s for c in chain for s in c.segments), key=lambda s: s.start)
        if not segs:
            continue
        n += 1
        v = Visit(
            visit_id=f"V{n:03d}",
            gids=list(dict.fromkeys(c.gid for c in chain)),
            start=chain[0].start, end=chain[-1].end,
            segments=segs,
            entered_via_entrance=chain[0].entered,
            exited_via_entrance=chain[-1].exited,
        )
        _score_visit(v, chain, persons, cfg, outages, journal)
        if any(s.accepted for s in segs if cfg.chair_kind(s.chair) != "wash"):
            visits.append(v)
        else:
            v.review_reasons.extend(s.review_reason for s in segs if s.review_reason)
            if v.review_reasons:
                v.status = "на проверке"
                possible.append(v)

    visits.extend(_staff_client_visits(segments, len(visits) + len(possible), cfg, outages))
    visits.sort(key=lambda v: v.start)
    return visits, possible


def _score_visit(v: Visit, chain: list[Fragment], persons: dict[str, Person], cfg: SalonConfig,
                 outages: list[Outage], journal: list[TrackEvent]) -> None:
    p = cfg.params
    v.zones_path = chairs_path(v.segments, cfg)
    conf = 1.0
    if any(s.no_data for s in v.segments) or any(_overlaps(o.start, o.end, v.start, v.end) for o in outages):
        conf -= 0.3
        v.flags.append("нет данных части камер")
    if not v.entered_via_entrance:
        conf -= 0.15
        v.flags.append("ID создан в зале")
    if not v.exited_via_entrance:
        conf -= 0.05
        v.flags.append("выход не зафиксирован")
    if len(chain) > 1:
        conf -= 0.1 * (len(chain) - 1)
        v.flags.append(f"склеено из {len(chain)} кусков")
    swaps = [e for e in journal if e.gid in v.gids and e.kind == "обменян" and v.start <= e.t <= v.end]
    if swaps:
        conf -= 0.2
        v.flags.append("была подмена ID")
    if v.duration_s > p.visit_max_s:
        v.review_reasons.append(f"визит дольше {p.visit_max_s / 3600:.0f} ч")
    for s in v.segments:
        if s.review_reason and not s.accepted:
            v.flags.append(s.review_reason)
    v.visit_conf = max(0.0, round(conf, 2))


def _staff_client_visits(segments: list[ChairSegment], offset: int, cfg: SalonConfig,
                         outages: list[Outage]) -> list[Visit]:
    """Мастер стрижёт другого мастера: визит = сама посадка, пометка «сотрудник»."""
    out = []
    for s in segments:
        if s.staff_client:
            offset += 1
            v = Visit(visit_id=f"V{offset:03d}", gids=[s.gid], start=s.start, end=s.end,
                      segments=[s], staff_client=True)
            v.zones_path = [s.chair]
            v.flags.append("сотрудник")
            if s.no_data or any(_overlaps(o.start, o.end, s.start, s.end) for o in outages):
                v.visit_conf = 0.7
            out.append(v)
    return out
