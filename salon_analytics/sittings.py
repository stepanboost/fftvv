"""Раздел 4.2: сегменты «кресло» и фильтры посадки."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .config import SalonConfig
from .models import ChairSegment, Person, Role, Tick


def master_label(p: Person) -> str:
    if p.role == Role.STAFF_UNKNOWN:
        return f"сотрудник? ({p.gid})"
    return p.name or p.gid


def voter_roles(cfg: SalonConfig) -> set[Role]:
    roles = {Role.MASTER, Role.STAFF_UNKNOWN}
    if cfg.params.admin_votes:
        roles.add(Role.ADMIN)
    return roles


def build_segments(ticks: Iterable[Tick], persons: dict[str, Person],
                   cfg: SalonConfig) -> list[ChairSegment]:
    """Непрерывные сидения одного ID в одном физическом кресле.

    Разрывы до `sitting_gap_merge_s` (встал поправить накидку, трекер мигнул)
    склеиваются. Смена ID в кресле или смена кресла — новый сегмент.
    """
    p = cfg.params
    by_gid: dict[str, list[Tick]] = defaultdict(list)
    for t in ticks:
        if t.sitting and t.chair:
            by_gid[t.gid].append(t)

    segments: list[ChairSegment] = []
    for gid, items in by_gid.items():
        role = persons.get(gid, Person(gid)).role
        if role == Role.GUEST:
            continue
        items.sort(key=lambda t: t.t)
        cur: ChairSegment | None = None
        for t in items:
            if cur and t.chair == cur.chair and t.t - cur.end <= p.sitting_gap_merge_s:
                cur.end = t.t + t.dt
                cur.seated_s += t.dt
            else:
                cur = ChairSegment(gid=gid, chair=t.chair, start=t.t, end=t.t + t.dt, seated_s=t.dt)
                segments.append(cur)
            if not t.cameras_ok:
                cur.no_data = True
    segments.sort(key=lambda s: s.start)
    return segments


def attach_master_time(segments: list[ChairSegment], ticks: Iterable[Tick],
                       persons: dict[str, Person], cfg: SalonConfig) -> None:
    """Сколько секунд каждый сотрудник стоял у кресла и работал руками.

    В кадре может быть несколько клиентов, кольца соседних кресел перекрываются.
    Тик сотрудника засчитывается одному креслу: тому, у чьего клиента руки, а
    если руки ни у кого — ближайшему занятому (ring_chair). Так мастер, который
    красит клиента в 2-2, стоя ближе к 2-1, не набирает «рядом» клиенту в 2-1.
    """
    roles = voter_roles(cfg)
    by_chair: dict[str, list[ChairSegment]] = defaultdict(list)
    for s in segments:
        by_chair[s.chair].append(s)
    for t in ticks:
        hands = t.hands_target
        near = hands or t.ring_chair
        if not near or near not in by_chair:
            continue
        person = persons.get(t.gid)
        if person is None or person.role not in roles:
            continue
        label = master_label(person)
        for s in by_chair[near]:
            if s.gid != t.gid and s.start <= t.t < s.end:
                s.near_s[label] = s.near_s.get(label, 0.0) + t.dt
                if hands:
                    s.hands_s[label] = s.hands_s.get(label, 0.0) + t.dt


def classify(segments: list[ChairSegment], persons: dict[str, Person],
             cfg: SalonConfig) -> list[ChairSegment]:
    """Проставляет accepted / staff_client / review_reason.

    Возвращает сегменты, которые имеют значение для визитов: засчитанные,
    спорные и мойку. Отдых мастера в кресле отбрасывается.
    """
    p = cfg.params
    kept: list[ChairSegment] = []
    for s in segments:
        kind = cfg.chair_kind(s.chair)
        role = persons.get(s.gid, Person(s.gid)).role
        near = sum(s.near_s.values())
        has_master = near >= p.min_master_near_s
        min_s = p.min_sitting_barber_s if kind == "barber" else p.min_sitting_s

        if kind == "wash":
            # мытьё — сегмент внутри визита, сам по себе не посадка
            kept.append(s)
            continue
        if role in (Role.MASTER, Role.ADMIN, Role.STAFF_UNKNOWN):
            if has_master and s.seated_s >= min_s:
                s.accepted = True
                s.staff_client = True     # мастер стрижёт другого мастера
                kept.append(s)
            continue                       # сотрудник отдыхает/ест — не посадка
        if s.seated_s >= min_s and has_master:
            s.accepted = True
        elif has_master and s.seated_s >= p.short_sitting_review_s:
            s.review_reason = f"короткая посадка {s.seated_s / 60:.0f} мин"
        elif not has_master and s.seated_s > p.sat_without_master_review_s:
            s.review_reason = f"сидел {s.seated_s / 60:.0f} мин без мастера"
        else:
            continue
        kept.append(s)
    return kept


def find_sittings(ticks: list[Tick], persons: dict[str, Person], cfg: SalonConfig) -> list[ChairSegment]:
    segments = build_segments(ticks, persons, cfg)
    attach_master_time(segments, ticks, persons, cfg)
    return classify(segments, persons, cfg)
