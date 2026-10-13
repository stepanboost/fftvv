"""Раздел 4.1: сопоставление локальных треков камер с глобальными ID.

Каждые 0.5 с треки каждой камеры сопоставляются с глобальными ID венгерским
алгоритмом. Стоимость = расстояние на плане + разница re-ID + штраф за
невозможное перемещение. Новый ID создаётся только в зоне входа.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import SalonConfig
from ..models import ZONE_ENTRANCE, Tick, TrackEvent
from ..similarity import cosine

Vec = list[float]


@dataclass
class LocalTrack:
    camera: str
    track_id: int
    zone: str
    plan_xy: Optional[tuple[float, float]] = None   # точка ног на плане зала (не для сидящих)
    reid: Optional[Vec] = None
    chair: Optional[str] = None       # сидит в этом физическом кресле
    ring_chair: Optional[str] = None
    hands_on: bool = False
    ir: bool = False                  # камера в ИК (ч/б): цвет одежды не сравниваем

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera, self.track_id)


@dataclass
class GlobalState:
    gid: str
    zone: str
    t_last: float
    xy: Optional[tuple[float, float]] = None
    chair: Optional[str] = None
    gallery: list[Vec] = field(default_factory=list)
    status: str = "active"            # active | lost | blind
    created_at_entrance: bool = True

    def reid_dist(self, v: Optional[Vec]) -> float:
        if v is None or not self.gallery:
            return 0.5
        return 1 - max(cosine(v, g) for g in self.gallery)

    def remember(self, v: Optional[Vec], limit: int = 10) -> None:
        if v is None:
            return
        if not self.gallery or min(1 - cosine(v, g) for g in self.gallery) > 0.15:
            self.gallery.append(v)
            del self.gallery[:-limit]


class GlobalIdManager:
    LOST_AFTER_S = 2.0
    SWAP_MARGIN = 0.25

    def __init__(self, cfg: SalonConfig, prefix: str = "G"):
        self.cfg = cfg
        self.prefix = prefix
        self.states: dict[str, GlobalState] = {}
        self.bindings: dict[tuple[str, int], str] = {}
        self.journal: list[TrackEvent] = []
        self.ticks: list[Tick] = []
        self._n = 0

    # --- допустимость перехода -------------------------------------------------
    def _reachable(self, s: GlobalState, zone: str, t: float, strict: bool = False) -> bool:
        """Смежная зона по графу переходов. strict: ещё и не быстрее минимального
        времени перехода (для восстановления потерянных ID)."""
        if zone == s.zone:
            return True
        min_s = self.cfg.transitions.get(s.zone, {}).get(zone)
        if min_s is None:
            return False
        return not strict or t - s.t_last >= min_s

    def _cost(self, s: GlobalState, tr: LocalTrack, t: float) -> float:
        p = self.cfg.params
        if not self._reachable(s, tr.zone, t):
            return p.impossible_penalty
        if self.bindings.get(tr.key) == s.gid:
            sticky = -0.5
        else:
            sticky = 0.0
        if tr.chair and s.chair == tr.chair:
            geo = 0.0                     # сидящий держится привязкой к креслу
        elif tr.plan_xy and s.xy and tr.zone == s.zone:
            d = math.dist(tr.plan_xy, s.xy)
            dt = max(t - s.t_last, p.match_interval_s)
            geo = d + (p.impossible_penalty if d / dt > p.max_speed_mps else 0.0)
        else:
            geo = 1.0
        return geo + self._reid_w(tr) * s.reid_dist(tr.reid) + sticky

    def _reid_w(self, tr: LocalTrack) -> float:
        p = self.cfg.params
        return p.reid_weight_ir if tr.ir else p.reid_weight

    # --- основной шаг ----------------------------------------------------------
    def step(self, t: float, tracks: list[LocalTrack]) -> dict[tuple[str, int], str]:
        p = self.cfg.params
        assigned: dict[tuple[str, int], str] = {}
        by_cam: dict[str, list[LocalTrack]] = defaultdict(list)
        for tr in tracks:
            by_cam[tr.camera].append(tr)

        active = [s for s in self.states.values() if s.status == "active"]
        for cam, cam_tracks in by_cam.items():
            if active:
                cost = np.array([[self._cost(s, tr, t) for s in active] for tr in cam_tracks])
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] <= p.max_match_cost:
                        assigned[cam_tracks[r].key] = active[c].gid
            for tr in cam_tracks:
                if tr.key not in assigned:
                    assigned[tr.key] = self._unmatched(tr, t)

        self._check_swaps(tracks, assigned, t)
        seen: dict[str, LocalTrack] = {}
        for tr in tracks:
            gid = assigned[tr.key]
            s = self.states[gid]
            s.zone, s.t_last, s.status = tr.zone, t, "active"
            s.chair = tr.chair
            if tr.plan_xy and not tr.chair:
                s.xy = tr.plan_xy
            if not tr.chair and not tr.ir:   # накидка и ч/б кадры галерею не портят
                s.remember(tr.reid)
            prev = seen.get(gid)
            if prev is None or (tr.hands_on and not prev.hands_on):
                seen[gid] = tr
        self.bindings = dict(assigned)

        for gid, tr in seen.items():
            self.ticks.append(Tick(t=t, gid=gid, zone=tr.zone, dt=p.match_interval_s,
                                   chair=tr.chair, sitting=tr.chair is not None,
                                   ring_chair=tr.ring_chair, hands_on=tr.hands_on))
        self._age(t, set(seen))
        return assigned

    def _new(self, tr: LocalTrack, t: float, at_entrance: bool) -> str:
        self._n += 1
        gid = f"{self.prefix}{self._n}"
        self.states[gid] = GlobalState(gid, tr.zone, t, tr.plan_xy, tr.chair,
                                       created_at_entrance=at_entrance)
        self.states[gid].remember(tr.reid)
        self.journal.append(TrackEvent(t, gid, "создан", tr.zone if at_entrance else f"в зале {tr.zone}"))
        return gid

    def _unmatched(self, tr: LocalTrack, t: float) -> str:
        if tr.zone == ZONE_ENTRANCE:
            return self._new(tr, t, at_entrance=True)
        # трек без пары посреди зала — ищем среди недавно потерянных
        wait = self.cfg.params.lost_id_wait_s
        best, best_cost = None, self.cfg.params.max_match_cost
        for s in self.states.values():
            if s.status == "active" or t - s.t_last > wait:
                continue
            c = self._reid_w(tr) * s.reid_dist(tr.reid) + (0 if self._reachable(s, tr.zone, t, strict=True) else 10)
            if c < best_cost:
                best, best_cost = s, c
        if best:
            self.journal.append(TrackEvent(t, best.gid, "восстановлен", f"{tr.camera}#{tr.track_id}"))
            return best.gid
        return self._new(tr, t, at_entrance=False)

    def _check_swaps(self, tracks: list[LocalTrack], assigned: dict, t: float) -> None:
        """После пересечения треков: сверка с галереями, при несовпадении — обмен назад."""
        for a in tracks:
            for b in tracks:
                if (a.key >= b.key or a.camera != b.camera or a.reid is None or b.reid is None
                        or a.ir or b.ir):
                    continue
                ga, gb = assigned[a.key], assigned[b.key]
                if ga == gb:
                    continue
                sa, sb = self.states[ga], self.states[gb]
                keep = sa.reid_dist(a.reid) + sb.reid_dist(b.reid)
                swap = sa.reid_dist(b.reid) + sb.reid_dist(a.reid)
                if keep - swap > self.SWAP_MARGIN:
                    assigned[a.key], assigned[b.key] = gb, ga
                    self.journal.append(TrackEvent(t, ga, "обменян", f"с {gb}"))
                    self.journal.append(TrackEvent(t, gb, "обменян", f"с {ga}"))

    def _age(self, t: float, seen: set[str]) -> None:
        blind = set(self.cfg.blind_zones)
        for s in self.states.values():
            if s.gid in seen or s.status != "active" or t - s.t_last < self.LOST_AFTER_S:
                continue
            near_blind = any(z in blind for z in self.cfg.transitions.get(s.zone, {}))
            s.status = "blind" if near_blind else "lost"
            self.journal.append(TrackEvent(t, s.gid, "потерян", s.status))
