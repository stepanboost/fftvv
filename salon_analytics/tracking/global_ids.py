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
    hands_chair: Optional[str] = None  # у чьей головы руки (scene.resolve_staff)
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
    born_t: float = 0.0

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
    CROWD_RADIUS_M = 1.5      # люди ближе — одна «толпа», ID сверяются вместе
    SAME_PERSON_M = 1.0       # один ID на двух камерах не может быть дальше
    GROUP_ENTRY_S = 600.0     # вошли компанией: отделившийся трек ищем у вошедших за 10 мин
    RECOVER_MAX_REID = 0.5    # потерянный ID отдаём новому треку только при похожей одежде
    RECOVER_SURE_REID = 0.3   # ...а рядом с вошедшей компанией — только при очень похожей

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
        by_cam: dict[str, list[LocalTrack]] = defaultdict(list)
        for tr in tracks:
            by_cam[tr.camera].append(tr)

        # 1. каждая камера независимо: треки ↔ активные ID
        active = [s for s in self.states.values() if s.status == "active"]
        cand: dict[tuple[str, int], tuple[str, float]] = {}
        for cam_tracks in by_cam.values():
            if not active:
                break
            cost = np.array([[self._cost(s, tr, t) for s in active] for tr in cam_tracks])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= p.max_match_cost:
                    cand[cam_tracks[r].key] = (active[c].gid, float(cost[r, c]))

        # 2. один ID не может быть двумя людьми на разных камерах
        self._resolve_cross_camera(tracks, cand, t)
        assigned = {k: g for k, (g, _) in cand.items()}

        # 3. треки без пары: вход, потерянные, отделившиеся от компании
        for tr in tracks:
            if tr.key not in assigned:
                assigned[tr.key] = self._unmatched(tr, t, taken=set(assigned.values()))

        # 4. после пересечений в толпе — сверка всей группы с галереями
        self._fix_crowd_swaps(tracks, assigned, t)

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
            if prev is None or (_working(tr) and not _working(prev)):
                seen[gid] = tr
        self.bindings = dict(assigned)

        for gid, tr in seen.items():
            self.ticks.append(Tick(t=t, gid=gid, zone=tr.zone, dt=p.match_interval_s,
                                   chair=tr.chair, sitting=tr.chair is not None,
                                   ring_chair=tr.ring_chair, hands_on=tr.hands_on,
                                   hands_chair=tr.hands_chair))
        self._age(t, set(seen))
        return assigned

    def _resolve_cross_camera(self, tracks: list[LocalTrack],
                              cand: dict[tuple[str, int], tuple[str, float]], t: float) -> None:
        by_key = {tr.key: tr for tr in tracks}
        by_gid: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for k, (g, _) in cand.items():
            by_gid[g].append(k)
        for gid, keys in by_gid.items():
            if len(keys) < 2:
                continue
            keys.sort(key=lambda k: cand[k][1])
            anchor = by_key[keys[0]]
            for k in keys[1:]:
                other = by_key[k]
                if _far(anchor, other, self.SAME_PERSON_M):
                    del cand[k]
                    self.journal.append(TrackEvent(t, gid, "конфликт камер",
                                                   f"{other.camera}#{other.track_id} отвязан"))

    def _fix_crowd_swaps(self, tracks: list[LocalTrack], assigned: dict, t: float) -> None:
        """Три и больше человек прошли вплотную: ID могли перепутаться по кругу
        (A→B→C→A), попарная сверка такое не чинит. Для каждой группы близких
        людей на камере решаем задачу о назначениях по галереям re-ID."""
        by_cam: dict[str, list[LocalTrack]] = defaultdict(list)
        for tr in tracks:
            if tr.reid is not None and not tr.ir and not tr.chair:
                by_cam[tr.camera].append(tr)
        for cam_tracks in by_cam.values():
            for group in _clusters(cam_tracks, self.CROWD_RADIUS_M):
                gids = [assigned[tr.key] for tr in group]
                if len(group) < 2 or len(set(gids)) < len(gids):
                    continue
                cost = np.array([[self.states[g].reid_dist(tr.reid) for g in gids] for tr in group])
                rows, cols = linear_sum_assignment(cost)
                current = float(np.trace(cost))
                if current - float(cost[rows, cols].sum()) <= self.SWAP_MARGIN:
                    continue
                for r, c in zip(rows, cols):
                    if r != c:
                        old, new = gids[r], gids[c]
                        assigned[group[r].key] = new
                        self.journal.append(TrackEvent(t, new, "обменян", f"был у трека с {old}"))

    def _new(self, tr: LocalTrack, t: float, at_entrance: bool, born_t: Optional[float] = None) -> str:
        self._n += 1
        gid = f"{self.prefix}{self._n}"
        self.states[gid] = GlobalState(gid, tr.zone, t, tr.plan_xy, tr.chair,
                                       created_at_entrance=at_entrance,
                                       born_t=t if born_t is None else born_t)
        self.states[gid].remember(tr.reid)
        self.journal.append(TrackEvent(t, gid, "создан", tr.zone if at_entrance else f"в зале {tr.zone}"))
        return gid

    def _unmatched(self, tr: LocalTrack, t: float, taken: frozenset | set = frozenset()) -> str:
        if tr.zone == ZONE_ENTRANCE:
            return self._new(tr, t, at_entrance=True)
        # трек без пары посреди зала — ищем среди недавно потерянных
        wait = self.cfg.params.lost_id_wait_s
        best, best_cost, best_reid = None, self.cfg.params.max_match_cost, 1.0
        for s in self.states.values():
            if s.status == "active" or t - s.t_last > wait or s.gid in taken:
                continue
            rd = s.reid_dist(tr.reid)
            if tr.reid is not None and s.gallery and rd > self.RECOVER_MAX_REID:
                continue
            c = self._reid_w(tr) * rd + (0 if self._reachable(s, tr.zone, t, strict=True) else 10)
            if c < best_cost:
                best, best_cost, best_reid = s, c, rd
        companion = self._group_companion(tr, t)
        if companion and (best is None or best_reid > self.RECOVER_SURE_REID):
            return self._split_from(companion, tr, t)
        if best:
            self.journal.append(TrackEvent(t, best.gid, "восстановлен", f"{tr.camera}#{tr.track_id}"))
            return best.gid
        return self._new(tr, t, at_entrance=False)

    def _group_companion(self, tr: LocalTrack, t: float) -> Optional[GlobalState]:
        """Вошли вдвоём-втроём и слиплись в один бокс: новый трек появился рядом
        с ID, созданным на входе за последние 10 минут."""
        if tr.plan_xy is None:
            return None
        best = None
        for s in self.states.values():
            if (s.status != "active" or not s.created_at_entrance or s.xy is None
                    or t - s.born_t > self.GROUP_ENTRY_S or s.zone != tr.zone):
                continue
            d = math.dist(s.xy, tr.plan_xy)
            if d <= self.SAME_PERSON_M and (best is None or d < best[0]):
                best = (d, s)
        return best[1] if best else None

    def _split_from(self, companion: GlobalState, tr: LocalTrack, t: float) -> str:
        """Второй ID создаётся задним числом от момента входа компании: тики
        спутника с момента входа копируются новому ID."""
        gid = self._new(tr, t, at_entrance=True, born_t=companion.born_t)
        self.journal.append(TrackEvent(t, gid, "отделился", f"от {companion.gid}, вход задним числом"))
        copied = [Tick(t=x.t, gid=gid, zone=x.zone, dt=x.dt, cameras_ok=x.cameras_ok)
                  for x in self.ticks if x.gid == companion.gid and x.t >= companion.born_t]
        self.ticks.extend(copied)
        self.ticks.sort(key=lambda x: (x.t, x.gid))
        return gid

    def _age(self, t: float, seen: set[str]) -> None:
        blind = set(self.cfg.blind_zones)
        for s in self.states.values():
            if s.gid in seen or s.status != "active" or t - s.t_last < self.LOST_AFTER_S:
                continue
            near_blind = any(z in blind for z in self.cfg.transitions.get(s.zone, {}))
            s.status = "blind" if near_blind else "lost"
            self.journal.append(TrackEvent(t, s.gid, "потерян", s.status))


def _working(tr: LocalTrack) -> bool:
    return bool(tr.hands_chair or tr.hands_on)


def _far(a: LocalTrack, b: LocalTrack, limit: float) -> bool:
    if a.chair and b.chair:
        return a.chair != b.chair
    if a.plan_xy and b.plan_xy and a.zone == b.zone:
        return math.dist(a.plan_xy, b.plan_xy) > limit
    return False   # разные зоны на соседних камерах (дверной проём) — не судим


def _clusters(tracks: list[LocalTrack], radius: float) -> list[list[LocalTrack]]:
    """Связные группы людей ближе radius друг к другу (на плане)."""
    groups: list[list[LocalTrack]] = []
    left = [tr for tr in tracks if tr.plan_xy]
    while left:
        group = [left.pop()]
        i = 0
        while i < len(group):
            near = [o for o in left if math.dist(o.plan_xy, group[i].plan_xy) <= radius]
            for o in near:
                left.remove(o)
            group += near
            i += 1
        groups.append(group)
    return groups
