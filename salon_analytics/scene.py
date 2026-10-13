"""Один кадр — много людей: кто в каком кресле и с кем работает каждый мастер.

В зале 2 три кресла стоят в ~1.2 м друг от друга, поэтому в кадре легко 3+
клиентов и 2+ мастера одновременно. Полигоны соседних кресел с потолочной
камеры перекрываются, кольца 0.4–1.2 м — тоже. Решения принимаются сразу для
всех людей кадра (венгерский алгоритм), а не по одному человеку.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Point, point_in_polygon
from .pose import L_WRIST, R_WRIST, Keypoints, head_center, pelvis

BIG = 1e6


def _center(poly: Sequence[Point]) -> Point:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def assign_sitters(sitting: dict[str, Keypoints], chairs: dict[str, Sequence[Point]]) -> dict[str, str]:
    """{человек: кресло} один к одному.

    Таз может попасть в два перекрывающихся полигона; тогда решает расстояние
    до центра полигона, и двое не могут оказаться в одном кресле.
    """
    people = [p for p, kp in sitting.items() if pelvis(kp)]
    chair_ids = list(chairs)
    if not people or not chair_ids:
        return {}
    cost = np.full((len(people), len(chair_ids)), BIG)
    for i, pid in enumerate(people):
        pv = pelvis(sitting[pid])
        for j, cid in enumerate(chair_ids):
            if point_in_polygon(pv, chairs[cid]):
                cost[i, j] = math.dist(pv, _center(chairs[cid]))
    rows, cols = linear_sum_assignment(cost)
    return {people[r]: chair_ids[c] for r, c in zip(rows, cols) if cost[r, c] < BIG}


@dataclass(frozen=True)
class StaffAttention:
    ring_chair: Optional[str]    # ближайшее занятое кресло, в кольце которого стоит
    hands_chair: Optional[str]   # у чьей головы руки


def resolve_staff(staff_xy: dict[str, Optional[Point]], staff_kp: dict[str, Keypoints],
                  sitters: dict[str, str], sitter_kp: dict[str, Keypoints],
                  chair_xy: dict[str, Point], ring: tuple[float, float],
                  hands_radius: float) -> dict[str, StaffAttention]:
    """Для каждого стоящего сотрудника — кресло «рядом» и кресло «руками».

    staff_xy — точка ног на плане (м), chair_xy — центры кресел на плане,
    staff_kp / sitter_kp — скелеты в координатах кадра, sitters — {человек: кресло}.
    «Рядом» засчитывается только одному креслу — ближайшему занятому, иначе
    мастер у 2-2 накручивал бы время клиентам в 2-1 и 2-3.
    «Руками» — кресло, голова сидящего в котором ближе всех к запястью, в
    пределах hands_radius. Это главный признак для голосования.
    """
    occupied = {chair: person for person, chair in sitters.items()}
    heads = {chair: head_center(sitter_kp[p]) for chair, p in occupied.items() if p in sitter_kp}
    out: dict[str, StaffAttention] = {}
    for sid in staff_kp.keys() | staff_xy.keys():
        ring_chair = None
        xy = staff_xy.get(sid)
        if xy is not None:
            best = None
            for chair in occupied:
                c = chair_xy.get(chair)
                if c is None:
                    continue
                d = math.dist(xy, c)
                if ring[0] <= d <= ring[1] and (best is None or d < best[0]):
                    best = (d, chair)
            ring_chair = best[1] if best else None

        hands_chair = None
        kp = staff_kp.get(sid)
        if kp is not None:
            best = None
            for w in (L_WRIST, R_WRIST):
                if not kp[w]:
                    continue
                for chair, head in heads.items():
                    if head is None or occupied[chair] == sid:
                        continue
                    d = math.dist(kp[w], head)
                    if d <= hands_radius and (best is None or d < best[0]):
                        best = (d, chair)
            hands_chair = best[1] if best else None
        out[sid] = StaffAttention(ring_chair, hands_chair)
    return out
