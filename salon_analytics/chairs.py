"""Реестр физических кресел: слияние сигналов с нескольких камер.

Главное правило: считаем физические кресла, а не прямоугольники на камерах.
Кресло занято, если оно занято хотя бы на одной камере с хорошим обзором;
ID сидящего берётся с камеры, которая видит кресло лучше (priority).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .config import SalonConfig


@dataclass(frozen=True)
class ViewObservation:
    camera: str
    polygon_name: str
    occupied: bool
    occupant: Optional[str] = None   # глобальный ID, если известен
    view_ok: bool = True             # камера жива, ракурс не сдвинут


def view_index(cfg: SalonConfig) -> dict[tuple[str, str], tuple[str, int]]:
    """(камера, полигон) → (физическое кресло, приоритет)."""
    idx: dict[tuple[str, str], tuple[str, int]] = {}
    for chair in cfg.chairs.values():
        for v in chair.views:
            key = (v.camera, v.polygon_name)
            if key in idx:
                raise ValueError(f"полигон {key} привязан к двум креслам")
            idx[key] = (chair.chair_id, v.priority)
    return idx


def fuse(cfg: SalonConfig, observations: Iterable[ViewObservation]) -> dict[str, Optional[str]]:
    """Возвращает {кресло: gid сидящего, "?" если ID неизвестен, None если свободно}.

    Кресла, которые сейчас не видит ни одна живая камера, в ответ не попадают.
    """
    idx = view_index(cfg)
    by_chair: dict[str, list[tuple[int, ViewObservation]]] = {}
    for ob in observations:
        key = (ob.camera, ob.polygon_name)
        if ob.view_ok and key in idx:
            chair_id, prio = idx[key]
            by_chair.setdefault(chair_id, []).append((prio, ob))
    result: dict[str, Optional[str]] = {}
    for chair_id, views in by_chair.items():
        busy = sorted((v for v in views if v[1].occupied), key=lambda v: -v[0])
        if not busy:
            result[chair_id] = None
            continue
        known = [ob.occupant for _, ob in busy if ob.occupant]
        result[chair_id] = known[0] if known else "?"
    return result


def bind_moved_chairs(cfg: SalonConfig, camera: str,
                      detected_boxes: list[tuple[float, float, float, float]],
                      max_center_dist: float = 0.12) -> dict[str, list[tuple[float, float]]]:
    """Кресла на колёсах сдвигаются (на cam3 ночью 2-1 и 2-2 стоят не в своих
    полигонах). Детекция YOLO класса «chair» в кадре привязывается к рабочему
    месту из реестра по ближайшему центру; полигон вида на этот кадр берётся
    от детекции. Возвращает {кресло: текущий полигон}; не найденные — по разметке.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from .config import box_to_polygon

    views = [(c.chair_id, v) for c in cfg.chairs.values() for v in c.views
             if v.camera == camera and v.polygon]
    result = {cid: v.polygon for cid, v in views}
    if not views or not detected_boxes:
        return result

    def center(poly):
        return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))

    cost = np.array([[np.hypot(*(np.subtract(center(v.polygon), center(box_to_polygon(list(b))))))
                      for b in detected_boxes] for _, v in views])
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_center_dist:
            result[views[r][0]] = box_to_polygon(list(detected_boxes[c]))
    return result
