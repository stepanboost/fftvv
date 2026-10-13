"""Проверка разметки камер: то, что на реальных кадрах уже ломалось.

Примеры с кадров 08.10.2026: полигон «??-8» на cam8 уходил за кадр, бокс
кресла 2-4 перекрывает мойку M-1 на cam2, cam1 размыта, cam5 не работает.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import SalonConfig
from .geometry import Point, point_in_polygon

VALID_ID = re.compile(r"^[0-9A-Za-zА-Яа-я]+(-[0-9A-Za-z]+)?$")


@dataclass(frozen=True)
class Issue:
    level: str        # error | warning
    where: str
    text: str


def overlap_ratio(a: Sequence[Point], b: Sequence[Point], n: int = 40) -> float:
    """Доля площади a, лежащая внутри b (оценка по сетке n×n в рамке a)."""
    xs, ys = [p[0] for p in a], [p[1] for p in a]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    inside_a = both = 0
    for i in range(n):
        for j in range(n):
            pt = (x0 + (i + 0.5) * (x1 - x0) / n, y0 + (j + 0.5) * (y1 - y0) / n)
            if point_in_polygon(pt, a):
                inside_a += 1
                both += point_in_polygon(pt, b)
    return both / inside_a if inside_a else 0.0


def validate(cfg: SalonConfig, overlap_warn: float = 0.15) -> list[Issue]:
    issues: list[Issue] = []
    for cam in cfg.cameras.values():
        if cam.quality == "off":
            issues.append(Issue("error", cam.camera_id, "камера не работает / смотрит не туда"))
        elif cam.quality == "poor":
            issues.append(Issue("warning", cam.camera_id, "плохое качество кадра — почистить объектив, понизить приоритет"))
        for name, poly in cam.zones.items():
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in poly):
                issues.append(Issue("error", f"{cam.camera_id}/{name}", "зона за пределами кадра"))

    per_cam: dict[str, list[tuple[str, str, list[Point]]]] = {}
    for chair in cfg.chairs.values():
        if not VALID_ID.match(chair.chair_id) or "?" in chair.chair_id:
            issues.append(Issue("error", chair.chair_id, "ID кресла не определён"))
        if not chair.views:
            issues.append(Issue("error", chair.chair_id, "кресло не видно ни на одной камере"))
        usable = [v for v in chair.views if cfg.cameras[v.camera].quality == "ok"]
        if chair.views and not usable:
            issues.append(Issue("warning", chair.chair_id, "видно только на камерах плохого качества"))
        for v in chair.views:
            where = f"{v.camera}/{v.polygon_name} → {chair.chair_id}"
            if not v.polygon:
                issues.append(Issue("warning", where, "полигон не размечен"))
                continue
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in v.polygon):
                issues.append(Issue("error", where, "полигон кресла за пределами кадра"))
            cam = cfg.cameras[v.camera]
            for mask in cam.mirror_masks + cam.ignore_masks:
                if overlap_ratio(v.polygon, mask) > overlap_warn:
                    issues.append(Issue("warning", where, "полигон кресла под маской зеркала/игнора"))
            per_cam.setdefault(v.camera, []).append((chair.chair_id, chair.kind, v.polygon))

    for cam_id, items in per_cam.items():
        for i, (ida, _, pa) in enumerate(items):
            for idb, _, pb in items[i + 1:]:
                r = max(overlap_ratio(pa, pb), overlap_ratio(pb, pa))
                if r > overlap_warn:
                    issues.append(Issue("warning", cam_id,
                                        f"{ida} и {idb} перекрываются на {r:.0%}: кресло по тазу, мойка по голове"))

    if not any(c.usable and (c.zone == "entrance" or "entrance" in c.zones) for c in cfg.cameras.values()):
        issues.append(Issue("error", "вход", "нет живой камеры с зоной входа — новые ID создавать негде"))
    elif not any(c.usable and c.zone == "entrance" for c in cfg.cameras.values()):
        issues.append(Issue("warning", "вход", "вход виден только с потолочной камеры зала — лица анфас не будет"))
    return issues
