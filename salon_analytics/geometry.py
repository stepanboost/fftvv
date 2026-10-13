"""Геометрия: гомография камера → план зала, полигоны, кольцо вокруг кресла."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = tuple[float, float]


def fit_homography(image_pts: Sequence[Point], plan_pts: Sequence[Point]) -> np.ndarray:
    """DLT по 4+ точкам на полу. Возвращает 3x3 матрицу пиксели → метры."""
    if len(image_pts) < 4 or len(image_pts) != len(plan_pts):
        raise ValueError("нужно минимум 4 пары точек")
    rows = []
    for (x, y), (u, v) in zip(image_pts, plan_pts):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def project(h: np.ndarray, pt: Point) -> Point:
    x, y, w = h @ np.array([pt[0], pt[1], 1.0])
    return (x / w, y / w)


def reprojection_error(h: np.ndarray, image_pts: Sequence[Point], plan_pts: Sequence[Point]) -> float:
    errs = [math.dist(project(h, p), q) for p, q in zip(image_pts, plan_pts)]
    return max(errs) if errs else 0.0


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def in_ring(pt: Point, center: Point, inner: float, outer: float) -> bool:
    d = math.dist(pt, center)
    return inner <= d <= outer


def max_shift_px(reference: Sequence[Point], current: Sequence[Point]) -> float:
    """Утренняя сверка опорных точек: максимальный сдвиг, px."""
    return max((math.dist(a, b) for a, b in zip(reference, current)), default=0.0)
