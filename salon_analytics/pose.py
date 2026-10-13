"""Эвристики по скелету YOLO-pose (COCO-17 ключевых точек)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .geometry import Point, point_in_polygon

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 13, 14

Keypoints = Sequence[Optional[Point]]   # None — точка не видна


def _mid(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


def pelvis(kp: Keypoints) -> Optional[Point]:
    return _mid(kp[L_HIP], kp[R_HIP])


def thigh_angle_deg(kp: Keypoints) -> Optional[float]:
    """Средний угол бедра (таз→колено) к горизонтали кадра."""
    angles = []
    for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if kp[hip] and kp[knee]:
            dx = kp[knee][0] - kp[hip][0]
            dy = kp[knee][1] - kp[hip][1]
            angles.append(abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))))
    return sum(angles) / len(angles) if angles else None


def is_seated(kp: Keypoints, chair_polygon: Sequence[Point], plan_shift_m: float,
              max_thigh_deg: float = 35.0, max_shift_m: float = 0.20) -> bool:
    """Сидит: бёдра примерно горизонтальны, таз в полигоне кресла, неподвижен."""
    p = pelvis(kp)
    if p is None or not point_in_polygon(p, chair_polygon):
        return False
    if plan_shift_m >= max_shift_m:
        return False
    angle = thigh_angle_deg(kp)
    # с потолочной камеры колени часто не видны: тогда решают таз и неподвижность
    return angle is None or angle <= max_thigh_deg


def head_center(kp: Keypoints) -> Optional[Point]:
    pts = [kp[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kp[i]]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def hands_on_head(master_kp: Keypoints, client_kp: Keypoints, radius_px: float) -> bool:
    """Мастер работает руками: хотя бы одно запястье у головы клиента."""
    head = head_center(client_kp)
    if head is None:
        return False
    for w in (L_WRIST, R_WRIST):
        if master_kp[w] and math.dist(master_kp[w], head) <= radius_px:
            return True
    return False
