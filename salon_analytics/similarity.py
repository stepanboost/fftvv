"""Сравнение галерей эмбеддингов (лицо, одежда)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

Vec = Sequence[float]


def cosine(a: Vec, b: Vec) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def gallery_similarity(a: Sequence[Vec], b: Sequence[Vec]) -> Optional[float]:
    """Максимальный cosine между двумя галереями; None, если одна пуста."""
    if not a or not b:
        return None
    return max(cosine(x, y) for x in a for y in b)


def same_person(face_a: Sequence[Vec], face_b: Sequence[Vec],
                reid_a: Sequence[Vec], reid_b: Sequence[Vec],
                face_thr: float, reid_thr: float) -> tuple[bool, float]:
    """Решение «тот же человек». Лицо главнее одежды при любом конфликте признаков.

    Возвращает (решение, оценка сходства 0..1 для выбора лучшего кандидата).
    """
    f = gallery_similarity(face_a, face_b)
    if f is not None:
        return f > face_thr, f
    r = gallery_similarity(reid_a, reid_b)
    if r is not None:
        return r > reid_thr, r * 0.9   # одежда весит меньше лица
    return False, 0.0
