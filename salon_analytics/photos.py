"""Раздел 6: выбор лучшего фото клиента и распознавание повторных клиентов."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .config import Params
from .models import FaceCandidate, PhotoChoice, Visit
from .similarity import cosine

# ценность источника: камера входа/кассы анфас > зеркало > потолок
SOURCE_WEIGHT = {"entrance": 1.0, "mirror": 0.8, "ceiling": 0.5}


def passes(c: FaceCandidate, p: Params) -> bool:
    return (c.source in SOURCE_WEIGHT and c.size_px >= p.face_min_px
            and abs(c.yaw_deg) < p.face_max_yaw_deg and c.sharpness >= p.face_min_sharpness
            and c.eyes_open and not c.occluded)


def quality(c: FaceCandidate, p: Params) -> float:
    size = min(1.0, c.size_px / (2 * p.face_min_px))
    pose = max(0.0, 1 - abs(c.yaw_deg) / 90)
    q = 0.35 * size + 0.3 * pose + 0.35 * c.sharpness
    return q * SOURCE_WEIGHT.get(c.source, 0.2)


def choose_photo(cands: Iterable[FaceCandidate], p: Params) -> PhotoChoice:
    cands = list(cands)
    good = sorted((c for c in cands if passes(c, p)), key=lambda c: -quality(c, p))
    if good:
        return PhotoChoice(best=good[0], spares=good[1:1 + p.photo_spares], has_face=True)
    bodies = sorted((c for c in cands if c.source == "body"), key=lambda c: -c.sharpness)
    return PhotoChoice(best=bodies[0] if bodies else None, has_face=False)


@dataclass(frozen=True)
class KnownFace:
    client_label: str
    date: str            # ДД.ММ последнего визита
    embedding: list[float]
    t: float


def repeat_status(choice: PhotoChoice, history: Iterable[KnownFace], p: Params,
                  before: float) -> tuple[str, Optional[KnownFace]]:
    """«повторный» только при хорошем лице и cosine > 0.6, иначе «возможно повторный»."""
    if not choice.best or not choice.best.embedding:
        return "", None
    retention = p.retention_days * 86400
    best, best_sim = None, 0.0
    for k in history:
        if k.t >= before or before - k.t > retention:
            continue
        sim = cosine(choice.best.embedding, k.embedding)
        if sim > best_sim:
            best, best_sim = k, sim
    if best is None:
        return "", None
    if choice.has_face and best_sim > p.repeat_face_cos:
        return f"повторный, был {best.date}", best
    if best_sim > p.possible_repeat_face_cos:
        return f"возможно повторный ({best.date})", best
    return "", None


def apply_photos(visits: list[Visit], faces: dict[str, list[FaceCandidate]], p: Params,
                 history: Iterable[KnownFace] = ()) -> None:
    history = list(history)
    for i, v in enumerate(visits, 1):
        cands = [c for g in v.gids for c in faces.get(g, []) if v.start <= c.t <= v.end]
        v.photo = choose_photo(cands, p)
        v.client_label = f"№{i}"
        if not v.photo.has_face:
            v.review_reasons.append("лица нет")
        note, _ = repeat_status(v.photo, history, p, before=v.start)
        v.repeat_note = note
