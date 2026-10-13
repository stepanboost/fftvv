"""Раздел 6: выбор лучшего фото клиента и распознавание повторных клиентов."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .config import Params
from .models import FaceCandidate, Person, PhotoChoice, Role, Visit
from .similarity import cosine

STAFF_ROLES = (Role.MASTER, Role.ADMIN, Role.STAFF_UNKNOWN)

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


def _max_cos(v: list[float], gallery: Iterable[list[float]]) -> float:
    return max((cosine(v, g) for g in gallery), default=0.0)


def guard(cands: Iterable[FaceCandidate], own: list[list[float]], foreign: list[list[float]],
          p: Params) -> list[FaceCandidate]:
    """Не путать мастера и клиента на фото.

    С потолка мастер стоит вплотную к клиенту, и кадр «лица клиента» легко
    оказывается лицом мастера (или соседа в кресле рядом). Кадр отбрасывается,
    если похож на чужое лицо сильнее порога и не похож на своё заметно сильнее.
    """
    kept = []
    for c in cands:
        if c.embedding is not None and foreign:
            f = _max_cos(c.embedding, foreign)
            o = _max_cos(c.embedding, own) if own else 0.0
            if f > p.photo_foreign_cos and not (own and o > f + p.photo_own_margin):
                continue
        kept.append(c)
    return kept


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def apply_photos(visits: list[Visit], faces: dict[str, list[FaceCandidate]], p: Params,
                 history: Iterable[KnownFace] = (), persons: Optional[dict[str, Person]] = None) -> None:
    history = list(history)
    persons = persons or {}
    staff = [e for pr in persons.values() if pr.role in STAFF_ROLES for e in pr.face]
    for i, v in enumerate(visits, 1):
        own = [e for g in v.gids for e in persons.get(g, Person(g)).face]
        if not v.staff_client:
            # галерея клиента могла набрать лицо мастера — такие эмбеддинги не «свои»
            own = [e for e in own if _max_cos(e, staff) <= p.photo_foreign_cos]
        # чужие: все сотрудники и клиенты, которые были в салоне одновременно
        neighbours = [e for w in visits if w is not v and _overlaps(v.start, v.end, w.start, w.end)
                      for g in w.gids for e in persons.get(g, Person(g)).face]
        foreign = neighbours if v.staff_client else staff + neighbours
        cands = [c for g in v.gids for c in faces.get(g, []) if v.start <= c.t <= v.end]
        v.photo = choose_photo(guard(cands, own, foreign, p), p)
        v.client_label = f"№{i}"
        if not v.photo.has_face:
            v.review_reasons.append("лица нет")
        note, _ = repeat_status(v.photo, history, p, before=v.start)
        v.repeat_note = note


def master_photos(persons: dict[str, Person], faces: dict[str, list[FaceCandidate]],
                  visits: list[Visit], p: Params) -> dict[str, PhotoChoice]:
    """Фото мастера с камеры: только кадры его собственных ID, сверенные с
    лицами клиентов и других мастеров. Кадр, уже стоящий у клиента, не берётся."""
    client_paths = {c.path for v in visits if v.photo for c in [v.photo.best, *v.photo.spares] if c}
    client_faces = [e for v in visits for g in v.gids for e in persons.get(g, Person(g)).face]
    by_name: dict[str, list[str]] = {}
    for gid, pr in persons.items():
        if pr.role == Role.MASTER:
            by_name.setdefault(pr.name or gid, []).append(gid)
    out = {}
    for name, gids in by_name.items():
        own = [e for g in gids for e in persons[g].face]
        other_masters = [e for n, gs in by_name.items() if n != name for g in gs for e in persons[g].face]
        cands = [c for g in gids for c in faces.get(g, []) if c.path not in client_paths]
        out[name] = choose_photo(guard(cands, own, client_faces + other_masters, p), p)
    return out
