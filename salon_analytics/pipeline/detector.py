"""Детектор кадра: YOLO-pose (Ultralytics) → BoT-SORT + OSNet (BoxMOT) → наши правила.

Связка взята из открытых проектов, ничего своего в детекции и трекинге нет:
- люди и скелеты: Ultralytics YOLO11-pose (на Mac — экспорт в CoreML);
- кресла (сдвинутые на колёсах): Ultralytics YOLO11, класс COCO «chair»;
- треки и признаки одежды: BoxMOT BotSort с ReID-моделью OSNet;
- лица: OpenCV YuNet + SFace (faces.py).
Наше поверх: кто в каком кресле (scene.assign_sitters), к какому клиенту
относится мастер (scene.resolve_staff), какое лицо чьё (один к одному).

Координаты разметки нормированы 0..1, скелеты тоже приводятся к 0..1.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..chairs import bind_moved_chairs
from ..config import SalonConfig
from ..geometry import fit_homography, point_in_polygon, project
from ..models import ZONE_ENTRANCE
from ..pose import is_seated, pelvis
from ..scene import assign_sitters, resolve_staff
from ..tracking.global_ids import LocalTrack
from .faces import Face, FaceEngine

COCO_PERSON, COCO_CHAIR = 0, 56
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class DetectorSettings:
    models_dir: Path = Path("models")
    pose_weights: str = "yolo11n-pose.pt"       # на Mac: yolo11n-pose.mlpackage
    chair_weights: str = "yolo11n.pt"
    reid_weights: str = "osnet_x0_25_msmt17.pt"  # BoxMOT скачает сам
    device: str = "cpu"                          # на Mac M4: "mps"
    use_reid: bool = True
    person_conf: float = 0.4
    kp_conf: float = 0.3
    chair_every: int = 50          # кресла двигаются редко — детекция раз в N кадров
    face_every: int = 5
    still_window: int = 8          # кадров для оценки неподвижности
    still_norm: float = 0.02       # без калибровки: смещение таза < 2% кадра
    ring_norm: tuple[float, float] = (0.03, 0.22)   # без калибровки: «кольцо» в долях кадра
    hands_radius: float = 0.06     # запястье у головы клиента, доля кадра
    fps: int = 10


@dataclass
class FaceObs:
    """Лицо, однозначно привязанное к локальному треку (до глобального ID)."""
    camera: str
    track_id: int
    face: Face
    source: str        # entrance | ceiling


@dataclass
class _CamState:
    tracker: object
    homography: Optional[np.ndarray]
    frame_no: int = 0
    chair_polys: dict[str, list] = field(default_factory=dict)
    pelvis_hist: dict[int, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=8)))


def _feet(kp, box_n) -> tuple[float, float]:
    ankles = [p for p in (kp[L_ANKLE], kp[R_ANKLE]) if p]
    if ankles:
        return (sum(p[0] for p in ankles) / len(ankles), max(p[1] for p in ankles))
    return ((box_n[0] + box_n[2]) / 2, box_n[3])


def _in_any(pt, polys) -> bool:
    return any(point_in_polygon(pt, p) for p in polys)


def match_faces_to_people(faces: list[Face], people: dict[int, tuple], kps: dict[int, list],
                          w: int, h: int) -> dict[int, Face]:
    """Лицо → человек строго один к одному.

    Кандидат — только если центр лица внутри верхней половины бокса человека;
    стоимость — расстояние от центра лица до носа/головы по скелету. Два
    человека не могут получить одно лицо, одно лицо не уйдёт двоим — так в
    тесноте у кресел лицо мастера не приклеится к клиенту.
    """
    ids = list(people)
    if not faces or not ids:
        return {}
    big = 1e6
    cost = np.full((len(faces), len(ids)), big)
    for i, f in enumerate(faces):
        cx, cy = f.center[0] / w, f.center[1] / h
        for j, tid in enumerate(ids):
            x0, y0, x1, y1 = people[tid]
            if not (x0 <= cx <= x1 and y0 <= cy <= y0 + (y1 - y0) * 0.5):
                continue
            head = [p for p in kps[tid][:5] if p]
            hx, hy = (np.mean([p[0] for p in head]), np.mean([p[1] for p in head])) if head \
                else ((x0 + x1) / 2, y0 + (y1 - y0) * 0.12)
            cost[i, j] = math.hypot(cx - hx, cy - hy)
    rows, cols = linear_sum_assignment(cost)
    return {ids[c]: faces[r] for r, c in zip(rows, cols) if cost[r, c] < big}


class CVDetector:
    """Реализация протокола live.Detector."""

    def __init__(self, cfg: SalonConfig, s: DetectorSettings | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.s = s or DetectorSettings()
        md = Path(self.s.models_dir)
        self.pose = YOLO(str(md / self.s.pose_weights) if (md / self.s.pose_weights).exists() else self.s.pose_weights)
        self.chairs = YOLO(str(md / self.s.chair_weights) if (md / self.s.chair_weights).exists() else self.s.chair_weights)
        self.faces = FaceEngine(md)
        self.cams: dict[str, _CamState] = {}
        self.face_obs: list[FaceObs] = []      # забирает live-цикл после manager.step
        self.reid_by_track: dict[tuple[str, int], list[float]] = {}

    # --- служебное ---------------------------------------------------------------
    def _make_tracker(self):
        from boxmot import BotSort

        # камеры неподвижны — компенсация движения камеры не нужна
        kw = dict(device=self.s.device, half=False, frame_rate=self.s.fps, use_cmc=False)
        if self.s.use_reid:
            md = Path(self.s.models_dir)
            w = md / self.s.reid_weights
            return BotSort(reid_weights=w if w.exists() else Path(self.s.reid_weights), **kw)
        return BotSort(use_embeddings=False, **kw)

    def _cam(self, camera: str) -> _CamState:
        if camera not in self.cams:
            cam = self.cfg.cameras[camera]
            h = None
            pts = cam.homography_points
            if len(pts) >= 4:
                h = fit_homography([tuple(p["image"]) for p in pts], [tuple(p["plan"]) for p in pts])
            polys = {c.chair_id: v.polygon for c in self.cfg.chairs.values() for v in c.views
                     if v.camera == camera and v.polygon}
            self.cams[camera] = _CamState(self._make_tracker(), h, chair_polys=polys)
        return self.cams[camera]

    def _zone(self, camera: str, feet) -> str:
        cam = self.cfg.cameras[camera]
        entrance = cam.zones.get("entrance")
        if entrance and point_in_polygon(feet, entrance):
            return ZONE_ENTRANCE
        return cam.zone

    # --- основной вызов ------------------------------------------------------------
    def tracks(self, camera: str, frame: np.ndarray, ir: bool) -> list[LocalTrack]:
        st = self._cam(camera)
        st.frame_no += 1
        cam = self.cfg.cameras[camera]
        h, w = frame.shape[:2]

        if st.frame_no % self.s.chair_every == 1:
            r = self.chairs(frame, classes=[COCO_CHAIR], conf=0.35, verbose=False, device=self.s.device)[0]
            boxes = [tuple(b) for b in r.boxes.xyxyn.cpu().numpy().tolist()]
            st.chair_polys = bind_moved_chairs(self.cfg, camera, boxes)

        r = self.pose(frame, classes=[COCO_PERSON], conf=self.s.person_conf, verbose=False, device=self.s.device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            st.tracker.update(np.empty((0, 6)), frame)
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        kxy = r.keypoints.xyn.cpu().numpy()
        kconf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])

        # зеркала и манекены: людей там нет, только отражения
        keep = []
        for i, b in enumerate(xyxy):
            c = ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h)
            if not _in_any(c, cam.mirror_masks + cam.ignore_masks):
                keep.append(i)
        if not keep:
            st.tracker.update(np.empty((0, 6)), frame)
            return []
        dets = np.concatenate([xyxy[keep], conf[keep, None], np.zeros((len(keep), 1))], axis=1)
        out = st.tracker.update(dets, frame)
        feats = {}
        for t in getattr(st.tracker, "active_tracks", []):
            tid = getattr(t, "id", getattr(t, "track_id", None))
            if tid is not None and getattr(t, "smooth_feat", None) is not None:
                feats[int(tid)] = np.asarray(t.smooth_feat, dtype=float).tolist()

        people: dict[int, tuple] = {}
        kps: dict[int, list] = {}
        out = np.asarray(out)
        for row in (out.reshape(-1, out.shape[-1]) if out.size else []):
            tid, det_i = int(row[4]), int(row[7])
            src = keep[det_i]
            people[tid] = (row[0] / w, row[1] / h, row[2] / w, row[3] / h)
            kps[tid] = [tuple(kxy[src, k]) if kconf[src, k] >= self.s.kp_conf else None for k in range(17)]

        # неподвижность таза
        shift: dict[int, float] = {}
        for tid, kp in kps.items():
            pv = pelvis(kp)
            hist = st.pelvis_hist[tid]
            if pv:
                hist.append(pv)
            shift[tid] = math.dist(hist[0], hist[-1]) if len(hist) >= 2 else 1.0

        # кто сидит и в каком кресле — по всему кадру сразу
        sitting = {tid: kp for tid, kp in kps.items()
                   if any(is_seated(kp, poly, shift[tid], max_shift_m=self.s.still_norm)
                          for poly in st.chair_polys.values())}
        seat = assign_sitters(sitting, st.chair_polys)

        # к какому клиенту относится каждый стоящий
        standing = [tid for tid in kps if tid not in seat]
        if st.homography is not None and all(self.cfg.chairs[c].plan_xy for c in st.chair_polys):
            to_plan = lambda p: project(st.homography, p)
            staff_xy = {t: to_plan(_feet(kps[t], people[t])) for t in standing}
            chair_xy = {c: self.cfg.chairs[c].plan_xy for c in st.chair_polys}
            ring = (self.cfg.params.ring_inner_m, self.cfg.params.ring_outer_m)
        else:   # камера не откалибрована — то же правило в долях кадра
            staff_xy = {t: _feet(kps[t], people[t]) for t in standing}
            chair_xy = {c: (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
                        for c, poly in st.chair_polys.items()}
            ring = self.s.ring_norm
        att = resolve_staff(staff_xy, {t: kps[t] for t in standing}, seat,
                            {t: kps[t] for t in seat}, chair_xy, ring, self.s.hands_radius)

        # лица — один к одному с людьми
        if st.frame_no % self.s.face_every == 1:
            src = "entrance" if cam.zone == ZONE_ENTRANCE else "ceiling"
            for tid, face in match_faces_to_people(self.faces.detect(frame), people, kps, w, h).items():
                fsrc = "entrance" if self._zone(camera, _feet(kps[tid], people[tid])) == ZONE_ENTRANCE else src
                self.face_obs.append(FaceObs(camera, tid, face, fsrc))

        result = []
        for tid in kps:
            feet = _feet(kps[tid], people[tid])
            plan = project(st.homography, feet) if st.homography is not None else None
            a = att.get(tid)
            result.append(LocalTrack(
                camera=camera, track_id=tid, zone=self._zone(camera, feet),
                plan_xy=None if tid in seat else plan, reid=feats.get(tid),
                chair=seat.get(tid), ring_chair=a.ring_chair if a else None,
                hands_chair=a.hands_chair if a else None, ir=ir))
        return result
