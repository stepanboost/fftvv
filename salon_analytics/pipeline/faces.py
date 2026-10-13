"""Лица: YuNet (детектор) + SFace (эмбеддинг) из OpenCV Zoo.

Лицензия Apache-2.0 — можно в коммерции, в отличие от моделей InsightFace
(buffalo_l — только некоммерческие). Порог «тот же человек» для SFace по
документации OpenCV — cosine ≥ 0.363.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
URLS = {YUNET: ZOO + "face_detection_yunet/" + YUNET, SFACE: ZOO + "face_recognition_sface/" + SFACE}


@dataclass
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h в пикселях
    score: float
    landmarks: np.ndarray                     # 5 точек: глаза, нос, углы рта
    embedding: list[float]
    crop: np.ndarray
    sharpness: float                          # 0..1
    yaw_deg: float

    @property
    def size_px(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)


def estimate_yaw(lm: np.ndarray) -> float:
    """Грубый поворот головы по 5 точкам: насколько нос смещён от середины глаз."""
    (rx, ry), (lx, ly), (nx, ny) = lm[0], lm[1], lm[2]
    eye_mid = (rx + lx) / 2
    eye_dist = abs(lx - rx) or 1.0
    return float(np.clip((nx - eye_mid) / eye_dist * 90.0, -90, 90))


def crop_sharpness(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(min(1.0, lap.var() / 400.0))


class FaceEngine:
    def __init__(self, models_dir: str | Path, score_thr: float = 0.7):
        import cv2

        self.cv2 = cv2
        d = Path(models_dir)
        self.det = cv2.FaceDetectorYN.create(str(d / YUNET), "", (320, 320), score_thr)
        self.rec = cv2.FaceRecognizerSF.create(str(d / SFACE), "")

    def embed_crop(self, frame: np.ndarray, row: np.ndarray) -> list[float]:
        aligned = self.rec.alignCrop(frame, row)
        v = self.rec.feature(aligned).flatten().astype(float)
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(frame)
        out = []
        for row in rows if rows is not None else []:
            x, y, fw, fh = (float(v) for v in row[:4])
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
            crop = frame[y0:y1, x0:x1].copy()
            lm = row[4:14].reshape(5, 2)
            out.append(Face((x, y, fw, fh), float(row[14]), lm, self.embed_crop(frame, row), crop,
                            crop_sharpness(crop.mean(axis=2) if crop.ndim == 3 else crop), estimate_yaw(lm)))
        return out

    def embed_image(self, path: str | Path) -> Optional[list[float]]:
        """Эмбеддинг самого крупного лица на фото (для регистрации мастеров)."""
        img = self.cv2.imread(str(path))
        if img is None:
            return None
        faces = self.detect(img)
        if not faces:
            return None
        return max(faces, key=lambda f: f.size_px).embedding


def download_models(models_dir: str | Path) -> list[Path]:
    import urllib.request

    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    got = []
    for name, url in URLS.items():
        p = d / name
        if not p.exists() or p.stat().st_size < 10_000:
            urllib.request.urlretrieve(url, p)
        got.append(p)
    return got
