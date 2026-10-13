"""Дневной конвейер реального времени (Mac mini, launchd).

Каркас: камеры → детектор (YOLO-pose + BoT-SORT) → менеджер глобальных ID →
тики в JSONL. Вечером `python -m salon_analytics report` строит отчёт по тикам.
Детектор подключается через протокол `Detector`; на железе ещё не проверено.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..config import SalonConfig, load_config
from ..tracking.global_ids import GlobalIdManager, LocalTrack
from .rtsp import SilenceMonitor, frames, is_ir_frame, stream_url


class Detector(Protocol):
    def tracks(self, camera: str, frame, ir: bool) -> list[LocalTrack]:
        """Локальные треки одного кадра: поза, кресло, кольцо, руки, re-ID.

        В кадре бывает 3+ клиента и несколько мастеров, поэтому решения по
        кадру принимаются сразу для всех: `scene.assign_sitters` (кто в каком
        кресле, один к одному) и `scene.resolve_staff` (ring_chair и
        hands_chair каждого сотрудника). По одному человеку их не считать.
        """


def load_detector(cfg: SalonConfig) -> Detector:
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit("нужны ultralytics/onnxruntime/insightface: pip install -r requirements.txt "
                         "(раскомментировать CV-зависимости)") from e
    raise SystemExit("детектор YOLO-pose + BoT-SORT подключается на этапе 2")


def run(cfg: SalonConfig, host: str, out: Path, alert=print) -> None:
    import cv2

    detector = load_detector(cfg)
    manager = GlobalIdManager(cfg)
    monitor = SilenceMonitor(alert)
    latest: dict[str, tuple[float, object]] = {}

    def reader(cam_id: str, channel: int) -> None:
        url = stream_url(host, channel, sub=True)
        for frame in frames(lambda: cv2.VideoCapture(url), lambda c: (c.read()[1] if c.isOpened() else None)):
            now = time.time()
            latest[cam_id] = (now, frame)
            monitor.frame(cam_id, now)

    for cam in cfg.cameras.values():
        if cam.usable:
            threading.Thread(target=reader, args=(cam.camera_id, cam.channel), daemon=True).start()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        while True:
            t = time.time()
            monitor.tick(t)
            tracks: list[LocalTrack] = []
            for cam_id, (_, frame) in list(latest.items()):
                tracks += detector.tracks(cam_id, frame, is_ir_frame(frame))
            n = len(manager.ticks)
            manager.step(t, tracks)
            for tick in manager.ticks[n:]:
                f.write(json.dumps(asdict(tick), ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(max(0.0, cfg.params.match_interval_s - (time.time() - t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nvr", default="192.168.1.64")
    ap.add_argument("--out", default="data/ticks.jsonl")
    a = ap.parse_args()
    run(load_config(a.config), a.nvr, Path(a.out))


if __name__ == "__main__":
    main()
