"""Дневной конвейер реального времени (Mac mini, автозапуск через launchd).

    python -m salon_analytics.pipeline.live --config config/salon.yaml --nvr 192.168.1.64

Камеры → CVDetector (YOLO-pose + BoT-SORT/OSNet + YuNet/SFace) → менеджер
глобальных ID → файлы дня в data/ГГГГ-ММ-ДД/:
    ticks.jsonl    — где был каждый ID каждые 0.5 с
    faces.jsonl    — кадры лиц (с путём к картинке и эмбеддингом)
    faces/*.jpg    — сами кадры лиц
    persons.json   — галереи одежды по ID (сохраняется раз в минуту)
    outages.jsonl  — окна «камера молчала»
Вечером `python -m salon_analytics nightly` строит отчёт по этим файлам.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from ..config import SalonConfig, load_config
from ..report import salon_tz
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


def load_detector(cfg: SalonConfig, models_dir: Path, device: str, use_reid: bool = True):
    try:
        from .detector import CVDetector, DetectorSettings
    except ImportError as e:
        raise SystemExit("нужны CV-зависимости: pip install -r requirements-cv.txt") from e
    return CVDetector(cfg, DetectorSettings(models_dir=models_dir, device=device, use_reid=use_reid))


class DayFiles:
    def __init__(self, root: Path, cfg: SalonConfig):
        self.root, self.cfg = root, cfg
        self.day = ""
        self._open()

    def _open(self) -> None:
        self.day = datetime.now(salon_tz(self.cfg)).strftime("%Y-%m-%d")
        self.dir = self.root / self.day
        (self.dir / "faces").mkdir(parents=True, exist_ok=True)
        self.ticks = (self.dir / "ticks.jsonl").open("a", encoding="utf-8")
        self.faces = (self.dir / "faces.jsonl").open("a", encoding="utf-8")
        self.outages = (self.dir / "outages.jsonl").open("a", encoding="utf-8")

    def rotate_if_needed(self) -> bool:
        if datetime.now(salon_tz(self.cfg)).strftime("%Y-%m-%d") != self.day:
            for f in (self.ticks, self.faces, self.outages):
                f.close()
            self._open()
            return True
        return False


def save_persons(manager: GlobalIdManager, path: Path) -> None:
    data = [{"gid": s.gid, "role": "client", "reid": s.gallery,
             "created_at_entrance": s.created_at_entrance} for s in manager.states.values()]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def run(cfg: SalonConfig, host: str, data_root: Path, models_dir: Path, device: str = "mps",
        sources: Optional[dict[str, str]] = None, max_steps: Optional[int] = None, alert=print,
        use_reid: bool = True) -> Path:
    import cv2

    detector = load_detector(cfg, models_dir, device, use_reid)
    manager = GlobalIdManager(cfg)
    monitor = SilenceMonitor(alert)
    files = DayFiles(data_root, cfg)
    latest: dict[str, object] = {}
    user = os.environ.get("NVR_USER", "admin")
    password = os.environ.get("NVR_PASSWORD", "")

    def reader(cam_id: str, url: str) -> None:
        def open_cap():
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        def read(cap):
            ok, fr = cap.read() if cap.isOpened() else (False, None)
            return fr if ok else None

        for frame in frames(open_cap, read):
            latest[cam_id] = frame
            monitor.frame(cam_id, time.time())

    for cam in cfg.cameras.values():
        if not cam.usable:
            continue
        if sources is not None:
            if cam.camera_id not in sources:
                continue
            url = sources[cam.camera_id]
        else:
            url = stream_url(host, cam.channel, sub=True, user=user, password=password)
        threading.Thread(target=reader, args=(cam.camera_id, url), daemon=True).start()

    step, last_save = 0, 0.0
    n_written = 0
    while max_steps is None or step < max_steps:
        t = time.time()
        if files.rotate_if_needed():
            manager = GlobalIdManager(cfg)       # новый день — новые ID
            n_written = 0
        monitor.tick(t)
        while monitor.outages:
            files.outages.write(json.dumps(asdict(monitor.outages.pop(0))) + "\n")
        tracks: list[LocalTrack] = []
        for cam_id, frame in list(latest.items()):
            tracks += detector.tracks(cam_id, frame, is_ir_frame(frame))
        assigned = manager.step(t, tracks)

        for tick in manager.ticks[n_written:]:
            files.ticks.write(json.dumps(asdict(tick), ensure_ascii=False) + "\n")
        n_written = len(manager.ticks)
        for obs in detector.face_obs:
            gid = assigned.get((obs.camera, obs.track_id))
            if gid is None:
                continue
            path = files.dir / "faces" / f"{gid}_{int(t * 1000)}_{obs.camera}.jpg"
            cv2.imwrite(str(path), obs.face.crop)
            files.faces.write(json.dumps({
                "gid": gid, "t": t, "source": obs.source, "path": str(path.relative_to(files.dir)),
                "size_px": obs.face.size_px, "yaw_deg": obs.face.yaw_deg, "sharpness": obs.face.sharpness,
                "eyes_open": obs.face.score >= 0.8, "occluded": obs.face.score < 0.8,
                "embedding": obs.face.embedding}) + "\n")
        detector.face_obs.clear()
        for f in (files.ticks, files.faces, files.outages):
            f.flush()
        if t - last_save > 60 or (max_steps is not None and step == max_steps - 1):
            save_persons(manager, files.dir / "persons.json")
            last_save = t
        step += 1
        time.sleep(max(0.0, cfg.params.match_interval_s - (time.time() - t)))
    save_persons(manager, files.dir / "persons.json")
    return files.dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nvr", default=os.environ.get("NVR_HOST", "192.168.1.64"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--device", default="mps", help="mps на Mac (Apple GPU), cpu — везде")
    ap.add_argument("--source", action="append", default=[], metavar="CAM=URL",
                    help="вместо регистратора — файл/URL для камеры (для проверки на записи)")
    ap.add_argument("--steps", type=int, default=None, help="остановиться после N шагов (проверка)")
    ap.add_argument("--no-reid", action="store_true", help="без модели одежды OSNet (только для проверки)")
    a = ap.parse_args()
    sources = dict(s.split("=", 1) for s in a.source) or None
    out = run(load_config(a.config), a.nvr, Path(a.data), Path(a.models), a.device, sources, a.steps,
              use_reid=not a.no_reid)
    print(f"готово: {out}")


if __name__ == "__main__":
    main()
