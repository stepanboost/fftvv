"""Проверка камер на месте: запускать на компьютере в сети салона.

    NVR_PASSWORD=... python -m salon_analytics probe --nvr 192.168.1.64 --out probe/

По каждому каналу регистратора: подключается по RTSP, снимает кадр, меряет
разрешение, частоту, задержку, яркость, резкость, ИК-режим; рисует поверх кадра
разметку из salon.yaml (зелёный — кресла, синий — мойки, жёлтый — зоны,
красный — маски) и сохраняет PNG. Итог — probe/report.json и report.txt.
Нужен только ffmpeg (brew install ffmpeg) и numpy.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import SalonConfig
from ..markup import validate
from .rtsp import is_ir_frame, stream_url

COLORS = {"chair": (0, 200, 0), "wash": (255, 140, 0), "zone": (0, 220, 255), "mask": (0, 0, 255)}  # BGR


@dataclass
class CameraCheck:
    camera: str
    channel: int
    ok: bool = False
    error: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    connect_s: float = 0.0
    brightness: float = 0.0       # 0..255
    sharpness: float = 0.0        # дисперсия лапласиана; < 30 — мутно
    ir: bool = False
    snapshot: str = ""
    warnings: list[str] = field(default_factory=list)


def hide_password(text: str) -> str:
    """rtsp://admin:secret@host → rtsp://admin:***@host — пароль не попадает в отчёт."""
    return re.sub(r"(rtsp://[^:/@\s]+):[^@\s]*@", r"\1:***@", text)


def _ffprobe(url: str, timeout: float) -> dict:
    cmd = ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,codec_name,avg_frame_rate,r_frame_rate", "-of", "json", url]
    if not url.startswith("rtsp"):
        cmd.remove("-rtsp_transport"); cmd.remove("tcp")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or "ffprobe не ответил").strip().splitlines()[-1])
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise RuntimeError("нет видеопотока")
    return streams[0]


def _grab(url: str, w: int, h: int, timeout: float) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-rtsp_transport", "tcp", "-i", url, "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    if not url.startswith("rtsp"):
        cmd.remove("-rtsp_transport"); cmd.remove("tcp")
    out = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if out.returncode != 0 or len(out.stdout) < w * h * 3:
        raise RuntimeError((out.stderr.decode(errors="replace") or "кадр не получен").strip().splitlines()[-1])
    return np.frombuffer(out.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def _fps(rate: str) -> float:
    try:
        a, b = rate.split("/")
        return float(a) / float(b) if float(b) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def sharpness(frame: np.ndarray) -> float:
    g = frame.astype(np.float32).mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _line(img: np.ndarray, p0, p1, color, thick: int = 2) -> None:
    h, w = img.shape[:2]
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n).round().astype(int)
    ys = np.linspace(p0[1], p1[1], n).round().astype(int)
    for dx in range(-thick // 2, thick // 2 + 1):
        for dy in range(-thick // 2, thick // 2 + 1):
            x, y = np.clip(xs + dx, 0, w - 1), np.clip(ys + dy, 0, h - 1)
            img[y, x] = color


def draw_markup(frame: np.ndarray, cfg: SalonConfig, camera: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    cam = cfg.cameras[camera]

    def poly(points, color):
        px = [(x * (w - 1), y * (h - 1)) for x, y in points]
        for a, b in zip(px, px[1:] + px[:1]):
            _line(img, a, b, color, thick=max(2, w // 640))

    for chair in cfg.chairs.values():
        for v in chair.views:
            if v.camera == camera and v.polygon:
                poly(v.polygon, COLORS["wash" if chair.kind == "wash" else "chair"])
    for z in cam.zones.values():
        poly(z, COLORS["zone"])
    for m in cam.mirror_masks + cam.ignore_masks:
        poly(m, COLORS["mask"])
    return img


def save_png(img: np.ndarray, path: Path) -> None:
    h, w = img.shape[:2]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{w}x{h}", "-i", "-", str(path)], input=img.tobytes(), check=True)


def check_camera(cfg: SalonConfig, camera: str, url: str, out: Path, timeout: float = 15) -> CameraCheck:
    cam = cfg.cameras[camera]
    r = CameraCheck(camera, cam.channel)
    t0 = time.time()
    try:
        info = _ffprobe(url, timeout)
        r.width, r.height = int(info["width"]), int(info["height"])
        r.codec = info.get("codec_name", "")
        r.fps = round(_fps(info.get("avg_frame_rate", "")) or _fps(info.get("r_frame_rate", "")), 1)
        frame = _grab(url, r.width, r.height, timeout)
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        r.error = hide_password(str(e)) if not isinstance(e, subprocess.TimeoutExpired) else f"таймаут {timeout:.0f} с"
        return r
    r.connect_s = round(time.time() - t0, 2)
    r.ok = True
    r.brightness = round(float(frame.mean()), 1)
    r.sharpness = round(sharpness(frame), 1)
    r.ir = is_ir_frame(frame)
    path = out / f"{camera}.png"
    save_png(draw_markup(frame, cfg, camera), path)
    r.snapshot = str(path)

    if r.brightness < 25:
        r.warnings.append("кадр почти чёрный")
    elif r.brightness > 230:
        r.warnings.append("кадр засвечен")
    if r.sharpness < 30:
        r.warnings.append("мутно: объектив/фокус/запотевание")
    if r.fps and r.fps < 5:
        r.warnings.append(f"низкая частота {r.fps} к/с")
    if cam.resolution and tuple(cam.resolution) != (r.width, r.height):
        r.warnings.append(f"разрешение {r.width}x{r.height}, в разметке {cam.resolution[0]}x{cam.resolution[1]} "
                          "(координаты нормированы — проверить, что пропорции те же)")
    if cam.resolution and abs(cam.resolution[0] / cam.resolution[1] - r.width / r.height) > 0.02:
        r.warnings.append("другие пропорции кадра — разметка поедет")
    return r


def run_probe(cfg: SalonConfig, host: str, out: Path, user: str, password: str, sub: bool = True,
              sources: Optional[dict[str, str]] = None, timeout: float = 15) -> list[CameraCheck]:
    """sources: {камера: url или файл} — подменить источник (для проверки без регистратора)."""
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for cam_id, cam in sorted(cfg.cameras.items(), key=lambda kv: kv[1].channel):
        if sources is not None:
            if cam_id not in sources:
                continue
            url = sources[cam_id]
        else:
            url = stream_url(host, cam.channel, sub=sub, user=user, password=password)
        results.append(check_camera(cfg, cam_id, url, out, timeout))

    lines = []
    for r in results:
        if r.ok:
            mode = "ИК ч/б" if r.ir else "цвет"
            lines.append(f"{r.camera} (канал {r.channel}): OK {r.width}x{r.height} {r.codec} {r.fps} к/с, "
                         f"{mode}, яркость {r.brightness:.0f}, резкость {r.sharpness:.0f}, "
                         f"подключение {r.connect_s} с")
        else:
            lines.append(f"{r.camera} (канал {r.channel}): НЕТ СИГНАЛА — {r.error}")
        lines += [f"    ! {w}" for w in r.warnings]
    lines.append("")
    lines.append("Разметка:")
    lines += [f"    {'ОШИБКА' if i.level == 'error' else 'внимание'}: {i.where}: {i.text}" for i in validate(cfg)]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(text)
    return results
