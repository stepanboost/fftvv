"""Потоки регистратора Hikvision DS-H208QA и контроль «камера молчит»."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional
from urllib.parse import quote

from ..models import Outage


def stream_url(host: str, channel: int, sub: bool = True, user: Optional[str] = None,
               password: Optional[str] = None, port: int = 554) -> str:
    """rtsp://user:pass@host:554/Streaming/Channels/101 (основной) или 102 (суб)."""
    user = user or os.environ.get("NVR_USER", "admin")
    password = password if password is not None else os.environ.get("NVR_PASSWORD", "")
    auth = f"{quote(user)}:{quote(password)}@" if password else f"{quote(user)}@"
    return f"rtsp://{auth}{host}:{port}/Streaming/Channels/{channel}0{2 if sub else 1}"


def playback_url(host: str, channel: int, start_local: str, end_local: str, **kw) -> str:
    """Клип из архива для карточки проверки. Время — местное регистратора, YYYYMMDDTHHMMSSZ."""
    base = stream_url(host, channel, sub=False, **kw).replace("/Streaming/Channels/", "/Streaming/tracks/")
    return f"{base}?starttime={start_local}&endtime={end_local}"


def frames(open_capture: Callable[[], object], read: Callable[[object], Optional[object]],
           backoff_s: tuple[float, ...] = (1, 2, 5, 10, 30)) -> Iterator[object]:
    """Бесконечный генератор кадров с переподключением при обрыве RTSP.

    open_capture/read внедряются, чтобы не тянуть OpenCV/PyAV в тесты:
    в проде это `cv2.VideoCapture(url)` и `cap.read()`.
    """
    attempt = 0
    while True:
        cap = open_capture()
        while True:
            frame = read(cap)
            if frame is None:
                break
            attempt = 0
            yield frame
        time.sleep(backoff_s[min(attempt, len(backoff_s) - 1)])
        attempt += 1


@dataclass
class SilenceMonitor:
    """Алерт в бот, если камера молчит > 2 мин; окна молчания — в «нет данных»."""
    alert: Callable[[str], None]
    silence_s: float = 120
    last_frame: dict[str, float] = field(default_factory=dict)
    silent_since: dict[str, float] = field(default_factory=dict)
    outages: list[Outage] = field(default_factory=list)

    def frame(self, camera: str, t: float) -> None:
        if camera in self.silent_since:
            self.outages.append(Outage(camera, self.silent_since.pop(camera), t))
            self.alert(f"{camera}: снова в эфире")
        self.last_frame[camera] = t

    def tick(self, t: float) -> None:
        for cam, last in self.last_frame.items():
            if cam not in self.silent_since and t - last > self.silence_s:
                self.silent_since[cam] = last
                self.alert(f"{cam}: нет кадров {int(t - last)} с")


def is_ir_frame(frame, max_channel_diff: float = 4.0) -> bool:
    """ИК-режим: каналы BGR почти совпадают. Ночью 08.10 все 8 камер были в ч/б."""
    import numpy as np

    f = np.asarray(frame, dtype=np.float32)
    if f.ndim < 3 or f.shape[2] < 3:
        return True
    small = f[::8, ::8, :3]
    return float(np.mean(np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2]))) / 2 < max_channel_diff
