"""Конфигурация салона: камеры, зоны, реестр физических кресел, пороги."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

MIN = 60.0


@dataclass
class Params:
    """Все пороги из ТЗ. Имена совпадают с ключами `params:` в salon.yaml."""
    # 4.2 посадки
    min_sitting_s: float = 5 * MIN
    min_sitting_barber_s: float = 3 * MIN
    min_master_near_s: float = 2 * MIN
    sitting_gap_merge_s: float = 3 * MIN
    short_sitting_review_s: float = 3 * MIN       # 3–5 мин → очередь проверки
    sat_without_master_review_s: float = 10 * MIN
    still_max_shift_m: float = 0.20
    ring_inner_m: float = 0.4
    ring_outer_m: float = 1.2
    # 4.3 визиты
    visit_close_absence_s: float = 20 * MIN
    lost_id_wait_s: float = 30 * MIN
    visit_max_s: float = 9 * 3600
    stitch_max_gap_s: float = 30 * MIN
    # пороги лица — под SFace (OpenCV Zoo): «тот же человек» от 0.363
    face_same_cos: float = 0.40
    stitch_face_cos: float = 0.40
    stitch_reid_cos: float = 0.7
    repeat_face_cos: float = 0.55
    possible_repeat_face_cos: float = 0.40
    # 5 мастера
    second_master_share: float = 0.20
    staff_exit_min_s: float = 2 * MIN
    staff_interval_split_s: float = 60 * MIN
    pinned_chair_bonus: float = 0.10
    booking_bonus: float = 0.10
    booking_window_s: float = 30 * MIN
    admin_votes: bool = False
    master_min_face_frames: int = 2      # мастер опознан, если лицо совпало хотя бы на 2 кадрах
    staff_presence_s: float = 3 * 3600   # в салоне > 3 ч и лицом не опознан — «сотрудник?»
    # 6 фото
    face_min_px: float = 80
    face_max_yaw_deg: float = 30
    face_min_sharpness: float = 0.4
    photo_spares: int = 3
    photo_foreign_cos: float = 0.40    # кадр похож на чужое лицо (мастер ↔ клиент) — выбросить
    photo_own_margin: float = 0.1      # ...если на своё лицо он похож не заметно сильнее
    retention_days: int = 90
    # 8 отчёт
    review_conf: float = 0.7
    crm_tolerance: int = 1
    # 4.1 глобальный менеджер
    match_interval_s: float = 0.5
    max_speed_mps: float = 2.0
    reid_weight: float = 2.0
    impossible_penalty: float = 100.0
    max_match_cost: float = 3.0
    calibration_shift_alert_px: float = 15
    reid_weight_ir: float = 0.5        # в ИК (ч/б) цвет одежды почти бесполезен
    chair_bind_max_m: float = 1.0      # кресло на колёсах: до рабочего места не дальше, м


@dataclass
class ChairView:
    camera: str
    polygon_name: str
    priority: int = 1   # больше — камера видит кресло лучше
    polygon: list[tuple[float, float]] = field(default_factory=list)   # нормировано 0..1


@dataclass
class Chair:
    chair_id: str
    hall: str
    kind: str = "regular"       # regular | barber | wash
    views: list[ChairView] = field(default_factory=list)
    plan_xy: Optional[tuple[float, float]] = None   # центр кресла на плане, м
    pinned_master: Optional[str] = None


@dataclass
class Camera:
    camera_id: str
    channel: int
    zone: str
    hall: Optional[str] = None
    homography_points: list[dict[str, list[float]]] = field(default_factory=list)
    mirror_masks: list[list[tuple[float, float]]] = field(default_factory=list)
    ignore_masks: list[list[tuple[float, float]]] = field(default_factory=list)
    zones: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    resolution: Optional[tuple[int, int]] = None
    quality: str = "ok"          # ok | poor | off
    notes: str = ""

    @property
    def usable(self) -> bool:
        return self.quality != "off"


@dataclass
class SalonConfig:
    name: str
    timezone: str
    utc_offset_hours: float
    cameras: dict[str, Camera]
    chairs: dict[str, Chair]
    transitions: dict[str, dict[str, float]]   # зона -> {смежная зона: мин. время перехода, с}
    blind_zones: list[str]
    params: Params
    osd_date_format: str = "%m-%d-%Y"

    def chair_kind(self, chair_id: str) -> str:
        c = self.chairs.get(chair_id)
        return c.kind if c else "regular"

    def pinned_master(self, chair_id: str) -> Optional[str]:
        c = self.chairs.get(chair_id)
        return c.pinned_master if c else None


def _params(raw: dict[str, Any]) -> Params:
    known = {f.name for f in fields(Params)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Неизвестные параметры: {sorted(unknown)}")
    return Params(**raw)


def _poly(raw: list) -> list[tuple[float, float]]:
    return [tuple(map(float, p)) for p in raw]


def box_to_polygon(box: list[float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = map(float, box)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def load_config(path: str | Path) -> SalonConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cameras = {
        cid: Camera(
            camera_id=cid,
            channel=c["channel"],
            zone=c["zone"],
            hall=c.get("hall"),
            homography_points=c.get("homography_points", []),
            mirror_masks=[_poly(m) for m in c.get("mirror_masks", [])],
            ignore_masks=[_poly(m) for m in c.get("ignore_masks", [])],
            zones={k: _poly(v) for k, v in c.get("zones", {}).items()},
            resolution=tuple(c["resolution"]) if c.get("resolution") else None,
            # YAML 1.1 читает голое off как false
            quality={False: "off", True: "ok"}.get(c.get("quality", "ok"), c.get("quality", "ok")),
            notes=c.get("notes", ""),
        )
        for cid, c in data["cameras"].items()
    }
    chairs = {}
    for chair_id, c in data["chairs"].items():
        views = [
            ChairView(
                camera=v["camera"],
                polygon_name=v["polygon"],
                priority=v.get("priority", 1),
                polygon=box_to_polygon(v["box"]) if "box" in v else _poly(v.get("points", [])),
            )
            for v in c.get("views", [])
        ]
        for v in views:
            if v.camera not in cameras:
                raise ValueError(f"{chair_id}: неизвестная камера {v.camera}")
        chairs[chair_id] = Chair(
            chair_id=chair_id,
            hall=c["hall"],
            kind=c.get("kind", "regular"),
            views=views,
            plan_xy=tuple(c["plan_xy"]) if c.get("plan_xy") else None,
            pinned_master=c.get("pinned_master"),
        )
    return SalonConfig(
        name=data.get("name", "salon"),
        timezone=data.get("timezone", "Asia/Astrakhan"),
        utc_offset_hours=data.get("utc_offset_hours", 4),
        cameras=cameras,
        chairs=chairs,
        transitions=data.get("transitions", {}),
        blind_zones=data.get("blind_zones", []),
        params=_params(data.get("params", {})),
        osd_date_format=data.get("osd_date_format", "%m-%d-%Y"),
    )


def default_config() -> SalonConfig:
    """Минимальная конфигурация для тестов и симуляции."""
    return SalonConfig(
        name="test", timezone="Asia/Astrakhan", utc_offset_hours=4,
        cameras={}, chairs={}, transitions={}, blind_zones=[], params=Params(),
    )
