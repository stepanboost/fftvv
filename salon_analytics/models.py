"""Структуры данных ядра правил.

Всё время внутри системы — UTC, секунды epoch (float). В местное время салона
переводится только в отчёте (см. report.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    MASTER = "master"
    CLIENT = "client"
    ADMIN = "admin"
    GUEST = "guest"          # сопровождающий, курьер
    STAFF_UNKNOWN = "staff?"  # похож на сотрудника, ждём ответа админа в боте


# Служебные зоны. Остальные зоны — имена из конфига (hall_a, hall_b, wash, ...).
ZONE_OUTSIDE = "outside"
ZONE_ENTRANCE = "entrance"
BLIND_PREFIX = "blind:"      # blind:wc, blind:wardrobe
BACKROOM_PREFIX = "back:"    # back:kitchen — «на месте, не в зале»


@dataclass(frozen=True)
class Tick:
    """Состояние одного глобального ID в момент t (выход глобального менеджера).

    Один тик покрывает интервал [t, t + dt).
    """
    t: float
    gid: str
    zone: str
    dt: float = 1.0
    chair: Optional[str] = None        # физическое кресло, в котором сидит (таз в полигоне)
    sitting: bool = False
    # Кольца соседних кресел перекрываются (2-1/2-2/2-3 стоят в ~1.2 м),
    # поэтому «рядом» и «руками» — разные кресла: ring_chair — ближайшее ЗАНЯТОЕ
    # кресло, в кольце 0.4–1.2 м которого стоит; hands_chair — кресло, у головы
    # сидящего в котором сейчас запястья (см. scene.resolve_staff).
    ring_chair: Optional[str] = None
    hands_on: bool = False             # руки у головы клиента в ring_chair (старый формат)
    cameras_ok: bool = True            # все камеры зоны отдавали кадры
    hands_chair: Optional[str] = None

    @property
    def present(self) -> bool:
        return self.zone != ZONE_OUTSIDE

    @property
    def hands_target(self) -> Optional[str]:
        """Кресло, с клиентом которого работают руки в этот момент."""
        if self.hands_chair:
            return self.hands_chair
        return self.ring_chair if self.hands_on else None


@dataclass
class Person:
    """Глобальный ID с ролью и галереей признаков."""
    gid: str
    role: Role = Role.CLIENT
    name: Optional[str] = None                 # имя мастера/админа
    face: list[list[float]] = field(default_factory=list)   # эмбеддинги лица (InsightFace)
    reid: list[list[float]] = field(default_factory=list)   # эмбеддинги одежды (OSNet), до 10
    created_at_entrance: bool = True
    events: list["TrackEvent"] = field(default_factory=list)


@dataclass
class TrackEvent:
    """Журнал трека: создан, потерян, склеен, обменян, исправлен админом."""
    t: float
    gid: str
    kind: str
    detail: str = ""


@dataclass
class ChairSegment:
    """Сегмент «кресло»: непрерывное сидение одного ID в одном физическом кресле."""
    gid: str
    chair: str
    start: float
    end: float
    seated_s: float = 0.0
    near_s: dict[str, float] = field(default_factory=dict)    # мастер -> сек в кольце
    hands_s: dict[str, float] = field(default_factory=dict)   # мастер -> сек руками у клиента
    accepted: bool = False
    staff_client: bool = False   # мастер стрижёт другого мастера
    review_reason: Optional[str] = None
    no_data: bool = False        # пересекается с окном «нет данных»

    @property
    def duration_s(self) -> float:
        return self.end - self.start


@dataclass
class Visit:
    visit_id: str
    gids: list[str]
    start: float
    end: float
    segments: list[ChairSegment] = field(default_factory=list)
    zones_path: list[str] = field(default_factory=list)   # B-1 → мойка → B-1
    entered_via_entrance: bool = True
    exited_via_entrance: bool = True
    master: Optional[str] = None
    second_master: Optional[str] = None
    second_master_role: Optional[str] = None   # «мойка» и т.п.
    visit_conf: float = 1.0
    master_conf: float = 0.0
    staff_client: bool = False
    photo: Optional["PhotoChoice"] = None
    client_label: str = ""
    repeat_note: str = ""
    service_hint: str = ""
    status: str = "авто"
    review_reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return round(min(self.visit_conf, self.master_conf), 2)

    @property
    def duration_s(self) -> float:
        return self.end - self.start

    @property
    def seated_s(self) -> float:
        return sum(s.seated_s for s in self.segments if s.accepted)


@dataclass
class FaceCandidate:
    gid: str
    t: float
    source: str          # entrance | mirror | ceiling | body
    path: str
    size_px: float = 0.0
    yaw_deg: float = 90.0
    sharpness: float = 0.0   # 0..1
    eyes_open: bool = False
    occluded: bool = True
    embedding: Optional[list[float]] = None


@dataclass
class PhotoChoice:
    best: Optional[FaceCandidate]
    spares: list[FaceCandidate] = field(default_factory=list)
    has_face: bool = False


@dataclass
class MasterDay:
    name: str
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    departure_approx: bool = False
    intervals: list[tuple[float, float]] = field(default_factory=list)
    exits: list[tuple[float, float]] = field(default_factory=list)   # (start, end) вне салона
    on_site_s: float = 0.0
    work_s: float = 0.0
    clients: int = 0
    colleagues: int = 0

    @property
    def load(self) -> float:
        return self.work_s / self.on_site_s if self.on_site_s else 0.0


@dataclass
class Outage:
    camera: str
    start: float
    end: float
