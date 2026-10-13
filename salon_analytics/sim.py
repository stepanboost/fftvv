"""Синтетический день: собирает тики глобальных ID из простого сценария.

Нужен для тестов правил и демо отчёта без видео. Время — минуты от начала дня.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .models import ZONE_ENTRANCE, FaceCandidate, Person, Role, Tick


class DayBuilder:
    def __init__(self, day: str = "2026-10-05", open_local_h: float = 9, utc_offset_h: float = 4, dt: float = 10.0):
        y, m, d = map(int, day.split("-"))
        self.t0 = datetime(y, m, d, tzinfo=timezone.utc).timestamp() + (open_local_h - utc_offset_h) * 3600
        self.dt = dt
        self.ticks: list[Tick] = []
        self.persons: dict[str, Person] = {}
        self.faces: dict[str, list[FaceCandidate]] = {}

    def at(self, minute: float) -> float:
        return self.t0 + minute * 60

    def person(self, gid: str, role: Role = Role.CLIENT, name: Optional[str] = None,
               face: Optional[list[float]] = None, reid: Optional[list[float]] = None) -> Person:
        p = Person(gid, role, name, face=[face] if face else [], reid=[reid] if reid else [])
        self.persons[gid] = p
        return p

    def stay(self, gid: str, zone: str, m0: float, m1: float, *, chair: Optional[str] = None,
             ring_chair: Optional[str] = None, hands: bool = False, cameras_ok: bool = True,
             hands_chair: Optional[str] = None) -> None:
        t = self.at(m0)
        end = self.at(m1)
        while t < end:
            self.ticks.append(Tick(t=t, gid=gid, zone=zone, dt=self.dt, chair=chair,
                                   sitting=chair is not None, ring_chair=ring_chair,
                                   hands_on=hands, cameras_ok=cameras_ok, hands_chair=hands_chair))
            t += self.dt

    def fill(self, gid: str, zone: str, m0: float, m1: float) -> None:
        """Фоновое присутствие: тики только там, где у ID ещё ничего нет."""
        busy = {round((t.t - self.t0) / self.dt) for t in self.ticks if t.gid == gid}
        t = self.at(m0)
        while t < self.at(m1):
            if round((t - self.t0) / self.dt) not in busy:
                self.ticks.append(Tick(t=t, gid=gid, zone=zone, dt=self.dt))
            t += self.dt
        self.ticks.sort(key=lambda x: (x.t, x.gid))

    def enter(self, gid: str, m: float) -> None:
        self.stay(gid, ZONE_ENTRANCE, m, m + 0.5)

    def leave(self, gid: str, m: float) -> None:
        self.stay(gid, ZONE_ENTRANCE, m - 0.5, m)

    def face(self, gid: str, m: float, source: str = "entrance", size: float = 120, yaw: float = 5,
             emb: Optional[list[float]] = None, path: Optional[str] = None) -> None:
        self.faces.setdefault(gid, []).append(FaceCandidate(
            gid=gid, t=self.at(m), source=source, path=path or f"faces/{gid}_{int(m)}.jpg",
            size_px=size, yaw_deg=yaw, sharpness=0.8, eyes_open=True, occluded=False, embedding=emb))

    def service(self, client: str, master: str, chair: str, zone: str, m0: float, m1: float,
                hands: bool = True) -> None:
        """Клиент сидит в кресле, мастер стоит в кольце и работает руками."""
        self.stay(client, zone, m0, m1, chair=chair)
        self.stay(master, zone, m0, m1, ring_chair=chair, hands=hands)


def demo_day() -> DayBuilder:
    """День из сценариев раздела 4.4: окрашивание с выдержкой и мойкой, перекур,
    пересадка между залами, барбер, отдых мастера, ребёнок на диване."""
    b = DayBuilder()
    b.person("m1", Role.MASTER, "Регина")
    b.person("m2", Role.MASTER, "Алексей")
    b.person("m3", Role.MASTER, "Ислам")
    b.person("adm", Role.ADMIN, "Админ")
    shifts = (("m1", 0, 660), ("m2", 10, 600), ("m3", 60, 540), ("adm", 0, 660))
    for gid, start, end in shifts:
        b.enter(gid, start)
        b.leave(gid, end)

    # Регина: окрашивание 3 ч — кресло → выдержка на диване (с перекуром) → мойка → кресло
    b.person("c1", face=[1, 0, 0, 0])
    b.enter("c1", 30); b.face("c1", 30.2, emb=[1, 0, 0, 0])
    b.service("c1", "m1", "1-1", "hall_b", 31, 75)
    b.stay("c1", "hall_b", 75, 109.5)              # выдержка на диване
    b.leave("c1", 110)
    b.person("c1b", face=[0.97, 0.05, 0, 0])     # вернулся после перекура — новый ID на входе
    b.enter("c1b", 122)
    b.stay("c1b", "hall_b", 122.5, 130)
    b.service("c1b", "m2", "M-1", "wash", 130, 140)   # мойку делает Алексей
    b.service("c1b", "m1", "1-1", "hall_b", 140, 215)
    b.leave("c1b", 216)

    # Алексей: две стрижки, вторая — с пересадкой из зала А в зал Б
    b.person("c2", face=[0, 1, 0, 0])
    b.enter("c2", 20); b.face("c2", 20.2, emb=[0, 1, 0, 0])
    b.service("c2", "m2", "2-1", "hall_a", 21, 60)
    b.leave("c2", 62)
    b.person("c3")
    b.enter("c3", 240); b.face("c3", 241, source="ceiling", size=60, yaw=50)
    b.service("c3", "m2", "2-2", "hall_a", 241, 260)
    b.service("c3", "m2", "1-2", "hall_b", 262, 290)
    b.leave("c3", 292)

    # Ислам: барбер 4 мин (засчитывается, барберский порог 3 мин) и окантовка 4 мин в зале (на проверку)
    b.person("c4", face=[0, 0, 1, 0])
    b.enter("c4", 300); b.face("c4", 300.2, emb=[0, 0, 1, 0])
    b.service("c4", "m3", "barber-8", "barber", 301, 305)
    b.leave("c4", 306)
    b.person("c5", face=[0, 0, 0, 1])
    b.enter("c5", 320); b.face("c5", 320.2, emb=[0, 0, 0, 1])
    b.service("c5", "m3", "2-3", "hall_a", 321, 325)
    b.leave("c5", 326)

    # Мастер сел в клиентское кресло поесть — не посадка
    b.stay("m1", "hall_a", 400, 430, chair="2-3")
    # Ребёнок ждёт на диване — не визит
    b.person("kid")
    b.enter("kid", 30); b.stay("kid", "hall_b", 30.5, 215); b.leave("kid", 216)
    # Алексей вышел покурить на 15 мин
    b.leave("m2", 180); b.enter("m2", 195)
    for gid, start, end in shifts:
        if gid == "m2":
            b.fill(gid, "hall_a", start + 0.5, 179.5)
            b.fill(gid, "hall_a", 195.5, end - 0.5)
        else:
            b.fill(gid, "hall_a", start + 0.5, end - 0.5)
    return b


def crowded_day() -> DayBuilder:
    """Зал 2 битком: три клиента в 2-1, 2-2, 2-3 одновременно и два мастера.

    Регина стрижёт 2-1, потом наносит краску в 2-3, стоя ближе к 2-2;
    Алексей ведёт 2-2. Четвёртый клиент ждёт на диване и садится в 2-1,
    когда первый уходит. Ещё двое вошли компанией и сидят на диване.
    """
    b = DayBuilder()
    b.person("m1", Role.MASTER, "Регина")
    b.person("m2", Role.MASTER, "Алексей")
    for gid in ("m1", "m2"):
        b.enter(gid, 0)
    for gid, face in (("c1", [1, 0, 0, 0]), ("c2", [0, 1, 0, 0]), ("c3", [0, 0, 1, 0]), ("c4", [0, 0, 0, 1])):
        b.person(gid, face=face)
    b.enter("c1", 5); b.enter("c2", 5); b.enter("c3", 6); b.enter("c4", 20)
    for gid, m in (("c1", 5), ("c2", 5), ("c3", 6), ("c4", 20)):
        b.face(gid, m + 0.2, emb=b.persons[gid].face[0])
    b.stay("c1", "hall_a", 6, 50, chair="2-1")
    b.stay("c2", "hall_a", 6, 70, chair="2-2")
    b.stay("c3", "hall_a", 7, 90, chair="2-3")
    b.stay("c4", "hall_a", 20.5, 52)          # ждёт на диване
    b.stay("c4", "hall_a", 52, 85, chair="2-1")
    # Регина: 2-1 руками, затем краска в 2-3, геометрически ближе к 2-2
    b.stay("m1", "hall_a", 6, 46, ring_chair="2-1", hands=True)
    b.stay("m1", "hall_a", 46, 52, ring_chair="2-1")                       # убирает место
    b.stay("m1", "hall_a", 52, 70, ring_chair="2-2", hands_chair="2-3")
    b.stay("m1", "hall_a", 70, 85, ring_chair="2-1", hands_chair="2-1")    # c4 в 2-1
    # Алексей: 2-2
    b.stay("m2", "hall_a", 6, 66, ring_chair="2-2", hands_chair="2-2")
    for gid, m in (("c1", 51), ("c2", 71), ("c3", 91), ("c4", 86)):
        b.leave(gid, m)
    for gid in ("m1", "m2"):
        b.fill(gid, "hall_a", 0.5, 100)
    # компания из двух человек на диване — не визиты
    b.person("g1"); b.person("g2")
    b.enter("g1", 10); b.enter("g2", 10)
    b.stay("g1", "hall_a", 10.5, 60); b.stay("g2", "hall_a", 10.5, 60)
    b.leave("g1", 61); b.leave("g2", 61)
    return b
