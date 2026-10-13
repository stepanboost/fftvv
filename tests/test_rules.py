"""Сценарии раздела 4.4 / 5.1 ТЗ на синтетических днях."""
from pathlib import Path

import pytest

from salon_analytics.attribution import Booking, Shift
from salon_analytics.config import load_config
from salon_analytics.engine import DayInput, analyze_day
from salon_analytics.models import Outage, Role
from salon_analytics.sim import DayBuilder, demo_day

CFG = load_config(Path(__file__).resolve().parent.parent / "config" / "salon.yaml")


def run(b: DayBuilder, **kw):
    return analyze_day(DayInput(b.ticks, b.persons, b.faces, **kw), CFG)


def masters(b: DayBuilder, *pairs):
    for gid, name in pairs:
        b.person(gid, Role.MASTER, name)


def simple_client(b, gid, master, chair="2-1", zone="hall_a", m0=0, dur=40, face=True):
    b.person(gid, face=[1, 0, 0] if face else None)
    b.enter(gid, m0)
    if face:
        b.face(gid, m0 + 0.2)
    b.service(gid, master, chair, zone, m0 + 1, m0 + 1 + dur)
    b.leave(gid, m0 + dur + 2)


def test_simple_haircut():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    simple_client(b, "c1", "m1")
    r = run(b)
    assert len(r.visits) == 1
    v = r.visits[0]
    assert v.master == "Регина" and v.status == "авто" and v.confidence >= 0.9
    assert round(v.seated_s / 60) == 40


def test_demo_day():
    r = analyze_day(DayInput(demo_day().ticks, demo_day().persons, demo_day().faces), CFG)
    by_master = {}
    for v in r.visits:
        by_master.setdefault(v.master, []).append(v)
    assert {k: len(v) for k, v in by_master.items()} == {"Регина": 1, "Алексей": 2, "Ислам": 1}
    regina = by_master["Регина"][0]
    assert regina.zones_path == ["1-1", "мойка", "1-1"]
    assert regina.second_master == "Алексей" and regina.second_master_role == "мойка"
    assert not r.violations


def test_coloring_with_smoke_break_is_one_visit():
    """Окрашивание: кресло → диван → вышел покурить (новый ID на входе) → кресло."""
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1", face=[1, 0, 0])
    b.enter("c1", 0); b.service("c1", "m1", "2-1", "hall_a", 1, 60)
    b.stay("c1", "hall_a", 60, 99.5); b.leave("c1", 100)
    b.person("c1b", face=[0.98, 0.1, 0])
    b.enter("c1b", 115); b.service("c1b", "m1", "2-1", "hall_a", 116, 200); b.leave("c1b", 201)
    r = run(b)
    assert len(r.visits) == 1
    assert r.visits[0].gids == ["c1", "c1b"]
    assert any(e.kind == "склеен" for e in r.journal)


def test_repeat_visit_same_day_after_long_absence():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1", face=[1, 0, 0])
    b.enter("c1", 0); b.service("c1", "m1", "2-1", "hall_a", 1, 30); b.leave("c1", 31)
    b.enter("c1", 90); b.service("c1", "m1", "2-1", "hall_a", 91, 120); b.leave("c1", 121)
    assert len(run(b).visits) == 2


def test_lost_id_mid_hall_short_is_same_visit():
    """Перекрытие/трекер мигнул посреди зала, ID вернулся через 10 мин."""
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1")
    b.enter("c1", 0); b.service("c1", "m1", "2-1", "hall_a", 1, 30)
    b.service("c1", "m1", "2-1", "hall_a", 40, 60); b.leave("c1", 61)
    r = run(b)
    assert len(r.visits) == 1
    assert len(r.visits[0].segments) == 2   # разрыв > 3 мин — два сегмента, один визит


def test_short_gap_in_chair_merges_segments():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1")
    b.enter("c1", 0); b.service("c1", "m1", "2-1", "hall_a", 1, 20)
    b.stay("c1", "hall_a", 20, 22)   # встал поправить накидку
    b.service("c1", "m1", "2-1", "hall_a", 22, 40); b.leave("c1", 41)
    v = run(b).visits[0]
    assert len(v.segments) == 1


def test_chair_change_between_halls():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1")
    b.enter("c1", 0)
    b.service("c1", "m1", "2-1", "hall_a", 1, 20)
    b.service("c1", "m1", "1-2", "hall_b", 22, 40)
    b.leave("c1", 41)
    r = run(b)
    assert len(r.visits) == 1 and r.visits[0].zones_path == ["2-1", "1-2"]


def test_master_resting_in_chair_is_not_visit():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.stay("m1", "hall_a", 0, 30, chair="2-3")
    assert not run(b).visits


def test_master_cuts_master():
    b = DayBuilder(); masters(b, ("m1", "Регина"), ("m2", "Алексей"))
    b.service("m2", "m1", "2-2", "hall_a", 0, 30)
    r = run(b)
    assert len(r.visits) == 1
    v = r.visits[0]
    assert v.staff_client and v.master == "Регина" and "сотрудник" in v.flags


def test_kid_on_sofa_is_not_visit():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("kid"); b.enter("kid", 0); b.stay("kid", "hall_a", 0.5, 60); b.leave("kid", 61)
    assert not run(b).visits and not run(b).possible


def test_sat_without_master_goes_to_review():
    b = DayBuilder()
    b.person("c1"); b.enter("c1", 0); b.stay("c1", "hall_a", 1, 15, chair="2-1"); b.leave("c1", 16)
    r = run(b)
    assert not r.visits
    assert len(r.possible) == 1 and "без мастера" in r.possible[0].review_reasons[0]


def test_short_service_review_and_barber_threshold():
    b = DayBuilder(); masters(b, ("m1", "Ислам"))
    b.person("c1"); b.enter("c1", 0); b.service("c1", "m1", "2-1", "hall_a", 1, 5); b.leave("c1", 6)
    b.person("c2"); b.enter("c2", 10); b.service("c2", "m1", "barber-8", "barber", 11, 15); b.leave("c2", 16)
    r = run(b)
    assert [v.zones_path for v in r.visits] == [["barber-8"]]     # барбер: порог 3 мин
    assert len(r.possible) == 1 and "короткая" in r.possible[0].review_reasons[0]


def test_two_parallel_clients_one_master():
    """У одного выдержка, другого стрижёт — оба визита мастеру."""
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.person("c1"); b.enter("c1", 0)
    b.service("c1", "m1", "2-1", "hall_a", 1, 20)
    b.stay("c1", "hall_a", 20, 80, chair="2-1")      # выдержка в кресле
    b.leave("c1", 81)
    b.person("c2"); b.enter("c2", 25); b.service("c2", "m1", "2-2", "hall_a", 26, 70); b.leave("c2", 71)
    r = run(b)
    assert [v.master for v in r.visits] == ["Регина", "Регина"]


def test_longest_hands_time_wins_and_colorist_second():
    b = DayBuilder(); masters(b, ("m1", "Регина"), ("m2", "Алексей"))
    b.person("c1"); b.enter("c1", 0)
    b.stay("c1", "hall_a", 1, 61, chair="2-1")
    b.stay("m1", "hall_a", 1, 41, ring_chair="2-1", hands=True)
    b.stay("m2", "hall_a", 41, 61, ring_chair="2-1", hands=True)
    b.leave("c1", 62)
    v = run(b).visits[0]
    assert v.master == "Регина" and v.second_master == "Алексей" and v.second_master_role is None


def test_chatting_master_without_hands_does_not_win():
    b = DayBuilder(); masters(b, ("m1", "Регина"), ("m2", "Алексей"))
    b.person("c1"); b.enter("c1", 0)
    b.stay("c1", "hall_a", 1, 41, chair="2-1")
    b.stay("m1", "hall_a", 1, 21, ring_chair="2-1", hands=True)
    b.stay("m2", "hall_a", 1, 41, ring_chair="2-1", hands=False)   # болтает рядом
    b.leave("c1", 42)
    assert run(b).visits[0].master == "Регина"


def test_admin_does_not_vote():
    b = DayBuilder(); masters(b, ("m1", "Регина")); b.person("a", Role.ADMIN, "Админ")
    b.person("c1"); b.enter("c1", 0)
    b.stay("c1", "hall_a", 1, 41, chair="2-1")
    b.stay("a", "hall_a", 1, 41, ring_chair="2-1", hands=True)
    b.stay("m1", "hall_a", 1, 10, ring_chair="2-1", hands=True)
    b.leave("c1", 42)
    assert run(b).visits[0].master == "Регина"


def test_master_not_on_shift_cannot_get_visit():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    simple_client(b, "c1", "m1")
    r = run(b, shifts=[Shift("Алексей", b.at(0), b.at(600))])
    assert r.visits[0].master is None and "мастер не определён" in r.visits[0].review_reasons


def test_unknown_staff_goes_to_review():
    b = DayBuilder(); b.person("s", Role.STAFF_UNKNOWN)
    simple_client(b, "c1", "s")
    v = run(b).visits[0]
    assert v.master.startswith("сотрудник?") and v.status == "на проверке"


def test_booking_raises_confidence():
    b = DayBuilder(); masters(b, ("m1", "Регина"), ("m2", "Алексей"))
    b.person("c1", face=[1, 0, 0]); b.enter("c1", 0); b.face("c1", 0.2)
    b.stay("c1", "hall_a", 1, 41, chair="2-1")
    b.stay("m1", "hall_a", 1, 29, ring_chair="2-1", hands=True)
    b.stay("m2", "hall_a", 29, 41, ring_chair="2-1", hands=True)
    b.leave("c1", 42)
    base = run(b).visits[0].master_conf
    boosted = run(b, bookings=[Booking("Регина", b.at(0))]).visits[0].master_conf
    assert boosted > base


def test_camera_outage_marks_visit_approx():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    simple_client(b, "c1", "m1")
    r = run(b, outages=[Outage("cam3", b.at(10), b.at(20))], cameras=["cam3", "cam4"])
    v = r.visits[0]
    assert v.visit_conf < 1 and "нет данных части камер" in v.flags
    assert r.coverage < 1


def test_no_face_goes_to_review():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    simple_client(b, "c1", "m1", face=False)
    v = run(b).visits[0]
    assert "лица нет" in v.review_reasons


@pytest.mark.parametrize("crm,expect", [(1, False), (2, False), (4, True)])
def test_crm_reconcile(crm, expect):
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    simple_client(b, "c1", "m1")
    r = run(b, crm_checks=crm)
    assert any(x.rule == "расхождение с CRM" for x in r.violations) is expect


def test_master_day_exits_and_intervals():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.enter("m1", 0); b.stay("m1", "hall_a", 0.5, 59.5); b.leave("m1", 60)     # перекур 10 мин
    b.enter("m1", 70); b.stay("m1", "hall_a", 70.5, 179.5); b.leave("m1", 180)  # ушёл на 2 ч
    b.enter("m1", 300); b.stay("m1", "hall_a", 300.5, 400)
    d = run(b).masters["Регина"]
    assert len(d.exits) == 2 and len(d.intervals) == 2
    assert d.departure_approx   # уход не попал на камеру входа


def test_master_hands_on_two_clients_same_minute_is_violation():
    b = DayBuilder(); masters(b, ("m1", "Регина"))
    b.stay("m1", "hall_a", 0, 1, ring_chair="2-1", hands=True)
    b.stay("m1", "hall_a", 0.5, 1, ring_chair="2-2", hands=True)
    r = run(b)
    assert any(x.rule == "мастер руками у двух клиентов" for x in r.violations)
