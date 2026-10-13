"""В одном кадре 3 и более клиентов: рассадка, кому руки мастера, ID в толпе."""
from pathlib import Path

from salon_analytics.config import load_config
from salon_analytics.engine import DayInput, analyze_day
from salon_analytics.models import Person, Role
from salon_analytics.scene import assign_sitters, resolve_staff
from salon_analytics.sim import DayBuilder, crowded_day
from salon_analytics.tracking.global_ids import GlobalIdManager, LocalTrack
from salon_analytics.visits import Fragment, stitch

CFG = load_config(Path(__file__).resolve().parent.parent / "config" / "salon.yaml")


def kp(pelvis=None, head=None, wrist=None):
    k = [None] * 17
    if pelvis:
        k[11] = k[12] = pelvis
    if head:
        k[0] = head
    if wrist:
        k[9] = wrist
    return k


def test_crowded_day_every_client_to_right_master():
    b = crowded_day()
    r = analyze_day(DayInput(b.ticks, b.persons, b.faces), CFG)
    got = sorted((v.gids[0], v.master, tuple(v.zones_path)) for v in r.visits)
    assert got == [("c1", "Регина", ("2-1",)), ("c2", "Алексей", ("2-2",)),
                   ("c3", "Регина", ("2-3",)), ("c4", "Регина", ("2-1",))]
    c2 = next(v for v in r.visits if v.gids == ["c2"])
    assert c2.second_master is None        # Регина стояла ближе к 2-2, но руки были в 2-3
    assert not r.possible                  # компания на диване — не визиты и не спорные
    assert not r.violations


def test_hands_beat_proximity_between_chairs():
    """Мастер стоит ближе к 2-1, а работает руками с клиентом в 2-2."""
    b = DayBuilder()
    b.person("m1", Role.MASTER, "Регина")
    b.person("c1"); b.person("c2")
    b.enter("c1", 0); b.enter("c2", 0)
    b.stay("c1", "hall_a", 1, 31, chair="2-1")
    b.stay("c2", "hall_a", 1, 31, chair="2-2")
    b.stay("m1", "hall_a", 1, 31, ring_chair="2-1", hands_chair="2-2")
    b.leave("c1", 32); b.leave("c2", 32)
    r = analyze_day(DayInput(b.ticks, b.persons), CFG)
    assert [v.gids for v in r.visits] == [["c2"]]
    assert r.visits[0].master == "Регина"
    assert [v.gids for v in r.possible] == [["c1"]]    # сидел 30 мин без мастера → проверка


def test_assign_sitters_overlapping_polygons_one_to_one():
    chairs = {"2-1": [(0.2, 0.5), (0.4, 0.5), (0.4, 0.8), (0.2, 0.8)],
              "2-2": [(0.35, 0.5), (0.6, 0.5), (0.6, 0.8), (0.35, 0.8)],     # перекрывает 2-1
              "2-3": [(0.55, 0.5), (0.8, 0.5), (0.8, 0.8), (0.55, 0.8)]}
    sitting = {"a": kp(pelvis=(0.30, 0.65)), "b": kp(pelvis=(0.38, 0.65)), "c": kp(pelvis=(0.66, 0.65))}
    got = assign_sitters(sitting, chairs)
    assert got == {"a": "2-1", "b": "2-2", "c": "2-3"}
    assert len(set(got.values())) == 3


def test_assign_sitters_more_people_than_chairs():
    chairs = {"2-1": [(0.2, 0.5), (0.4, 0.5), (0.4, 0.8), (0.2, 0.8)]}
    got = assign_sitters({"a": kp(pelvis=(0.3, 0.6)), "b": kp(pelvis=(0.31, 0.62))}, chairs)
    assert list(got.values()) == ["2-1"] and len(got) == 1


def test_resolve_staff_three_clients_two_masters():
    sitters = {"c1": "2-1", "c2": "2-2", "c3": "2-3"}
    sitter_kp = {"c1": kp(head=(0.30, 0.40)), "c2": kp(head=(0.50, 0.40)), "c3": kp(head=(0.70, 0.40))}
    chair_xy = {"2-1": (1.0, 1.5), "2-2": (2.2, 1.5), "2-3": (3.4, 1.5)}
    staff_xy = {"m1": (1.7, 2.2), "m2": (3.4, 2.4)}       # m1 между 2-1 и 2-2
    staff_kp = {"m1": kp(wrist=(0.51, 0.41)), "m2": kp(wrist=(0.9, 0.9))}
    got = resolve_staff(staff_xy, staff_kp, sitters, sitter_kp, chair_xy, ring=(0.4, 1.2), hands_radius=0.05)
    assert got["m1"].hands_chair == "2-2"
    assert got["m1"].ring_chair == "2-2"     # 2-1 и 2-2 оба в кольце, 2-2 ближе
    assert got["m2"].hands_chair is None and got["m2"].ring_chair == "2-3"


def _t(cam, tid, zone, xy, reid=None):
    return LocalTrack(cam, tid, zone, plan_xy=xy, reid=reid)


def _enter_three(m):
    reids = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    gids = {}
    for i, (tid, r) in enumerate(reids.items()):
        gids[tid] = m.step(i * 0.5, [_t("cam3", tid, "entrance", (tid * 3.0, 0), r)])[("cam3", tid)]
    return reids, gids


def test_three_way_crossing_keeps_ids():
    """Трое прошли вплотную: позиции почти совпали, решает одежда."""
    m = GlobalIdManager(CFG)
    reids, gids = _enter_three(m)
    m.step(2.0, [_t("cam3", tid, "hall_a", (1.0, 1), reids[tid]) for tid in reids])   # все в одной точке
    a = m.step(2.5, [_t("cam3", 1, "hall_a", (0.5, 1), reids[3]),
                     _t("cam3", 2, "hall_a", (1.0, 1), reids[1]),
                     _t("cam3", 3, "hall_a", (1.5, 1), reids[2])])
    assert a[("cam3", 1)] == gids[3] and a[("cam3", 2)] == gids[1] and a[("cam3", 3)] == gids[2]


def test_crowd_fix_repairs_cyclic_mixup():
    """Назначение перепутано по кругу A→B→C→A — попарный обмен такое не чинит."""
    m = GlobalIdManager(CFG)
    reids, gids = _enter_three(m)
    tracks = [_t("cam3", tid, "hall_a", (0.5 * tid, 1), reids[tid]) for tid in reids]
    wrong = {("cam3", 1): gids[2], ("cam3", 2): gids[3], ("cam3", 3): gids[1]}
    m._fix_crowd_swaps(tracks, wrong, 3.0)
    assert wrong == {("cam3", tid): gids[tid] for tid in reids}
    assert sum(e.kind == "обменян" for e in m.journal) == 3


def test_one_id_cannot_sit_in_two_chairs_on_two_cameras():
    """Двое в накидках в соседних креслах: одежда одинаковая, ног не видно."""
    cape = [0.5, 0.5, 0.5]
    m = GlobalIdManager(CFG)
    g = m.step(0, [_t("cam3", 1, "entrance", (0, 0), cape)])[("cam3", 1)]
    sit = lambda cam, tid, chair: LocalTrack(cam, tid, "hall_a", reid=cape, chair=chair)
    m.step(0.5, [sit("cam3", 1, "2-1")])
    a = m.step(1.0, [sit("cam3", 1, "2-1"), sit("cam4", 5, "2-2")])
    assert a[("cam3", 1)] == g and a[("cam4", 5)] != g
    assert any(e.kind == "конфликт камер" for e in m.journal)


def test_group_entry_split_gets_entrance_retroactively():
    """Вошли втроём одним боксом; в зале бокс распался на три трека."""
    m = GlobalIdManager(CFG)
    g = m.step(0, [_t("cam3", 1, "entrance", (0, 0), [1, 0, 0])])[("cam3", 1)]
    m.step(0.5, [_t("cam3", 1, "hall_a", (0.3, 0), [1, 0, 0])])
    m.step(1.0, [_t("cam3", 1, "hall_a", (0.5, 0), [1, 0, 0])])
    a = m.step(1.5, [_t("cam3", 1, "hall_a", (0.6, 0), [1, 0, 0]),
                     _t("cam3", 2, "hall_a", (0.9, 0.3), [0, 1, 0]),
                     _t("cam3", 3, "hall_a", (0.4, 0.5), [0, 0, 1])])
    new = {a[("cam3", 2)], a[("cam3", 3)]}
    assert g not in new and len(new) == 2
    for gid in new:
        assert m.states[gid].created_at_entrance
        first = min((x for x in m.ticks if x.gid == gid), key=lambda x: x.t)
        assert first.zone == "entrance" and first.t == 0
    assert sum(e.kind == "отделился" for e in m.journal) == 2


def test_stitch_picks_most_similar_not_first():
    """Двое клиентов потеряли ID почти одновременно; куски склеиваются по лицу."""
    persons = {g: Person(g, face=[f]) for g, f in
               (("a", [1, 0, 0]), ("b", [0, 1, 0]), ("a2", [0.96, 0.3, 0]), ("b2", [0.6, 0.8, 0]))}
    frags = [Fragment("a", 0, 100, True, False), Fragment("b", 0, 110, True, False),
             Fragment("a2", 200, 300, False, True), Fragment("b2", 210, 300, False, True)]
    stitch(frags, persons, CFG, [])
    assert frags[0].next is frags[2] and frags[1].next is frags[3]
