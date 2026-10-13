"""Мастер и клиент на фото не путаются и не смешиваются."""
from pathlib import Path

from salon_analytics import masters_page
from salon_analytics.config import load_config
from salon_analytics.engine import DayInput, analyze_day
from salon_analytics.models import Role
from salon_analytics.sim import DayBuilder

CFG = load_config(Path(__file__).resolve().parent.parent / "config" / "salon.yaml")
REGINA, ALEX = [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]
C1, C2 = [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]


def day():
    b = DayBuilder()
    b.person("m1", Role.MASTER, "Регина", face=REGINA)
    b.person("m2", Role.MASTER, "Алексей", face=ALEX)
    b.enter("m1", 0); b.face("m1", 0.2, emb=REGINA, path="faces/regina.jpg")
    b.enter("m2", 0); b.face("m2", 0.3, emb=ALEX, path="faces/alex.jpg")
    b.person("c1", face=C1); b.person("c2", face=C2)
    b.enter("c1", 5); b.enter("c2", 5)
    b.face("c1", 5.2, emb=C1, path="faces/c1.jpg", size=100)
    b.face("c2", 5.2, emb=C2, path="faces/c2.jpg", size=100)
    b.service("c1", "m1", "2-1", "hall_a", 6, 46)
    b.service("c2", "m2", "2-2", "hall_a", 6, 46)
    b.leave("c1", 47); b.leave("c2", 47)
    return b


def run(b):
    return analyze_day(DayInput(b.ticks, b.persons, b.faces), CFG)


def photo_of(r, gid):
    return next(v for v in r.visits if gid in v.gids).photo.best.path


def test_clean_day_each_person_own_photo():
    r = run(day())
    assert photo_of(r, "c1") == "faces/c1.jpg" and photo_of(r, "c2") == "faces/c2.jpg"
    assert r.master_photos["Регина"].best.path == "faces/regina.jpg"
    assert r.master_photos["Алексей"].best.path == "faces/alex.jpg"
    assert {n: d.clients for n, d in r.masters.items()} == {"Регина": 1, "Алексей": 1}


def test_master_face_in_client_track_is_rejected():
    """Мастер наклонился над клиентом, и его лицо попало в трек клиента —
    причём кадр лучше, чем настоящее лицо клиента."""
    b = day()
    b.face("c1", 20, emb=[0.98, 0, 0.2, 0, 0], path="faces/c1_is_regina.jpg", size=200)
    r = run(b)
    assert photo_of(r, "c1") == "faces/c1.jpg"


def test_neighbour_client_face_is_rejected():
    b = day()
    b.face("c1", 20, emb=[0, 0, 0.1, 0.99, 0], path="faces/c1_is_c2.jpg", size=200)
    assert photo_of(run(b), "c1") == "faces/c1.jpg"


def test_client_face_in_master_track_is_rejected():
    b = day()
    b.face("m1", 30, emb=[0.1, 0, 0.99, 0, 0], path="faces/regina_is_c1.jpg", size=200)
    assert run(b).master_photos["Регина"].best.path == "faces/regina.jpg"


def test_only_contaminated_frames_means_no_face_not_wrong_face():
    b = DayBuilder()
    b.person("m1", Role.MASTER, "Регина", face=REGINA)
    b.person("c1")                                  # своего лица в галерее нет
    b.enter("c1", 5)
    b.face("c1", 20, emb=[0.97, 0.1, 0.1, 0, 0], path="faces/regina_again.jpg", size=200)
    b.service("c1", "m1", "2-1", "hall_a", 6, 46); b.leave("c1", 47)
    v = run(b).visits[0]
    assert v.photo.best is None or v.photo.best.path != "faces/regina_again.jpg"
    assert "лица нет" in v.review_reasons


def test_master_photo_never_reused_as_client_photo():
    b = day()
    b.faces["c1"].append(b.faces["m1"][0])          # один и тот же кадр у обоих
    r = run(b)
    assert r.master_photos["Регина"].best.path != photo_of(r, "c1")


def test_colleague_haircut_not_counted_as_client():
    b = day()
    b.service("m2", "m1", "2-3", "hall_a", 60, 90)  # Регина стрижёт Алексея
    d = run(b).masters["Регина"]
    assert d.clients == 1 and d.colleagues == 1


def test_masters_page(tmp_path):
    (tmp_path / "faces").mkdir()
    for name in ("regina", "alex", "c1", "c2"):
        (tmp_path / "faces" / f"{name}.jpg").write_bytes(b"\xff\xd8\xff" + name.encode())
    page = masters_page.render(run(day()), CFG, tmp_path)
    assert "Регина" in page and "Алексей" in page and "1 клиент" in page
    assert page.count('class="mphoto"') == 2 and page.count("data:image/jpeg;base64") == 4
    assert "<script" not in page


def test_contaminated_client_gallery_does_not_let_master_face_through():
    b = day()
    b.persons["c1"].face.append([0.99, 0, 0.1, 0, 0])   # в галерею клиента попало лицо Регины
    b.face("c1", 20, emb=[0.98, 0, 0.15, 0, 0], path="faces/c1_is_regina.jpg", size=200)
    assert photo_of(run(b), "c1") == "faces/c1.jpg"
