"""Детектор и роли. Тест с настоящими моделями пропускается, если их нет."""
from pathlib import Path

import numpy as np
import pytest

from salon_analytics.config import load_config
from salon_analytics.models import FaceCandidate, Person, Role, Tick

CFG = load_config(Path(__file__).resolve().parent.parent / "config" / "salon.yaml")
MODELS = Path(__file__).resolve().parent.parent / "models"


def _face(cx, cy, size=40):
    from salon_analytics.pipeline.faces import Face
    return Face((cx - size / 2, cy - size / 2, size, size), 0.9, np.zeros((5, 2)), [1.0], np.zeros((1, 1, 3)), 0.5, 0)


def test_faces_matched_one_to_one_in_crowd():
    from salon_analytics.pipeline.detector import match_faces_to_people
    # мастер стоит вплотную за клиентом: боксы перекрываются
    people = {1: (0.40, 0.30, 0.60, 0.95), 2: (0.48, 0.20, 0.70, 0.95)}
    kps = {1: [(0.50, 0.38)] + [None] * 16, 2: [(0.60, 0.27)] + [None] * 16}
    faces = [_face(500, 380), _face(600, 270)]           # кадр 1000x1000
    got = match_faces_to_people(faces, people, kps, 1000, 1000)
    assert got[1] is faces[0] and got[2] is faces[1]
    # одно лицо на двоих — достаётся только одному
    got = match_faces_to_people([faces[0]], people, kps, 1000, 1000)
    assert list(got) == [1]


def test_face_outside_body_not_attached():
    from salon_analytics.pipeline.detector import match_faces_to_people
    got = match_faces_to_people([_face(100, 100)], {1: (0.4, 0.3, 0.6, 0.9)}, {1: [None] * 17}, 1000, 1000)
    assert got == {}


def _cand(gid, emb):
    return FaceCandidate(gid, 0, "ceiling", "x.jpg", 100, 0, 0.8, True, False, emb)


def test_roles_master_by_face_staff_by_presence_client_rest():
    from salon_analytics.roles import assign_roles
    masters = {"Регина": [[1, 0, 0, 0]], "Алексей": [[0, 1, 0, 0]]}
    persons = {g: Person(g) for g in ("a", "b", "c", "d")}
    faces = {"a": [_cand("a", [0.95, 0.1, 0, 0]), _cand("a", [0.9, 0.2, 0, 0])],
             "c": [_cand("c", [0, 0, 1, 0])],
             "d": [_cand("d", [0.9, 0.1, 0.3, 0])]}            # один кадр похож на Регину
    ticks = [Tick(t=float(i), gid="b", zone="hall_a", dt=3600) for i in range(4)]
    assign_roles(persons, faces, ticks, masters, CFG.params)
    assert persons["a"].role == Role.MASTER and persons["a"].name == "Регина"
    assert persons["a"].face == masters["Регина"]
    assert persons["b"].role == Role.STAFF_UNKNOWN
    assert persons["c"].role == Role.CLIENT and persons["c"].face == [[0, 0, 1, 0]]
    assert persons["d"].role == Role.MASTER or persons["d"].face == []   # чужое лицо в галерею клиента не идёт


@pytest.mark.skipif(not (MODELS / "face_detection_yunet_2023mar.onnx").exists(), reason="нет моделей")
def test_real_models_on_sample_image():
    pytest.importorskip("ultralytics"); pytest.importorskip("boxmot")
    import cv2
    from ultralytics.utils import ASSETS
    from salon_analytics.pipeline.detector import CVDetector, DetectorSettings
    det = CVDetector(CFG, DetectorSettings(models_dir=MODELS, use_reid=False))
    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    for _ in range(4):
        tracks = det.tracks("cam3", img, ir=False)
    assert len(tracks) >= 2
    owners = [f.track_id for f in det.face_obs if f.camera == "cam3"]
    assert len(set(owners)) >= 2
    embs = {f.track_id: f.face.embedding for f in det.face_obs}
    a, b = list(embs.values())[:2]
    assert float(np.dot(a, b)) < 0.363            # разные люди — разные лица


def test_cleanup_old_days(tmp_path):
    from salon_analytics.cli import cleanup_old_days
    for d in ("2026-06-01", "2026-09-30", "2026-10-14"):
        (tmp_path / d).mkdir()
    (tmp_path / "masters.json").write_text("[]")
    assert cleanup_old_days(tmp_path, "2026-10-14", 90) == ["2026-06-01"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-09-30", "2026-10-14", "masters.json"]
