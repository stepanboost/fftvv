from pathlib import Path

import numpy as np
import pytest

from salon_analytics import storage
from salon_analytics.chairs import ViewObservation, bind_moved_chairs, fuse
from salon_analytics.config import load_config
from salon_analytics.engine import DayInput, analyze_day
from salon_analytics.geometry import fit_homography, project, reprojection_error
from salon_analytics.markup import validate
from salon_analytics.models import FaceCandidate
from salon_analytics.photos import KnownFace, PhotoChoice, choose_photo, repeat_status
from salon_analytics.pipeline.rtsp import SilenceMonitor, is_ir_frame, stream_url
from salon_analytics.pose import hands_on_head, is_seated
from salon_analytics.report import salon_tz, to_csv
from salon_analytics.sim import demo_day
from salon_analytics.tracking.global_ids import GlobalIdManager, LocalTrack

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "salon.yaml"
CFG = load_config(CFG_PATH)


def test_homography_roundtrip():
    img = [(100, 500), (900, 520), (850, 100), (150, 90)]
    plan = [(0, 0), (4, 0), (4, 3), (0, 3)]
    h = fit_homography(img, plan)
    assert reprojection_error(h, img, plan) < 1e-6
    x, y = project(h, (500, 300))
    assert 0 < x < 4 and 0 < y < 3


def test_chair_on_two_cameras_is_one_chair():
    """2-1 размечено на cam3 (пол) и cam4 (бокс): одна физическая посадка."""
    obs = [ViewObservation("cam3", "2-1", True, "G5"), ViewObservation("cam4", "chair 2-1", True, "G5"),
           ViewObservation("cam4", "chair 2-2", False)]
    assert fuse(CFG, obs) == {"2-1": "G5", "2-2": None}


def test_fuse_prefers_better_camera_and_skips_dead_view():
    obs = [ViewObservation("cam3", "2-1", True, "G1"), ViewObservation("cam4", "chair 2-1", True, "G2")]
    assert fuse(CFG, obs)["2-1"] == "G2"      # cam4 priority 2
    obs = [ViewObservation("cam3", "2-1", False), ViewObservation("cam4", "chair 2-1", True, "G2", view_ok=False)]
    assert fuse(CFG, obs)["2-1"] is None


def test_moved_chair_rebinds_to_detection():
    # на cam3 кресло 2-3 отъехало вправо-вверх; 2-1 и 2-2 не найдены — остаются по разметке
    moved = bind_moved_chairs(CFG, "cam3", [(0.66, 0.45, 0.76, 0.70)])
    assert moved["2-3"][0] == (0.66, 0.45)
    assert moved["2-1"] == CFG.chairs["2-1"].views[0].polygon


def test_markup_finds_real_problems():
    issues = {(i.level, i.where, i.text) for i in validate(CFG)}
    assert ("error", "cam5", "камера не работает / смотрит не туда") in issues
    assert any("2-4 и M-1" in t for _, _, t in issues)


def test_markup_catches_off_frame_polygon():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg.chairs["barber-8"].views[0].polygon = [(0.45, 0.35), (1.3, 0.35), (1.3, 1.1), (0.44, 1.1)]
    assert any(i.level == "error" and "за пределами кадра" in i.text for i in validate(cfg))


def test_pose_seated_and_hands():
    kp = [None] * 17
    kp[11], kp[12], kp[13], kp[14] = (0.5, 0.5), (0.55, 0.5), (0.6, 0.52), (0.65, 0.53)
    chair = [(0.4, 0.4), (0.7, 0.4), (0.7, 0.7), (0.4, 0.7)]
    assert is_seated(kp, chair, plan_shift_m=0.05)
    assert not is_seated(kp, chair, plan_shift_m=0.5)
    client = [None] * 17; client[0] = (0.5, 0.3)
    master = [None] * 17; master[9] = (0.52, 0.31)
    assert hands_on_head(master, client, radius_px=0.05)


def face(src="entrance", size=120, yaw=5, emb=None):
    return FaceCandidate("c", 0, src, f"{src}.jpg", size, yaw, 0.8, True, False, emb)


def test_photo_priority_and_fallback():
    p = CFG.params
    ch = choose_photo([face("ceiling", 200), face("mirror", 120), face("entrance", 100)], p)
    assert ch.best.source == "entrance" and len(ch.spares) == 2
    ch = choose_photo([face("ceiling", 60), FaceCandidate("c", 0, "body", "body.jpg", sharpness=0.5)], p)
    assert not ch.has_face and ch.best.path == "body.jpg"


def test_repeat_client_rules():
    p = CFG.params
    hist = [KnownFace("№3", "21.09", [1, 0, 0], t=0)]
    good = PhotoChoice(face(emb=[0.99, 0.1, 0]), has_face=True)
    assert repeat_status(good, hist, p, before=86400)[0] == "повторный, был 21.09"
    weak = PhotoChoice(face(emb=[0.99, 0.1, 0]), has_face=False)
    assert repeat_status(weak, hist, p, before=86400)[0].startswith("возможно повторный")
    assert repeat_status(good, hist, p, before=100 * 86400)[0] == ""   # старше 90 дней


def _track(cam, tid, zone, xy, reid=None, ir=False):
    return LocalTrack(cam, tid, zone, plan_xy=xy, reid=reid, ir=ir)


def test_global_ids_entrance_and_handover_between_cameras():
    m = GlobalIdManager(CFG)
    g = m.step(0, [_track("cam3", 1, "entrance", (0.0, 0.0), [1, 0, 0])])[("cam3", 1)]
    t = 0.5
    for i in range(6):
        t += 0.5
        a = m.step(t, [_track("cam3", 1, "hall_a", (0.2 * i, 0.0), [1, 0, 0]),
                       _track("cam4", 7, "hall_a", (0.2 * i + 0.1, 0.0), [1, 0, 0])])
        assert a[("cam3", 1)] == g and a[("cam4", 7)] == g
    assert sum(e.kind == "создан" for e in m.journal) == 1


def test_global_ids_new_id_mid_hall_is_flagged_and_lost_is_recovered():
    m = GlobalIdManager(CFG)
    g = m.step(0, [_track("cam3", 1, "entrance", (0, 0), [1, 0, 0])])[("cam3", 1)]
    m.step(0.5, [_track("cam3", 1, "hall_a", (0.3, 0), [1, 0, 0])])
    m.step(5, [])                                            # потерян
    back = m.step(60, [_track("cam4", 9, "hall_a", (1, 1), [0.98, 0.05, 0])])[("cam4", 9)]
    assert back == g
    other = m.step(61, [_track("cam4", 9, "hall_a", (1, 1), [0.98, 0.05, 0]),
                        _track("cam4", 10, "hall_a", (3, 3), [0, 0, 1])])[("cam4", 10)]
    assert other != g and not m.states[other].created_at_entrance


def test_global_ids_impossible_speed_not_matched():
    m = GlobalIdManager(CFG)
    g = m.step(0, [_track("cam3", 1, "entrance", (0, 0), [1, 0, 0])])[("cam3", 1)]
    a = m.step(0.5, [_track("cam3", 2, "entrance", (10, 0), [1, 0, 0])])
    assert a[("cam3", 2)] != g


def test_ir_frame_detection():
    gray = np.full((64, 64, 3), 120, dtype=np.uint8)
    color = gray.copy(); color[..., 2] = 220
    assert is_ir_frame(gray) and not is_ir_frame(color)


def test_rtsp_and_silence_monitor():
    assert stream_url("10.0.0.5", 3, user="admin", password="p@ss") == \
        "rtsp://admin:p%40ss@10.0.0.5:554/Streaming/Channels/302"
    alerts = []
    mon = SilenceMonitor(alerts.append)
    mon.frame("cam1", 0); mon.tick(60); mon.tick(130); mon.frame("cam1", 200)
    assert len(alerts) == 2 and mon.outages[0].start == 0 and mon.outages[0].end == 200


def test_csv_and_storage(tmp_path):
    b = demo_day()
    r = analyze_day(DayInput(b.ticks, b.persons, b.faces), CFG)
    csv_text = to_csv(r, CFG)
    assert "Регина;Алексей (мойка)" in csv_text and "09:30–12:36" in csv_text   # местное время UTC+4
    db = tmp_path / "s.db"
    storage.save_day(storage.connect(db), "2026-10-05", r, CFG)
    rows = storage.run_readonly_query(db, "SELECT master, count(*) FROM visits GROUP BY master ORDER BY master")
    assert rows == [("Алексей", 2), ("Ислам", 1), ("Регина", 1)]
    for bad in ("DELETE FROM visits", "SELECT 1; DROP TABLE visits", "SELECT * FROM track_events"):
        with pytest.raises(ValueError):
            storage.run_readonly_query(db, bad)


def test_salon_tz_offset():
    from datetime import datetime, timezone
    assert datetime(2026, 10, 5, 10, tzinfo=timezone.utc).astimezone(salon_tz(CFG)).hour == 14


def test_probe_hides_password_and_checks_file_source(tmp_path):
    import shutil
    import subprocess

    from salon_analytics.pipeline.probe import hide_password, run_probe
    assert hide_password("rtsp://admin:p%40ss@10.0.0.5:554/x: refused") == "rtsp://admin:***@10.0.0.5:554/x: refused"
    if not shutil.which("ffmpeg"):
        pytest.skip("нет ffmpeg")
    src = tmp_path / "cam3.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10",
                    "-t", "1", "-pix_fmt", "yuv420p", str(src)], check=True)
    res = run_probe(CFG, "", tmp_path / "out", "admin", "", sources={"cam3": str(src)}, timeout=10)
    assert res[0].ok and (res[0].width, res[0].height) == (640, 360)
    assert (tmp_path / "out" / "cam3.png").exists() and (tmp_path / "out" / "report.txt").exists()
