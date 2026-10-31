"""Basic project self-test. Run after installing requirements: python self_test.py

Covers posture geometry, landmark gating/smoothing, automatic (reference-based) posture evaluation,
camera placement checks, scoring and CSV export.
No camera is needed: landmarks are synthesized in a 1280x720 frame.
"""
from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

import numpy as np

from config import CONFIG
from game.active_break import ActiveBreak
from game.body_pose import BodyPose
from game.coin_game import COIN_RADIUS, CoinGame
from game.stretch_game import EXERCISES, StretchGame
from geometry import signed_lean_deg, signed_tilt_deg
from pose_engine import LandmarkPoint, LandmarkStabilizer, OneEuroFilter, PoseEngine, mirror_landmarks, side_overrides
from history import HistoryStore, filter_since, summarize
from notifications import PopupNotifier
from posture_analyzer import PostureAnalyzer, indicator_score, score_band
from session import SessionTracker

W, H = 1280.0, 720.0


def lp(px, py, vis=0.95):
    """Landmark from pixel coordinates in a 1280x720 frame."""
    return LandmarkPoint(px / W, py / H, 0.0, vis, W, H)


def front_pose(tilt_deg=0.0, head_shift_px=0.0, hips=True, lean_deg=0.0, eyes=True, ears=True, head_roll_deg=0.0,
               shoulder_half=150.0, ear_half=60.0, nose_yaw_px=0.0):
    """Seated person facing the camera; shoulders 300 px apart, centred at (640, 420)."""
    half = shoulder_half
    dy = math.tan(math.radians(tilt_deg)) * half
    mid = (640.0, 420.0)
    hx = mid[0] + head_shift_px
    roll = math.tan(math.radians(head_roll_deg))
    lms = {
        # image-left shoulder is the person's right shoulder (un-mirrored camera image)
        "right_shoulder": lp(mid[0] - half, mid[1] + dy),
        "left_shoulder": lp(mid[0] + half, mid[1] - dy),
        "nose": lp(hx + nose_yaw_px, 250),
    }
    if eyes:
        lms["right_eye"], lms["left_eye"] = lp(hx - 30, 225 + 30 * roll), lp(hx + 30, 225 - 30 * roll)
    if ears:
        lms["right_ear"], lms["left_ear"] = lp(hx - ear_half, 240 + ear_half * roll), lp(hx + ear_half, 240 - ear_half * roll)
    if hips:
        hip_mid = (mid[0] - math.tan(math.radians(lean_deg)) * 260, mid[1] + 260)
        lms["right_hip"], lms["left_hip"] = lp(hip_mid[0] - 110, hip_mid[1]), lp(hip_mid[0] + 110, hip_mid[1])
    return lms


def side_pose(facing=1, torso_deg=0.0, forward_head_deg=5.0, head_pitch_deg=0.0, side="left", far_vis=0.2,
              far_offset_px=5.0, roll_deg=0.0, nose_drop_px=25.0):
    """Person seen from the side, facing image-right (+1) or image-left (-1).

    torso_deg: trunk (hip -> shoulder) from vertical, + forward. forward_head_deg: neck (shoulder -> ear)
    relative to the trunk axis. head_pitch_deg: ear -> eye line below horizontal. The nose tip sits
    nose_drop_px below the ear level (anatomy), which must NOT affect the head-pitch reading.
    roll_deg rotates the whole picture (a tilted camera). far_offset_px shifts the far-side landmarks.
    """
    hip = (640.0, 600.0)
    t = math.radians(torso_deg)
    sh = (hip[0] + facing * math.sin(t) * 300, hip[1] - math.cos(t) * 300)
    a = math.radians(torso_deg + forward_head_deg)
    ear = (sh[0] + facing * math.sin(a) * 140, sh[1] - math.cos(a) * 140)
    p = math.radians(head_pitch_deg)
    eye = (ear[0] + facing * 45 * math.cos(p), ear[1] + 45 * math.sin(p))
    nose = (ear[0] + facing * 60, ear[1] + nose_drop_px)
    near = {"ear": ear, "eye": eye, "shoulder": sh, "hip": hip}
    other = "right" if side == "left" else "left"

    def rot(pt):
        if not roll_deg:
            return pt
        r, (cx, cy) = math.radians(roll_deg), (640.0, 450.0)
        dx, dy = pt[0] - cx, pt[1] - cy
        return (cx + dx * math.cos(r) - dy * math.sin(r), cy + dx * math.sin(r) + dy * math.cos(r))

    lms = {"nose": lp(*rot(nose))}
    for joint, pt in near.items():
        lms[f"{side}_{joint}"] = lp(*rot(pt))
        lms[f"{other}_{joint}"] = lp(*rot((pt[0] - far_offset_px, pt[1])), far_vis)
    return lms


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol


def test_geometry():
    # Pixel-space angles: a 5-degree line stays 5 degrees regardless of the frame aspect ratio.
    assert approx(signed_tilt_deg((0, 10), (100, 10 - math.tan(math.radians(5)) * 100)), 5.0)
    assert approx(signed_tilt_deg((100, 0), (0, 10)), math.degrees(math.atan2(10, 100)))  # order independent
    assert approx(signed_lean_deg((0, 100), (10, 0), forward_x=1), math.degrees(math.atan2(10, 100)))
    assert approx(signed_lean_deg((0, 100), (10, 0), forward_x=-1), -math.degrees(math.atan2(10, 100)))
    # Normalized coordinates would distort the same 5-degree line in a 16:9 frame (the old bug).
    p, q = lp(500, 400), lp(800, 400 - math.tan(math.radians(5)) * 300)
    assert approx(signed_tilt_deg(p.pxy, q.pxy), 5.0)
    assert signed_tilt_deg(p.xy, q.xy) > 8.0


def test_one_euro():
    rng = random.Random(1)
    f = OneEuroFilter(CONFIG.landmark_filter_min_cutoff, CONFIG.landmark_filter_beta, CONFIG.landmark_filter_d_cutoff)
    raw = [100 + rng.gauss(0, 3) for _ in range(120)]
    out = [f(x, i / 20) for i, x in enumerate(raw)]
    assert np.std(out[20:]) < 0.5 * np.std(raw[20:]), "jitter must be reduced"
    f2 = OneEuroFilter(CONFIG.landmark_filter_min_cutoff, CONFIG.landmark_filter_beta, CONFIG.landmark_filter_d_cutoff)
    values = [f2(0.0 if i < 10 else 200.0, i / 20) for i in range(20)]
    assert values[-1] > 190, "a real movement must be followed within ~0.5 s"


def test_stabilizer():
    st = LandmarkStabilizer(0.5)
    t = 0.0
    out = st.update({"nose": lp(640, 250, 0.9), "left_wrist": lp(900, 300, 0.2), "right_wrist": lp(1400, 300, 0.9)}, t)
    assert "nose" in out and "left_wrist" not in out, "low-confidence landmarks are hidden"
    assert "right_wrist" not in out, "out-of-frame landmarks are hidden"
    t += 0.05
    out = st.update({"nose": lp(640, 250, 0.45)}, t)
    assert "nose" in out, "hysteresis keeps a shown landmark slightly below the threshold"
    t += 0.05
    out = st.update({"nose": lp(640, 250, 0.35)}, t)
    assert "nose" not in out
    for _ in range(3):
        t += 0.05
        st.update({"nose": lp(640, 250, 0.9)}, t)
    t += 0.05
    out = st.update({"nose": lp(1200, 650, 0.9)}, t)   # teleport across the frame
    assert "nose" not in out, "an implausible jump is rejected"
    for _ in range(CONFIG.landmark_outlier_max_frames):
        t += 0.05
        out = st.update({"nose": lp(1200, 650, 0.9)}, t)
    assert "nose" in out and approx(out["nose"].x * W, 1200, 1.0), "a jump that persists is accepted"
    assert st.update(None, t + 0.05) is None and not st.tracks, "losing the pose clears all tracks"


def test_mirror_and_drawing():
    lms = {"nose": lp(100, 200), "left_shoulder": lp(300, 400)}
    m = mirror_landmarks(lms)
    assert approx(m["nose"].x, 1 - lms["nose"].x, 1e-9) and m["left_shoulder"].y == lms["left_shoulder"].y
    # The posture skeleton must never draw arms or hands.
    pose = PoseEngine.__new__(PoseEngine)
    frame = np.zeros((720, 1280, 3), np.uint8)
    lms = front_pose()
    lms["left_wrist"], lms["right_wrist"] = lp(1150, 150), lp(130, 150)
    lms["left_elbow"], lms["right_elbow"] = lp(1000, 250), lp(280, 250)
    pose.draw_front(frame, lms, ["shoulder"])
    assert frame[150, 1150].sum() == 0 and frame[150, 130].sum() == 0 and frame[250, 1000].sum() == 0
    assert frame[420, 640].sum() > 0, "shoulder line is drawn"
    side = np.zeros((720, 1280, 3), np.uint8)
    pose.draw_side(side, side_pose(far_vis=0.9), "left")
    assert side.sum() > 0


def test_front_features():
    an = PostureAnalyzer()
    f = an.extract_front(front_pose(tilt_deg=5.0), now=0)
    assert f.valid and approx(f.values["shoulder_tilt_deg"], 5.0, 0.01), "true pixel-space tilt"
    f = PostureAnalyzer().extract_front(front_pose(tilt_deg=-4.0, head_shift_px=60, lean_deg=8, head_roll_deg=-7), now=0)
    assert approx(f.values["shoulder_tilt_deg"], -4.0, 0.01), "tilt keeps its direction"
    assert approx(f.values["head_tilt_deg"], -7.0, 0.05)
    assert approx(f.values["side_lean_deg"], 8.0, 0.05)
    assert approx(f.values["neck_lateral_deg"], math.degrees(math.atan2(60, 180)), 0.05), "shoulder-centre -> ear-centre"
    f = PostureAnalyzer().extract_front(front_pose(hips=False, eyes=False), now=0)
    assert f.valid and "side_lean_deg" not in f.values and "head_tilt_deg" in f.values, "ear line backs up the eyes"
    turned = PostureAnalyzer().extract_front(front_pose(nose_yaw_px=50), now=0)
    assert turned.valid and "head_tilt_deg" not in turned.values and "face the camera" in turned.issue
    # camera / placement checks give guidance instead of a measurement
    far = PostureAnalyzer().extract_front(front_pose(shoulder_half=50, ear_half=20), now=0)
    assert not far.valid and "move closer" in far.issue
    rotated = PostureAnalyzer().extract_front(front_pose(shoulder_half=80, ear_half=60), now=0)
    assert not rotated.valid and "turn your body" in rotated.issue
    bad = front_pose()
    bad["nose"] = lp(640, 600)
    assert "rotated" in PostureAnalyzer().extract_front(bad, now=0).issue
    missing = front_pose()
    del missing["left_shoulder"]
    assert "both shoulders" in PostureAnalyzer().extract_front(missing, now=0).issue


def test_side_features():
    for facing in (1, -1):
        s = PostureAnalyzer().extract_side(side_pose(facing=facing, torso_deg=12.0, forward_head_deg=18,
                                                     head_pitch_deg=30), now=0)
        assert s.valid and approx(s.values["torso_lean_deg"], 12.0, 0.01), f"facing {facing}"
        assert approx(s.values["forward_head_deg"], 18.0, 0.01), "neck measured relative to the trunk"
        assert approx(s.values["head_flexion_deg"], 30.0 - CONFIG.head_pitch_neutral_deg, 0.01)
    # Forward head is independent of whole-body lean/recline (root cause of false warnings).
    for torso in (-20.0, 0.0, 15.0):
        fh = PostureAnalyzer().extract_side(side_pose(torso_deg=torso, forward_head_deg=8), now=0).values
        assert approx(fh["forward_head_deg"], 8.0, 0.01), f"torso {torso}"
    # A rolled (tilted) side camera does not change forward head (both angles shift equally).
    rolled = PostureAnalyzer().extract_side(side_pose(forward_head_deg=8, roll_deg=8), now=0).values
    assert approx(rolled["forward_head_deg"], 8.0, 0.01) and abs(rolled["torso_lean_deg"]) > 7
    # Head pitch uses the ear -> eye line: where the nose tip sits does not matter.
    for drop in (5.0, 25.0, 45.0):
        hp = PostureAnalyzer().extract_side(side_pose(head_pitch_deg=0, nose_drop_px=drop), now=0).values
        assert approx(hp["head_flexion_deg"], 0.0, 0.01), "level head reads level"
    # Oblique camera: far-side joints shifted sideways are averaged consistently (midline for ear AND
    # shoulder AND hip), so the lateral ear offset is not read as a forward head.
    oblique = PostureAnalyzer().extract_side(side_pose(forward_head_deg=8, far_vis=0.9, far_offset_px=80), now=0)
    assert oblique.valid and approx(oblique.values["forward_head_deg"], 8.0, 0.01)
    assert PostureAnalyzer().extract_side(side_pose(torso_deg=-10.0), now=0).values["torso_lean_deg"] < 0, "recline is negative"
    # Side selection hysteresis: a marginally better far side does not steal tracking.
    an = PostureAnalyzer()
    an.extract_side(side_pose(side="left", far_vis=0.2), now=0)
    close = side_pose(side="left", far_vis=0.2)
    for name in ("right_ear", "right_shoulder", "right_hip"):
        close[name] = lp(close[name].x * W, close[name].y * H, 0.99)
    an.extract_side(close, now=0)
    assert an.side_choice == "left", "no switch for a marginal difference"
    an.extract_side(side_pose(side="right", far_vis=0.1), now=0)
    assert an.side_choice == "right", "switches when the other side is clearly better"
    # Near hip hidden but the far hip visible -> still measurable.
    mixed = side_pose(side="left", far_vis=0.0)
    del mixed["left_hip"]
    mixed["right_hip"] = lp(640, 600, 0.9)
    assert PostureAnalyzer().extract_side(mixed, now=0).valid
    # No hip / no eye -> guidance or just that indicator missing, never a fake measurement.
    hidden = side_pose(far_vis=0.0)
    del hidden["left_hip"]
    res = PostureAnalyzer().extract_side(hidden, now=0)
    assert not res.valid and "hip not visible" in res.issue
    no_eye = side_pose(far_vis=0.0)
    del no_eye["left_eye"]
    res = PostureAnalyzer().extract_side(no_eye, now=0)
    assert res.valid and "head_flexion_deg" not in res.values and "forward_head_deg" in res.values
    # Oblique side camera: warn, and refuse when too frontal.
    warn = PostureAnalyzer().extract_side(side_pose(far_vis=0.9, far_offset_px=120), now=0)
    assert warn.valid and "angled" in warn.issue
    refuse = PostureAnalyzer().extract_side(side_pose(far_vis=0.9, far_offset_px=200), now=0)
    assert not refuse.valid and "90" in refuse.issue
    # Brief drop-outs are bridged with the last reading, then reported.
    an = PostureAnalyzer()
    an.extract_side(side_pose(), now=0.0)
    held = an.extract_side(None, now=0.3)
    assert held.valid and held.held
    assert not an.extract_side(None, now=0.3 + CONFIG.measurement_hold_seconds + 0.1).valid


def test_side_hip_threshold():
    st = LandmarkStabilizer(CONFIG.side_landmark_visibility_threshold, side_overrides())
    out = st.update({"left_hip": lp(640, 600, 0.4), "left_ear": lp(600, 200, 0.4)}, 0.0)
    assert "left_hip" in out and "left_ear" not in out, "seated hips get their own (lower) threshold"


def evaluate(an, front, side, t):
    return an.classify(an.extract_front(front, now=t), an.extract_side(side, now=t), 0, now=t)


class Feed:
    """Drives a PopupNotifier with simulated monotonic timestamps at a fixed frame rate."""

    def __init__(self, notifier, fps=20):
        self.n, self.t, self.dt, self.actions = notifier, 0.0, 1.0 / fps, []

    def run(self, seconds, risk="WARNING", reliable=True):
        for _ in range(round(seconds / self.dt)):
            self.t += self.dt
            action = self.n.update(self.t, risk, reliable)
            if action:
                self.actions.append((round(self.t, 2), action))
                if action == "close":
                    self.n.closed()   # what the GUI does when it closes the popup
        return self


def test_popup_notifier():
    trigger, visible, recovery = CONFIG.popup_trigger_seconds, CONFIG.popup_visible_seconds, CONFIG.popup_recovery_seconds
    # Shown only after `trigger` seconds of confirmed bad posture, never before.
    f = Feed(PopupNotifier()).run(trigger - 0.1)
    assert f.actions == [] and f.n.visible is None
    f.run(0.2)
    first_bad = f.dt   # time is measured from the first confirmed bad frame
    assert f.actions == [(round(first_bad + trigger, 2), "show")] and f.n.visible == "posture", f.actions
    # Auto-closes after being visible `visible` seconds; the same episode never reopens it.
    f.run(visible - 0.1)
    assert f.n.visible == "posture"
    f.run(0.2)
    assert f.actions[-1] == (round(first_bad + trigger + visible, 2), "close") and f.n.visible is None
    f.run(60)
    assert len(f.actions) == 2, "one popup per episode"
    # A short GOOD moment does not end the episode; `recovery` seconds of GOOD do.
    f.run(recovery - 0.5, "GOOD").run(5)
    assert len(f.actions) == 2 and f.n.notified
    f.run(recovery + 0.1, "GOOD")
    assert not f.n.notified and f.n.bad_seconds == 0
    f.run(trigger + 0.1)
    assert f.actions[-1][1] == "show", "a new episode can warn again"

    # Dismiss (user closes): not reopened while the episode continues.
    f = Feed(PopupNotifier()).run(trigger + 1)
    f.n.closed()
    f.run(30)
    assert [a for _, a in f.actions] == ["show"]
    # GOOD confirmed while visible closes it early and resets the episode.
    f = Feed(PopupNotifier()).run(trigger + 1).run(recovery + 0.1, "GOOD")
    assert [a for _, a in f.actions] == ["show", "close"] and f.actions[-1][0] < trigger + 1 + visible
    assert not f.n.notified

    # Unreliable data pauses the count (neither adds bad time nor warns) and interrupts recovery.
    f = Feed(PopupNotifier()).run(10).run(20, "N/A", reliable=False).run(4.9)
    assert f.actions == [] and approx(f.n.bad_seconds, 14.9, 0.06)
    f.run(0.2)
    assert f.actions[-1][1] == "show"
    f = Feed(PopupNotifier()).run(trigger + 1)
    f.run(2, "GOOD").run(1, "N/A", reliable=False).run(2, "GOOD")
    assert f.n.notified and f.n.visible == "posture", "recovery must be 3 *continuous* GOOD seconds"
    # Frame stalls cannot fast-forward the timer.
    n = PopupNotifier()
    n.update(0.0, "WARNING", True)
    assert n.update(30.0, "WARNING", True) is None and n.bad_seconds <= CONFIG.session_max_tick_seconds

    # Snooze: no posture popup for popup_snooze_seconds, even for a new episode; afterwards it may show.
    f = Feed(PopupNotifier()).run(trigger + 1)
    f.n.snooze(f.t)
    f.n.closed()
    snooze_end = f.t + CONFIG.popup_snooze_seconds
    f.run(recovery + 0.1, "GOOD").run(trigger + 5)
    assert [a for _, a in f.actions] == ["show"] and f.t < snooze_end
    f.run(snooze_end - f.t + 0.1)
    assert [a for _, a in f.actions] == ["show", "show"] and f.actions[-1][0] >= snooze_end

    # Only one popup at a time: the break reminder blocks the posture popup and vice versa.
    n = PopupNotifier()
    n.opened("break", 0.0)
    f = Feed(n).run(trigger + 1)
    assert [a for _, a in f.actions] == ["close", "show"], "break popup auto-closes, posture popup only after it"
    assert f.actions[0][0] == round(visible, 2) and f.actions[1][0] > f.actions[0][0]
    assert f.n.visible == "posture" and not f.n.break_popup_allowed(), "no break reminder over a posture popup"


def test_history_store():
    import sqlite3
    import time as _time
    from datetime import datetime
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "data" / "history.sqlite3"
        store = HistoryStore(db)
        assert store.available and store.sessions() == [] and db.exists(), "created on first use"
        session = SessionTracker()
        step(session, 0.4, "monitoring", ["Forward Head"], "WARNING", 70)
        store.save(session.snapshot())
        assert store.flush() and store.sessions() == [], "too little valid monitoring: not stored"
        session.monitored_seconds += CONFIG.history_min_monitored_seconds          # pretend a longer session
        session.durations["Warning"] += CONFIG.history_min_monitored_seconds
        session.score_time_sum += 70 * CONFIG.history_min_monitored_seconds
        for _ in range(3):                                                          # periodic saves, pause, resume
            store.save(session.snapshot())
        session.pause()
        store.save(session.snapshot())
        session.resume()
        step(session, 0.3, "monitoring", ["Slouching"], "POOR", 40)
        store.save(session.snapshot())
        store.flush()
        rows = store.sessions()
        assert len(rows) == 1 and rows[0].id == session.session_id, "one row per session (upsert)"
        r = rows[0]
        assert r.ended_at is None and not r.recovered, "still open"
        assert r.score == session.average_score, "same score as the dashboard"
        p = session.percentages()
        assert (r.good_percent, r.warning_percent, r.poor_percent) == (p["Good Posture"], p["Warning"], p["Poor"])
        assert approx(r.monitored_seconds, session.monitored_seconds, 1e-6)
        assert r.top_issue[0] == "Forward Head" and approx(r.slouching_seconds, 0.3, 0.02)
        store.save(session.snapshot(), final=True)
        store.close()

        # Next app start: data is loaded; an unfinished session (crash) is recovered, never deleted.
        crashed = SessionTracker()
        crashed.monitored_seconds, crashed.score_time_sum = 60.0, 60.0 * 90
        crashed.durations["Good Posture"] = 60.0
        crashed.started_at -= 3 * 86400
        store = HistoryStore(db)
        store.save(crashed.snapshot())
        store.flush()
        store._thread = None   # simulate the process dying: no final save
        store = HistoryStore(db)
        rows = store.sessions()
        assert [x.id for x in rows] == [session.session_id, crashed.session_id], "newest first"
        assert rows[0].ended_at is not None and not rows[0].recovered
        assert rows[1].recovered and rows[1].ended_at is not None and rows[1].score == 90
        totals = summarize(rows)
        assert totals["count"] == 2 and totals["average_score"] == round(
            (r.average_score * r.monitored_seconds + 90 * 60) / (r.monitored_seconds + 60)), "time-weighted"
        # Local-calendar filters.
        now = datetime.now()
        assert filter_since("all") is None
        assert [x.id for x in store.sessions(filter_since("today", now))] == [session.session_id]
        assert len(store.sessions(filter_since("7d", now))) == 2
        assert filter_since("today", now) == now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        store.close()

        # Failures never crash the app and never delete data.
        blocker = Path(tmp) / "not_a_dir"
        blocker.write_text("x")
        bad = HistoryStore(blocker / "history.sqlite3")
        assert not bad.available and "unavailable" in bad.error and bad.sessions() == []
        bad.save(session.snapshot())    # ignored, no exception
        corrupt = Path(tmp) / "corrupt.sqlite3"
        corrupt.write_bytes(b"definitely not a database" * 100)
        broken = HistoryStore(corrupt)
        assert not broken.available and broken.sessions() == []
        assert corrupt.read_bytes().startswith(b"definitely not"), "a damaged file is left untouched"
        assert len(HistoryStore(db).sessions()) == 2, "other data intact"
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2

        # Regression: incomplete / damaged rows are skipped (not shown as zeros) and never crash or get deleted.
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO sessions (id, started_at, updated_at) VALUES ('no_stats', 1, 1)")
            conn.execute("INSERT INTO sessions (id, started_at, updated_at, monitored_seconds, good_percent) "
                         "VALUES ('bad', 2, 2, 'oops', 'n/a')")
            conn.execute("INSERT INTO sessions (id, started_at, updated_at, monitored_seconds) VALUES ('partial', 3, 3, 42)")
        rows = HistoryStore(db).sessions()
        ids = [x.id for x in rows]
        assert "no_stats" not in ids and "bad" not in ids and len(rows) == 3
        part = rows[-1]
        assert part.id == "partial" and part.score is None and part.good_percent == 0.0 and part.top_issue is None
        assert summarize(rows)["count"] == 3
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 5, "damaged rows kept on disk"

        # Regression: a live session flagged 'recovered' by a second app instance is un-flagged on its next save.
        live = SessionTracker()
        live.monitored_seconds, live.score_time_sum = 30.0, 30.0 * 80
        first = HistoryStore(db)
        first.save(live.snapshot())
        first.flush()
        HistoryStore(db).close()                      # another instance starts: marks open sessions recovered
        first.save(live.snapshot())
        first.flush()
        again = next(x for x in first.sessions() if x.id == live.session_id)
        assert not again.recovered and again.ended_at is None
        first.close()


def game_pose(nose=(0.5, 0.35), wrists=((0.38, 0.85), (0.62, 0.85)), elbows=((0.33, 0.70), (0.67, 0.70)),
              wrist_vis=0.95, elbow_vis=0.95, shoulders=((0.40, 0.55), (0.60, 0.55))):
    """BodyPose in displayed (mirrored) game coordinates of a 4:3 frame."""
    def p(x, y, v=0.95):
        return LandmarkPoint(x, y, 0.0, v, 1280, 960)
    return BodyPose.from_landmarks({
        "nose": p(*nose), "left_shoulder": p(*shoulders[0]), "right_shoulder": p(*shoulders[1]),
        "left_wrist": p(*wrists[0], wrist_vis), "right_wrist": p(*wrists[1], wrist_vis),
        "left_elbow": p(*elbows[0], elbow_vis), "right_elbow": p(*elbows[1], elbow_vis)}, aspect=4 / 3)


def play(game, pose, seconds, t, fps=20):
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        game.update(t, pose)
    return t


def test_active_break_games():
    assert [e.key for e in EXERCISES][0] == "raise_arms"
    rest = game_pose()
    raised = game_pose(wrists=((0.38, 0.12), (0.62, 0.12)), elbows=((0.37, 0.27), (0.63, 0.27)))
    raised_out = game_pose(wrists=((0.35, -0.05), (0.65, -0.05)), elbows=((0.36, 0.18), (0.64, 0.18)), wrist_vis=0.05)
    hidden_arms = game_pose(wrist_vis=0.05, elbow_vis=0.05)
    hands_low_elbows_up = game_pose(wrists=((0.38, 0.6), (0.62, 0.6)), elbows=((0.36, 0.2), (0.64, 0.2)))

    # Detection: hands above the top of the frame are judged by the elbows; visible low hands never count.
    assert raised_out.arms_raised() and raised.arms_raised() and not hands_low_elbows_up.arms_raised()
    assert not rest.arms_raised() and not hidden_arms.arms_raised()
    g = StretchGame()
    play(g, raised_out, 2.0, 0.0)
    assert g.completed == 1 and g.score == CONFIG.stretch_points, "raise arms counted with hands out of frame"

    # The hold starts only with a valid pose; status feedback.
    g = StretchGame()
    t = play(g, rest, 0.5, 0.0)
    assert g.hold_started is None and g.hold_progress(t) == 0 and g.status == "adjust"
    play(g, hidden_arms, 0.2, t)
    assert g.status == "not_visible"
    play(g, None, 0.2, t + 0.2)
    assert g.status == "not_visible" and not g.matching
    t = play(g, raised, 0.5, t + 0.4)
    assert g.status == "detected" and 0.2 < g.hold_progress(t) < 0.5

    # Missing landmarks never count and never skip the pose.
    g = StretchGame()
    play(g, None, 30.0, 0.0)
    assert g.completed == 0 and g.index == 0 and g.message == "", "no skip while the body is not tracked"

    # A short tracking drop pauses the hold (lost time not counted); a long one drops it.
    g = StretchGame()
    t = play(g, raised, 1.0, 0.0)
    t = play(g, None, 0.6, t)
    assert 0.6 < g.hold_progress(t) < 0.7, "progress frozen while tracking is lost"
    t = play(g, raised, 0.4, t)
    assert g.completed == 0, "lost time is not counted toward the hold"
    play(g, raised, 0.2, t)
    assert g.completed == 1
    g = StretchGame()
    t = play(g, raised, 1.0, 0.0)
    t = play(g, None, 1.5, t)
    t = play(g, raised, 0.8, t)
    assert g.completed == 0 and g.hold_progress(t) < 0.6, "hold restarted after a long drop-out"

    # Skip only after stretch_exercise_timeout_seconds of *tracked* attempts; once; no duplicate scoring.
    g = StretchGame()
    t = play(g, rest, CONFIG.stretch_exercise_timeout_seconds - 0.5, 0.0)
    t = play(g, None, 20.0, t)                       # camera lag / out of view: timer paused
    assert g.index == 0
    t = play(g, rest, 1.0, t)
    assert g.index == 1 and g.message == "Skipped" and g.score == 0 and g.completed == 0
    g = StretchGame()
    t = play(g, raised, 6.0, 0.0)                    # holding far longer than needed
    assert g.completed == 1 and g.score == CONFIG.stretch_points and g.index == 1, "scored once"
    # A camera stall cannot fast-forward the skip timer.
    g = StretchGame()
    g.update(0.0, rest)
    g.update(30.0, rest)
    assert g.index == 0

    # Coins: a hand on the coin scores once, a hand elsewhere never.
    cg = CoinGame(rng=random.Random(3))
    t = play(cg, rest, 0.2, 0.0)
    coin = cg.coin
    far = game_pose(wrists=((0.02 if coin.x > 0.5 else 0.98, 0.95), (0.02 if coin.x > 0.5 else 0.98, 0.95)))
    t = play(cg, far, 0.5, t)
    assert cg.coins_collected == 0
    on_coin = game_pose(wrists=((coin.x, coin.y), (coin.x + 0.5 * COIN_RADIUS / cg.aspect, coin.y)))
    cg.update(t + 0.05, on_coin)
    assert cg.coins_collected == 1 and cg.score == CONFIG.coin_points, "both hands on one coin: +5 once"
    assert cg.coin is not coin, "collected coin replaced"
    # Bombs: dodging toward the correct side (displayed coordinates) scores, standing still is a hit.
    for side, dodge_nose, expect in (("left", (0.62, 0.35), "avoided"), ("right", (0.38, 0.35), "avoided"),
                                     ("top", (0.5, 0.48), "avoided"), ("left", (0.5, 0.35), "hit"),
                                     ("left", (0.38, 0.35), "hit")):
        cg = CoinGame(rng=random.Random(1))
        t = play(cg, rest, 1.0, 0.0)
        cg._spawn_bomb(t, side)
        t = play(cg, game_pose(nose=dodge_nose), 2.2, t)
        got = "avoided" if cg.bombs_avoided == 1 and cg.bomb_hits == 0 else ("hit" if cg.bomb_hits == 1 else "?")
        assert got == expect, (side, dodge_nose, got)

    # Full sequence Stretch -> Coin -> Stretch -> Coin -> results; statistics add up; finishes.
    ab = ActiveBreak(0.0, rng=random.Random(5))
    seen, t, done_at = [], 0.0, None
    while t < 200 and done_at is None:
        t += 0.05
        ab.update(t, raised if ab.state == "stretch" else rest)
        if ab.state in ("stretch", "coin") and (not seen or seen[-1] != ab.state):
            seen.append(ab.state)
        if ab.finished(t):
            done_at = t
    assert seen == ["stretch", "coin", "stretch", "coin"] and ab.state == "results" and done_at is not None, seen
    games = ab.finished_games
    assert ab.score == sum(x.score for x in games) and ab.stretches_completed == sum(
        x.completed for x in games if isinstance(x, StretchGame)) and ab.stretches_completed >= 2
    assert abs(ab.played - CONFIG.active_break_total_seconds) < 1e-6


def run_frames(an, front, side, start, seconds, persistent=0.0, fps=20):
    res = None
    for i in range(int(seconds * fps)):
        t = start + i / fps
        res = an.classify(an.extract_front(front, now=t), an.extract_side(side, now=t), persistent, now=t)
    return res


def test_classification():
    assert indicator_score(0) == 100 and indicator_score(1) == 80 and indicator_score(2) == 40 and indicator_score(3) == 0
    assert score_band("GOOD") == (80, 100) and score_band("WARNING") == (60, 79) and score_band("POOR") == (0, 59)
    an = PostureAnalyzer()
    good = evaluate(an, front_pose(), side_pose(), 0.0)
    assert good.risk_level == "GOOD" and good.labels == ["Good Posture"] and good.score >= 80, good
    assert all(v is not None for v in good.components.values())

    # A reasonably upright computer posture (slight lean back, head a little forward, looking slightly
    # down at the screen) must not raise neck warnings.
    upright = run_frames(PostureAnalyzer(), front_pose(),
                         side_pose(torso_deg=-8, forward_head_deg=12, head_pitch_deg=10), 0.0, 3.0)
    assert upright.labels == ["Good Posture"], upright.labels
    # Clearly forward head and sustained neck flexion are still detected, separately.
    fh = run_frames(PostureAnalyzer(), front_pose(), side_pose(forward_head_deg=32), 0.0, 2.0)
    assert fh.metrics["Forward head"][3] and not fh.metrics["Neck flexion"][3]
    nf = run_frames(PostureAnalyzer(), front_pose(), side_pose(forward_head_deg=5, head_pitch_deg=30), 0.0, 2.0)
    assert nf.metrics["Neck flexion"][3] and not nf.metrics["Forward head"][3]
    # Leaning the whole trunk forward is Slouching, not Forward Head.
    lean = run_frames(PostureAnalyzer(), front_pose(), side_pose(torso_deg=28, forward_head_deg=5), 0.0, 2.0)
    assert "Slouching" in lean.labels and "Forward Head" not in lean.labels

    # Flags only after metric_enter_seconds (brief movements are ignored); score stays in the status band.
    an = PostureAnalyzer()
    bad_front, bad_side = front_pose(tilt_deg=9), side_pose(torso_deg=28, forward_head_deg=35)
    first = evaluate(an, bad_front, bad_side, 0.0)
    assert first.risk_level == "GOOD" and first.raw_score < 80 <= first.score, "not flagged instantly"
    later = evaluate(an, bad_front, bad_side, CONFIG.metric_enter_seconds + 0.1)
    assert {"Forward Head", "Slouching", "Uneven Shoulder"} <= set(later.labels) and later.risk_level == "WARNING"
    assert 60 <= later.score <= 79, "WARNING score is in the warning band (consistent colour)"

    # Severity = magnitude AND duration.
    mild = run_frames(PostureAnalyzer(), front_pose(tilt_deg=5.5), side_pose(), 0.0, 3.0, persistent=60)
    assert mild.risk_level == "WARNING" and 60 <= mild.score <= 79, "minor deviation stays WARNING however long"
    moderate = PostureAnalyzer()
    assert run_frames(moderate, front_pose(tilt_deg=6.5), side_pose(), 0.0, 3.0, persistent=10).risk_level == "WARNING"
    long_moderate = run_frames(moderate, front_pose(tilt_deg=6.5), side_pose(), 3.0, 1.0, persistent=25)
    assert long_moderate.risk_level == "POOR" and long_moderate.score <= 59, "not-minor + long -> POOR"
    severe_an = PostureAnalyzer()
    early = run_frames(severe_an, front_pose(tilt_deg=9), side_pose(), 0.0, 2.0)
    assert early.risk_level == "WARNING"
    severe = run_frames(severe_an, front_pose(tilt_deg=9), side_pose(), 2.0, CONFIG.poor_severe_seconds)
    assert severe.risk_level == "POOR" and severe.score <= 59, "severe deviation -> POOR without waiting 20 s"

    # Hysteresis: settling just under the limit is not enough; clearly inside for metric_exit_seconds clears it.
    an = PostureAnalyzer()
    run_frames(an, front_pose(tilt_deg=6), side_pose(), 0.0, 2.0)
    nearly = run_frames(an, front_pose(tilt_deg=4.8), side_pose(), 3.0, 3.0)
    assert "Uneven Shoulder" in nearly.labels, "4.8 deg is inside the limit but above the hysteresis band"
    cleared = run_frames(an, front_pose(tilt_deg=1), side_pose(), 6.0, 3.0)
    assert cleared.labels == ["Good Posture"] and cleared.score >= 80, cleared.labels

    # Supported recline is acceptable; extreme recline and a hunched combination are not.
    recline = run_frames(PostureAnalyzer(), front_pose(), side_pose(torso_deg=-20), 0.0, 2.0)
    assert recline.labels == ["Good Posture"]
    too_far = run_frames(PostureAnalyzer(), front_pose(), side_pose(torso_deg=-40), 0.0, 2.0)
    assert "Slouching" in too_far.labels
    hunched = run_frames(PostureAnalyzer(), front_pose(),
                         side_pose(torso_deg=0.8 * CONFIG.torso_forward_max_deg,
                                   forward_head_deg=0.8 * CONFIG.forward_head_max_deg), 0.0, 2.0)
    assert "Slouching" in hunched.labels and hunched.metrics["Hunched posture"][3]
    assert not hunched.metrics["Trunk forward lean"][3] and not hunched.metrics["Forward head"][3]

    # Head tilt and side lean map to the alignment label.
    tilted = run_frames(PostureAnalyzer(), front_pose(head_roll_deg=15, lean_deg=12), side_pose(), 0.0, 2.0)
    assert "Leaning Left/Right" in tilted.labels and tilted.components["Alignment"] < 80

    # One camera lost: the other camera's indicators are still evaluated; the missing ones stay unavailable.
    an = PostureAnalyzer()
    front_only = evaluate(an, front_pose(tilt_deg=1), None, 0.0)
    assert front_only.partial and front_only.risk_level == "GOOD" and front_only.score is not None
    assert front_only.components["Neck"] is None and front_only.components["Back"] is None
    assert front_only.components["Shoulder"] is not None and "Forward head" not in front_only.metrics
    assert any("Side camera" in g for g in front_only.guidance) and "unavailable" in front_only.reasons[0]
    side_only = run_frames(PostureAnalyzer(), None, side_pose(torso_deg=30), 0.0, 2.0)
    assert side_only.partial and "Slouching" in side_only.labels and side_only.components["Shoulder"] is None
    # Nothing reliable at all: Insufficient Data, never a score.
    none = evaluate(PostureAnalyzer(), None, None, 0.0)
    assert none.risk_level == "N/A" and none.score is None and none.status == "Insufficient Data"
    partial = evaluate(PostureAnalyzer(), front_pose(eyes=False, ears=False, hips=False), side_pose(), 0.0)
    assert "Head tilt" not in partial.metrics and "Side lean" not in partial.metrics and partial.score is not None


def step(session, dt, state, labels=(), risk="", score=None):
    import time as _time
    session.last_update = _time.time() - dt
    session.tick(state, labels, risk, score)


def test_session_statistics():
    s = SessionTracker()
    step(s, 0.4, "monitoring", ["Good Posture"], "GOOD", 90)
    step(s, 0.2, "monitoring", ["Forward Head", "Slouching"], "WARNING", 70)
    step(s, 0.5, "insufficient")
    step(s, 0.3, "partial", ["Uneven Shoulder"], "WARNING", 50)
    step(s, 0.5, "break")
    step(s, 3.0, "monitoring", ["Good Posture"], "GOOD", 100)   # stall: clamped to session_max_tick_seconds
    cap = CONFIG.session_max_tick_seconds
    assert approx(s.monitored_seconds, 0.6 + cap, 0.02), "only valid monitoring counts"
    assert approx(s.insufficient_seconds, 0.5, 0.02) and approx(s.partial_seconds, 0.3, 0.02)
    assert approx(s.break_seconds, 0.5, 0.02)
    d = s.durations
    assert approx(d["Good Posture"] + d["Warning"] + d["Poor"], s.monitored_seconds, 0.02), "consistent denominator"
    assert approx(d["Forward Head"], 0.2, 0.02) and approx(d["Slouching"], 0.2, 0.02), "conditions may overlap"
    assert d["Uneven Shoulder"] == 0.0, "partial time is not counted"
    expected = (90 * 0.4 + 70 * 0.2 + 100 * cap) / (0.6 + cap)
    assert s.average_score == round(expected), "time-weighted average"
    # Frame rate does not matter: ten 0.04 s frames == one 0.4 s frame.
    a, b = SessionTracker(), SessionTracker()
    for _ in range(10):
        step(a, 0.04, "monitoring", ["Good Posture"], "GOOD", 90)
    step(a, 0.2, "monitoring", ["Slouching"], "WARNING", 60)
    step(b, 0.4, "monitoring", ["Good Posture"], "GOOD", 90)
    for _ in range(5):
        step(b, 0.04, "monitoring", ["Slouching"], "WARNING", 60)
    assert a.average_score == b.average_score == 80 and a.percentages() == b.percentages()


def test_session_csv():
    session = SessionTracker()
    step(session, 0.3, "monitoring", ["Good Posture"], "GOOD", 0)
    assert session.average_score == 0, "zero is a valid score"
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(session.save_csv(tmp))
        rows = dict(line.split(",", 1) for line in csv_path.read_text().splitlines()[1:])
        assert {"session_seconds", "average_posture_score", "good_percent", "duration_Forward Head",
                "monitored_seconds", "insufficient_data_seconds", "active_break_seconds"} <= set(rows)


def test_session_pause_resume():
    import time as _time
    session = SessionTracker()
    session.started_at -= 10                     # 10 s of monitoring so far
    session.durations["Slouching"] = 3.0
    step(session, 0.5, "monitoring", ["Good Posture"], "GOOD", 85)
    session.pause()
    frozen = session.elapsed
    break_before = session.last_break_at
    _time.sleep(0.3)                             # cameras stopped for a moment
    assert session.elapsed == frozen, "session clock frozen while stopped"
    session.resume()
    assert abs(session.elapsed - frozen) < 0.05, "stopped time is not counted"
    assert session.durations["Slouching"] == 3.0 and session.average_score == 85, "statistics are kept"
    assert session.last_break_at - break_before >= 0.3, "sitting-too-long timer paused as well"
    assert not session.paused and session.current_bad_since is None
    session.resume()                             # resuming twice is harmless
    assert abs(session.elapsed - frozen) < 0.05


def main():
    tests = [test_geometry, test_one_euro, test_stabilizer, test_mirror_and_drawing,
             test_front_features, test_side_features, test_side_hip_threshold, test_classification, test_session_statistics,
             test_session_csv, test_session_pause_resume, test_popup_notifier,
             test_history_store, test_active_break_games]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print("All self-tests passed.")


if __name__ == "__main__":
    main()
