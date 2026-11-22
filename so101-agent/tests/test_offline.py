"""Offline tests with a simulated arm and a synthetic overhead camera. No hardware needed.

    python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import so101_agent.arm as arm_mod  # noqa: E402
from so101_agent.arm import Arm, ArmError, MockBackend  # noqa: E402
from so101_agent.calibration import camera_moved, save_calibration, touch_calibration  # noqa: E402
from so101_agent.config import Config  # noqa: E402
from so101_agent.pointing import parse_points  # noqa: E402
from so101_agent.skills import Skills  # noqa: E402
from so101_agent.vision import MockCamera, TableMap, annotate, marker_image  # noqa: E402

arm_mod.time.sleep = lambda s: None  # run motions instantly

TABLE_Z = -0.0024  # bottom of the SO-101 base in the URDF frame
PPM = 1500  # synthetic top-down canvas resolution, px per meter
X_RANGE, Y_RANGE = (0.0, 0.45), (-0.30, 0.30)
MARKERS = {0: (0.13, 0.15), 1: (0.13, -0.15), 2: (0.27, 0.10), 3: (0.27, -0.10), 4: (0.20, 0.0), 5: (0.29, 0.0)}
BLOCKS = {"red": ((0.22, 0.08), (0, 0, 220)), "blue": ((0.21, -0.12), (220, 60, 0)), "far": ((0.40, 0.24), (0, 200, 0))}


def canvas_px(x, y):
    return ((Y_RANGE[1] - y) * PPM, (X_RANGE[1] - x) * PPM)  # +y is left, far is up


def camera_homography():
    """Canvas -> camera image, with some perspective as if the camera were slightly tilted."""
    corners_table = [(0.40, 0.27), (0.40, -0.27), (0.06, -0.27), (0.06, 0.27)]
    src = np.float32([canvas_px(*c) for c in corners_table])
    dst = np.float32([[40, 25], [600, 15], [630, 470], [15, 460]])
    return cv2.getPerspectiveTransform(src, dst)


H_CAM = camera_homography()


def true_pixel(x, y):
    p = H_CAM @ np.array([*canvas_px(x, y), 1.0])
    return p[0] / p[2], p[1] / p[2]


def render(blocks=BLOCKS, markers=MARKERS, shift_px=(0, 0)):
    w, h = int((Y_RANGE[1] - Y_RANGE[0]) * PPM), int((X_RANGE[1] - X_RANGE[0]) * PPM)
    canvas = np.full((h, w, 3), 235, np.uint8)
    size = int(0.04 * PPM)
    for mid, (x, y) in markers.items():
        cx, cy = canvas_px(x, y)
        m = cv2.cvtColor(marker_image(mid, size), cv2.COLOR_GRAY2BGR)
        x0, y0 = int(cx - size / 2), int(cy - size / 2)
        canvas[y0 - 10:y0 + size + 10, x0 - 10:x0 + size + 10] = 255
        canvas[y0:y0 + size, x0:x0 + size] = m
    for (x, y), color in blocks.values():
        cx, cy = canvas_px(x, y)
        r = int(0.0125 * PPM)
        cv2.rectangle(canvas, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), color, -1)
    Hs = H_CAM.copy()
    Hs[0, :] += shift_px[0] * Hs[2, :]
    Hs[1, :] += shift_px[1] * Hs[2, :]
    return cv2.warpPerspective(canvas, Hs, (640, 480), borderValue=(90, 90, 90))


def make_cfg(tmp_path) -> Config:
    cfg = Config(base_dir=tmp_path)
    cfg.robot.mock = True
    cfg.camera.mock = True
    return cfg


def simulated_touch(arm: Arm, backend: MockBackend):
    def ask(msg: str):
        if "Close the gripper" in msg:
            backend.state["gripper"] = 1.5
        elif "open the gripper" in msg:
            backend.state["gripper"] = 96.0
        m = re.search(r"marker (\d+)", msg)
        if m and "Touch" in msg:
            x, y = MARKERS[int(m.group(1))]
            q, _ = arm.kin.solve_down(x, y, TABLE_Z, arm.q())
            backend.state.update(dict(zip(arm_mod.JOINTS, q)))
    return ask


@pytest.fixture
def calibrated(tmp_path):
    cfg = make_cfg(tmp_path)
    backend = MockBackend()
    arm = Arm(cfg, backend=backend)
    cam = MockCamera(render=render)
    data = touch_calibration(arm, cam, simulated_touch(arm, backend), say=lambda s: None)
    save_calibration(cfg.calibration_path, data)
    arm.apply_calibration(data)
    backend.torque(True)
    backend.state.update(zip(arm_mod.MOTORS, [0.0, -100.0, 96.0, 60.0, 0.0, 2.0]))
    return cfg, arm, backend, cam, data


def tip_heights(arm: Arm, history) -> np.ndarray:
    return np.array([arm.kin.fk([s[j] for j in arm_mod.JOINTS])[2, 3] for s in history])


# --------------------------------------------------------------------------- kinematics
def test_ik_round_trip_across_workspace(tmp_path):
    arm = Arm(make_cfg(tmp_path), backend=MockBackend())
    for r in np.arange(0.09, 0.28, 0.03):
        for ang in (-70, -35, 0, 35, 70):
            for h in (0.012, 0.06):
                px, py = arm.kin.pan_xy
                x, y = px + r * math.cos(math.radians(ang)), py + r * math.sin(math.radians(ang))
                q, tilt = arm.kin.solve_down(x, y, TABLE_Z + h, arm.q(), max_tilt_deg=25)
                p = arm.kin.fk(q)[:3, 3]
                assert np.linalg.norm(p - [x, y, TABLE_Z + h]) < 2e-3
                assert tilt <= 25


def test_grasp_angle_rotates_jaws(tmp_path):
    arm = Arm(make_cfg(tmp_path), backend=MockBackend())
    q0, _ = arm.kin.solve_down(0.2, 0.05, 0.01, arm.q())
    q90, _ = arm.kin.solve_down(0.2, 0.05, 0.01, arm.q(), yaw_offset_deg=90)
    assert abs(abs(q90[4] - q0[4]) - 90) < 3  # wrist roll does the rotating
    assert np.allclose(q90[:4], q0[:4], atol=6)  # small shifts: the tip sits slightly off the roll axis


# --------------------------------------------------------------------------- calibration
def test_calibration_maps_pixels_to_table(calibrated):
    cfg, arm, backend, cam, data = calibrated
    assert abs(data["table_z"] - TABLE_Z) < 1e-3
    assert len(data["markers"]) == 6
    assert max(data["leave_one_out_mm"]) < 3.0
    tm = TableMap(data["homography"])
    for (x, y), _ in BLOCKS.values():
        u, v = true_pixel(x, y)
        ex, ey = tm.to_table(u, v)
        assert math.dist((ex, ey), (x, y)) < 0.002, (x, y, ex, ey)


def test_camera_bump_is_detected(calibrated):
    *_, data = calibrated
    assert camera_moved(data, render())[0] is False
    assert camera_moved(data, render(shift_px=(15, 0)))[0] is True


# --------------------------------------------------------------------------- skills
def test_pick_and_place_follow_safe_paths(calibrated):
    cfg, arm, backend, cam, _ = calibrated
    s = Skills(cfg, arm=arm, camera=cam)
    u, v = true_pixel(*BLOCKS["red"][0])
    backend.state.update(dict(zip(arm_mod.JOINTS, cfg.motion.home_joints_deg)))
    backend.history.clear()
    r = s.pick(u, v)
    assert not r.is_error, r.text
    assert "MISSED" in r.text  # nothing between the jaws in the mock
    zs = tip_heights(arm, backend.history)
    assert zs.min() > TABLE_Z + 0.005, zs.min()  # never touched the table
    assert abs(zs.min() - (TABLE_Z + cfg.motion.grasp_height_m)) < 0.002  # went down to grasp height
    # the lowest point was right above the block
    low = backend.history[int(np.argmin(zs))]
    p = arm.kin.fk([low[j] for j in arm_mod.JOINTS])[:3, 3]
    assert math.dist(p[:2], BLOCKS["red"][0]) < 0.003

    backend.grip_stop = 30.0  # now pretend something is between the jaws
    r = s.pick(u, v)
    assert "probably holding" in r.text, r.text
    r = s.pick(u, v)
    assert r.is_error and "already holding" in r.text
    tu, tv = true_pixel(0.25, -0.05)
    backend.history.clear()
    r = s.place(tu, tv)
    assert not r.is_error, r.text
    assert tip_heights(arm, backend.history).min() > TABLE_Z + 0.015


def test_out_of_reach_is_refused(calibrated):
    cfg, arm, backend, cam, _ = calibrated
    s = Skills(cfg, arm=arm, camera=cam)
    u, v = true_pixel(*BLOCKS["far"][0])
    before = dict(backend.state)
    r = s.pick(u, v)
    assert r.is_error and "farthest allowed" in r.text, r.text
    assert backend.state == before  # did not move


def test_moves_from_home_and_rest_stay_above_table(calibrated):
    cfg, arm, backend, cam, _ = calibrated
    for start in ([0.0, -100.0, 96.0, 60.0, 0.0], cfg.motion.home_joints_deg):
        for r in (0.09, 0.18, 0.27):
            for ang in (-75, 0, 75):
                backend.state.update(dict(zip(arm_mod.JOINTS, start)))
                backend.history.clear()
                px, py = arm.kin.pan_xy
                x, y = px + r * math.cos(math.radians(ang)), py + r * math.sin(math.radians(ang))
                arm.move_to(x, y, cfg.motion.hover_height_m)
                assert tip_heights(arm, backend.history).min() > TABLE_Z + 0.004


def test_look_returns_gridded_image(calibrated, tmp_path):
    cfg, arm, backend, cam, _ = calibrated
    s = Skills(cfg, arm=arm, camera=cam)
    r = s.look(clear_view=True)
    assert not r.is_error and r.images and "camera position OK" not in r.text
    img = cv2.imdecode(np.frombuffer(r.images[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (480, 640, 3)
    (tmp_path / "look.jpg").write_bytes(r.images[0])
    rz = s.look(zoom_u=320, zoom_v=240, zoom=3)
    assert "region u" in rz.text


def test_uncalibrated_actions_explain_what_to_do(tmp_path):
    cfg = make_cfg(tmp_path)
    s = Skills(cfg, arm=Arm(cfg, backend=MockBackend()), camera=MockCamera(render=render))
    r = s.pick(300, 200)
    assert r.is_error and "calibrate.py" in r.text
    assert not s.look().is_error


def test_gemini_answer_parsing():
    pts = parse_points('```json\n[{"point": [500, 250], "label": "red block"}]\n```', 640, 480)
    assert pts == [(160.0, 240.0, "red block")]


# --------------------------------------------------------------------------- front ends
def test_mcp_server_and_agent_loop(calibrated, monkeypatch):
    cfg, arm, backend, cam, _ = calibrated
    monkeypatch.setenv("SO101_AGENT_CONFIG", str(cfg.base_dir / "missing.yaml"))
    sys.modules.pop("server", None)
    import server

    server.skills = Skills(cfg, arm=arm, camera=cam)
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"look", "pick", "place", "move_to", "nudge", "gripper", "home", "status", "rest", "locate"} <= names
    res = asyncio.run(server.mcp.call_tool("look", {"clear_view": True}))
    assert [c.type for c in res.content] == ["text", "image"]

    sys.modules.pop("agent", None)
    import agent

    u, v = true_pixel(*BLOCKS["blue"][0])
    script = [
        [SimpleNamespace(type="tool_use", id="t1", name="look", input={"clear_view": True})],
        [SimpleNamespace(type="tool_use", id="t2", name="pick", input={"u": u, "v": v})],
        [SimpleNamespace(type="text", text="I tried to pick up the blue block.")],
    ]

    class FakeMessages:
        def __init__(self):
            self.calls = []

        async def create(self, **kw):
            self.calls.append(kw)
            content = script[len(self.calls) - 1]
            stop = "tool_use" if content[0].type == "tool_use" else "end_turn"
            return SimpleNamespace(content=content, stop_reason=stop)

    fake = SimpleNamespace(messages=FakeMessages())
    a = agent.ArmAgent(client=fake, on_event=lambda m: None, keep_images=1)
    reply = asyncio.run(a.ask("pick up the blue block"))
    assert reply == "I tried to pick up the blue block."
    first_result = fake.messages.calls[1]["messages"][2]["content"][0]
    assert first_result["type"] == "tool_result" and first_result["content"][1]["type"] == "image"
    tool_names = {t["name"] for t in fake.messages.calls[0]["tools"]}
    assert "pick" in tool_names


def test_going_home_from_anywhere_stays_above_table(calibrated):
    cfg, arm, backend, cam, _ = calibrated
    px, py = arm.kin.pan_xy
    for r in (0.09, 0.18, 0.27):
        for ang in (-80, 0, 80):
            x, y = px + r * math.cos(math.radians(ang)), py + r * math.sin(math.radians(ang))
            arm.move_to(x, y, cfg.motion.grasp_height_m)
            backend.history.clear()
            arm.home()
            assert tip_heights(arm, backend.history).min() > TABLE_Z + 0.008


def test_env_file_loading(tmp_path, monkeypatch):
    from so101_agent.config import load_config

    for k in ("SO101_A", "SO101_B", "SO101_C", "SO101_D", "SO101_E", "SO101_SHELL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SO101_SHELL", "from-shell")
    (tmp_path / ".env").write_text(
        "# comment\n"
        "SO101_A=plain\n"
        'SO101_B="quoted # not a comment"\n'
        "export SO101_C=exported  # trailing comment\n"
        "SO101_D=\n"
        "SO101_SHELL=from-file\n"
        "not a setting\n"
    )
    (tmp_path / "config.yaml").write_text("robot:\n  mock: true\n")
    cfg = load_config(tmp_path / "config.yaml")
    import os

    assert cfg.robot.mock is True
    assert os.environ["SO101_A"] == "plain"
    assert os.environ["SO101_B"] == "quoted # not a comment"
    assert os.environ["SO101_C"] == "exported"
    assert "SO101_D" not in os.environ  # blank values are skipped
    assert os.environ["SO101_SHELL"] == "from-shell"  # the shell wins
    for k in ("SO101_A", "SO101_B", "SO101_C"):
        monkeypatch.delenv(k)
