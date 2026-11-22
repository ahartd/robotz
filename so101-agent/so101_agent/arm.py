"""Safe, speed-limited motion for the SO-101 follower arm.

Every move is streamed in small steps (never a jump), checked against joint limits and a
workspace box above the table, and Cartesian moves follow straight lines so the gripper
doesn't sweep through things on the table.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import numpy as np

from .config import Config, RobotCfg
from .kinematics import JOINTS, IKError, Kinematics, tilt_deg

log = logging.getLogger(__name__)

MOTORS = JOINTS + ["gripper"]
GRIPPER_SPEED = 120.0  # gripper units (0-100 scale) per second


class ArmError(RuntimeError):
    pass


class ArmStopped(ArmError):
    """Raised when a move is cut short by request_stop() (e.g. someone said "stop")."""


class LeRobotBackend:
    """Talks to the real arm through LeRobot."""

    def __init__(self, rc: RobotCfg):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # older LeRobot releases
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        if not rc.port:
            raise ArmError("Set robot.port in config.yaml (find it with `lerobot-find-port`).")
        # Keep holding position if the program exits, rather than dropping the arm.
        kwargs = {"port": rc.port, "id": rc.id, "use_degrees": True, "disable_torque_on_disconnect": False}
        if rc.calibration_dir:
            kwargs["calibration_dir"] = Path(rc.calibration_dir).expanduser()
        if rc.max_relative_target_deg:
            kwargs["max_relative_target"] = float(rc.max_relative_target_deg)
        self.robot = SO101Follower(SO101FollowerConfig(**kwargs))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise ArmError(
                f"No LeRobot calibration found for robot id '{rc.id}'. Run "
                f"`lerobot-calibrate --robot.type=so101_follower --robot.port={rc.port} --robot.id={rc.id}` "
                "or fix robot.id in config.yaml."
            )

    def read(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {m: float(obs[f"{m}.pos"]) for m in MOTORS}

    def write(self, targets: dict[str, float]) -> None:
        self.robot.send_action({f"{m}.pos": float(v) for m, v in targets.items()})

    def torque(self, on: bool) -> None:
        if on:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


class MockBackend:
    """Simulated arm for testing: commands take effect instantly."""

    def __init__(self, start=(0.0, -100.0, 96.0, 60.0, 0.0, 2.0)):
        self.state = dict(zip(MOTORS, map(float, start)))
        self.torque_on = True
        self.grip_stop: float | None = None  # set to simulate an object between the jaws
        self.history: list[dict[str, float]] = []

    def read(self) -> dict[str, float]:
        return dict(self.state)

    def write(self, targets: dict[str, float]) -> None:
        if not self.torque_on:
            return
        for m, v in targets.items():
            if m == "gripper" and self.grip_stop is not None:
                v = max(v, self.grip_stop)
            self.state[m] = float(v)
        self.history.append(dict(self.state))

    def torque(self, on: bool) -> None:
        self.torque_on = on

    def close(self) -> None:
        pass


class Arm:
    def __init__(self, cfg: Config, backend=None):
        self.cfg = cfg
        self.kin = Kinematics()
        self.backend = backend or (MockBackend() if cfg.robot.mock else LeRobotBackend(cfg.robot))
        self.torque_on = True
        self.rest_pose = self.backend.read()  # wherever the arm sat when we connected
        self.table_z: float | None = None
        self.gripper_closed = 0.0
        self.gripper_open = 100.0
        self.stop_event = threading.Event()  # set from any thread to freeze the arm mid-move

    # ---------------------------------------------------------------- state
    def apply_calibration(self, calib: dict) -> None:
        self.table_z = float(calib["table_z"])
        self.gripper_closed = float(calib["gripper_closed"])
        self.gripper_open = float(calib["gripper_open"])

    def joints(self) -> dict[str, float]:
        return self.backend.read()

    def q(self) -> np.ndarray:
        j = self.joints()
        return np.array([j[n] for n in JOINTS])

    def tip(self) -> dict[str, float]:
        """Gripper tip position (m, base frame); `height` is above the table if known."""
        T = self.kin.fk(self.q())
        x, y, z = (float(v) for v in T[:3, 3])
        out = {"x": x, "y": y, "z": z, "tilt_deg": tilt_deg(T)}
        if self.table_z is not None:
            out["height"] = z - self.table_z
        return out

    def gripper_fraction(self) -> float:
        span = self.gripper_open - self.gripper_closed
        return (self.joints()["gripper"] - self.gripper_closed) / span if span else 0.0

    def holding(self) -> bool:
        """True if the jaws stopped short of fully closed, i.e. something is between them."""
        span = abs(self.gripper_open - self.gripper_closed)
        frac = self.gripper_fraction()
        return self._gripper_cmd_closed and frac * span > self.cfg.motion.holding_margin

    _gripper_cmd_closed = False

    # ---------------------------------------------------------- stopping
    def request_stop(self) -> None:
        """Freeze the arm where it is. Safe to call from any thread; moves keep failing
        with ArmStopped until clear_stop() is called."""
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def _halt_if_stopped(self) -> None:
        if self.stop_event.is_set():
            try:
                self.backend.write(self.backend.read())  # hold the current position
            except Exception:
                log.exception("couldn't send a hold command")
            raise ArmStopped("stopped because the user said stop; the arm is holding still where it is")

    # ---------------------------------------------------------- low level
    def _ensure_torque(self) -> None:
        if not self.torque_on:
            # Hold wherever the arm is now, so enabling torque can't make it snap.
            self.backend.write(self.backend.read())
            self.backend.torque(True)
            self.torque_on = True

    def _stream(self, waypoints: list[np.ndarray], speed_scale: float = 1.0) -> None:
        """Send joint-space waypoints (6 values each, gripper last) at limited speed."""
        self._halt_if_stopped()
        self._ensure_torque()
        hz = self.cfg.motion.control_hz
        speed = self.cfg.motion.max_joint_speed_deg_s * max(0.05, min(speed_scale, 1.0))
        j = self.joints()
        current = np.array([j[m] for m in MOTORS])
        for wp in waypoints:
            wp = np.asarray(wp, dtype=float)
            delta = wp - current
            t = max(np.max(np.abs(delta[:5])) / speed, abs(delta[5]) / GRIPPER_SPEED)
            steps = max(1, math.ceil(t * hz))
            for i in range(1, steps + 1):
                self._halt_if_stopped()
                p = current + delta * (i / steps)
                self.backend.write(dict(zip(MOTORS, p)))
                time.sleep(1.0 / hz)
            current = wp
        time.sleep(0.15)  # let the servos settle

    def _check_joints(self, q5) -> None:
        if not self.kin.within_limits(q5, self.cfg.safety.joint_limit_margin_deg):
            raise ArmError(f"joint target {np.round(q5, 1).tolist()} is outside the safe joint limits")

    def _check_target(self, x: float, y: float, height: float) -> None:
        s = self.cfg.safety
        r, ang = self.kin.reach_polar(x, y)
        if r < s.min_radius_m:
            raise ArmError(f"target is {r * 100:.1f} cm from the arm's shoulder; the closest allowed is {s.min_radius_m * 100:.0f} cm")
        if r > s.max_radius_m:
            raise ArmError(f"target is {r * 100:.1f} cm from the arm's shoulder; the farthest allowed is {s.max_radius_m * 100:.0f} cm")
        if abs(ang) > s.max_abs_angle_deg:
            raise ArmError(f"target is {abs(ang):.0f}° to the side; the limit is {s.max_abs_angle_deg:.0f}°")
        if not (s.min_height_m <= height <= s.max_height_m):
            raise ArmError(
                f"height {height * 100:.1f} cm is outside the allowed {s.min_height_m * 100:.1f}-{s.max_height_m * 100:.0f} cm above the table"
            )

    def _require_table(self) -> float:
        if self.table_z is None:
            raise ArmError("the camera-to-arm calibration hasn't been done yet: run `python calibrate.py`")
        return self.table_z

    def _path_is_clear(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        """Check a straight joint-space path keeps the gripper tip above the table
        (or at least no lower than where it starts or ends, e.g. leaving the rest pose)."""
        z0, z1 = self.kin.fk(q_from)[2, 3], self.kin.fk(q_to)[2, 3]
        floor = min(self._require_table() + 0.005, z0 - 0.001, z1 - 0.001)
        for a in np.linspace(0.0, 1.0, 25):
            T = self.kin.fk(q_from + (q_to - q_from) * a)
            if T[2, 3] < floor:
                return False
        return True

    # ---------------------------------------------------------- motions
    def move_to(self, x: float, y: float, height: float, yaw_offset_deg: float = 0.0, speed_scale: float = 1.0,
                _detour: bool = True) -> dict:
        """Move the gripper tip to (x, y) at `height` above the table, pointing down."""
        table = self._require_table()
        self._check_target(x, y, height)
        z = table + height
        max_tilt = self.cfg.motion.max_tilt_deg
        q_now = self.q()
        g = self.joints()["gripper"]
        start = self.tip()
        start_r = self.kin.reach_polar(start["x"], start["y"])[0]
        try:
            q_goal, tilt = self.kin.solve_down(x, y, z, q_now, yaw_offset_deg, max_tilt, self.cfg.safety.joint_limit_margin_deg)
        except IKError as e:
            raise ArmError(str(e)) from None

        waypoints: list[np.ndarray] | None = None
        # Straight-line Cartesian path when we're already in a normal pointing-down pose.
        if start["tilt_deg"] <= max_tilt + 2 and start_r >= self.cfg.safety.min_radius_m - 0.02 and start.get("height", 0) >= 0:
            p0 = np.array([start["x"], start["y"], start["z"]])
            p1 = np.array([x, y, z])
            n = max(1, math.ceil(np.linalg.norm(p1 - p0) / 0.01))
            q = q_now
            waypoints = []
            try:
                for i in range(1, n + 1):
                    p = p0 + (p1 - p0) * (i / n)
                    q_prev = q
                    q, _ = self.kin.solve_down(*p, q, yaw_offset_deg, max_tilt, self.cfg.safety.joint_limit_margin_deg)
                    # 1 cm of travel should never need a big joint swing (that would mean the
                    # solver flipped to another arm configuration mid-path)
                    if i > 1 and np.max(np.abs(q - q_prev)) > 25:
                        raise IKError("path needs a large joint swing")
                    waypoints.append(np.append(q, g))
            except IKError:
                waypoints = None
            if waypoints and np.max(np.abs(waypoints[-1][:5] - q_goal)) > 5:
                q_goal = waypoints[-1][:5]
        if waypoints is None:
            # Joint-space move; if that would drag the gripper along the table, go via home.
            if not self._path_is_clear(q_now, q_goal):
                if not _detour:
                    raise ArmError("couldn't find a path to that spot that stays above the table")
                self.home()
                q_now = self.q()
                if not self._path_is_clear(q_now, q_goal):
                    raise ArmError("couldn't find a path to that spot that stays above the table")
            waypoints = [np.append(q_goal, g)]
        for wp in waypoints:
            self._check_joints(wp[:5])
        self._stream(waypoints, speed_scale)
        return {"lean_deg": tilt}

    def move_joints(self, q5, gripper: float | None = None, speed_scale: float = 1.0, check: bool = True) -> None:
        q5 = np.asarray(q5, dtype=float)
        if check:
            self._check_joints(q5)
        g = self.joints()["gripper"] if gripper is None else gripper
        self._stream([np.append(q5, g)], speed_scale)

    def set_gripper(self, fraction_open: float) -> None:
        fraction_open = float(np.clip(fraction_open, 0.0, 1.0))
        target = self.gripper_closed + fraction_open * (self.gripper_open - self.gripper_closed)
        self._gripper_cmd_closed = False
        self._stream([np.append(self.q(), target)])

    def open_gripper(self) -> None:
        self.set_gripper(self.cfg.motion.open_fraction)

    def close_gripper(self) -> bool:
        """Close the jaws; returns True if something seems to be held."""
        self._stream([np.append(self.q(), self.gripper_closed)])
        time.sleep(0.4)
        self._gripper_cmd_closed = True
        return self.holding()

    def home(self) -> None:
        """Fold up near the base, out of the overhead camera's way."""
        home = np.array(self.cfg.motion.home_joints_deg, dtype=float)
        if self.table_z is not None:
            t = self.tip()
            lift = self.cfg.motion.hover_height_m
            if t["height"] < lift - 0.01 and t["tilt_deg"] < 30 and self.kin.reach_polar(t["x"], t["y"])[0] >= self.cfg.safety.min_radius_m:
                try:  # come straight up first so we don't drag across the table
                    self.move_to(t["x"], t["y"], lift, _detour=False)
                except ArmError:
                    pass
        self.move_joints(home)

    def rest(self) -> None:
        """Go back to the pose the arm was in at startup, then let the motors go limp."""
        self.home()
        r = self.rest_pose
        # a pose the arm was really in, often right against the joint stops: skip the margin check
        self.move_joints([r[n] for n in JOINTS], gripper=r["gripper"], speed_scale=0.5, check=False)
        self.relax()

    def relax(self) -> None:
        """Turn the motors off. The arm will drop if it isn't resting on something."""
        self.backend.torque(False)
        self.torque_on = False

    def close(self) -> None:
        """Park in the rest pose with the motors off, then disconnect. If this gets cut
        short, the motors stay on and hold the arm where it is instead of dropping it."""
        try:
            self.clear_stop()
            if self.torque_on:
                self.rest()
        except Exception:
            log.exception("couldn't park the arm before disconnecting")
        finally:
            self.backend.close()
