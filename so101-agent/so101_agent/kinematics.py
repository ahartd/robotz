"""Forward/inverse kinematics for the SO-101, using placo and the official URDF.

Angles are in degrees, matching LeRobot's `use_degrees=True` convention (0 = the middle
pose you used during `lerobot-calibrate`). Positions are meters in the arm's base frame:
+x straight out in front of the arm, +y to the arm's left, +z up.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .config import PKG_DIR

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF_PATH = PKG_DIR / "assets" / "so101_kinematics.urdf"
TIP_FRAME = "gripper_frame_link"  # point between the jaw tips


class IKError(RuntimeError):
    pass


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def tilt_deg(T: np.ndarray) -> float:
    """Angle between the gripper's pointing axis and straight down."""
    return math.degrees(math.acos(float(np.clip(-T[2, 2], -1.0, 1.0))))


class Kinematics:
    def __init__(self, urdf: str | Path = URDF_PATH, frame: str = TIP_FRAME):
        import placo

        self.robot = placo.RobotWrapper(str(urdf))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.frame = frame
        self.task = self.solver.add_frame_task(frame, np.eye(4))
        self.limits = self._read_limits(urdf)
        self.pan_xy = self._read_pan_axis(urdf)  # the arm turns around this point
        # Gripper pointing straight down with the jaws lined up with the arm.
        self.R_down = self.fk(np.array([0.0, 0.0, 0.0, 90.0, 0.0]))[:3, :3].copy()

    @staticmethod
    def _read_limits(urdf: str | Path) -> dict[str, tuple[float, float]]:
        limits = {}
        for joint in ET.parse(str(urdf)).getroot().iter("joint"):
            name = joint.get("name")
            lim = joint.find("limit")
            if name in JOINTS and lim is not None:
                limits[name] = (math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper"))))
        return limits

    @staticmethod
    def _read_pan_axis(urdf: str | Path) -> tuple[float, float]:
        for joint in ET.parse(str(urdf)).getroot().iter("joint"):
            if joint.get("name") == "shoulder_pan":
                x, y, _ = (float(v) for v in joint.find("origin").get("xyz").split())
                return (x, y)
        return (0.0, 0.0)

    def reach_polar(self, x: float, y: float) -> tuple[float, float]:
        """(distance from the pan axis in m, angle in degrees; + is to the arm's left)."""
        dx, dy = x - self.pan_xy[0], y - self.pan_xy[1]
        return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))

    def _set(self, q_deg: np.ndarray) -> None:
        for name, v in zip(JOINTS, q_deg):
            self.robot.set_joint(name, math.radians(float(v)))
        self.robot.update_kinematics()

    def _get(self) -> np.ndarray:
        return np.array([math.degrees(self.robot.get_joint(n)) for n in JOINTS])

    def fk(self, q_deg) -> np.ndarray:
        self._set(np.asarray(q_deg, dtype=float)[:5])
        return np.array(self.robot.get_T_world_frame(self.frame))

    def within_limits(self, q_deg, margin_deg: float = 0.0) -> bool:
        for name, v in zip(JOINTS, q_deg):
            lo, hi = self.limits[name]
            if not (lo + margin_deg <= v <= hi - margin_deg):
                return False
        return True

    def down_pose(self, x: float, y: float, z: float, yaw_offset_deg: float = 0.0, lean_deg: float = 0.0) -> np.ndarray:
        """Gripper pointing down at (x, y, z), optionally leaning `lean_deg` within the arm's plane."""
        az = math.atan2(y - self.pan_xy[1], x - self.pan_xy[0])
        R = _rot_z(az + math.radians(yaw_offset_deg)) @ self.R_down
        if lean_deg:
            axis = np.array([-math.sin(az), math.cos(az), 0.0])  # horizontal, across the reach direction
            R = _axis_angle(axis, math.radians(lean_deg)) @ R
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def _solve(self, seed: np.ndarray, T: np.ndarray, orientation_weight: float, rounds: int = 25):
        q = np.array(seed, dtype=float)
        for _ in range(rounds):
            self._set(q)
            self.task.T_world_frame = T
            self.task.configure(self.frame, "soft", 1.0, orientation_weight)
            for _ in range(8):
                self.solver.solve(True)
                self.robot.update_kinematics()
            q = self._get()
            if np.linalg.norm(self.fk(q)[:3, 3] - T[:3, 3]) < 5e-4:
                break
        F = self.fk(q)
        return q, float(np.linalg.norm(F[:3, 3] - T[:3, 3])), tilt_deg(F)

    def solve_down(
        self,
        x: float,
        y: float,
        z: float,
        seed,
        yaw_offset_deg: float = 0.0,
        max_tilt_deg: float = 0.0,
        margin_deg: float = 3.0,
    ) -> tuple[np.ndarray, float]:
        """Joint angles that put the gripper tip at (x, y, z) pointing down.

        Tries a vertical gripper first; if that's out of reach, lets it lean one way or the
        other in 5° steps up to `max_tilt_deg`. Returns (joints, lean). Raises IKError.
        """
        seed = np.asarray(seed, dtype=float)[:5]
        dx, dy = x - self.pan_xy[0], y - self.pan_xy[1]
        pan = -math.degrees(math.atan2(dy, dx))
        roll = seed[4]
        seeds = [seed, np.array([pan, *seed[1:4], roll]), np.array([pan, -60, 80, 70, roll]),
                 np.array([pan, 0, 0, 80, roll]), np.array([pan, 40, -30, 80, roll]), np.array([pan, 60, -60, 85, roll])]
        leans = [0.0]
        step = 5.0
        while step <= max_tilt_deg + 1e-9:
            leans += [step, -step]
            step += 5.0
        for lean in leans:
            T = self.down_pose(x, y, z, yaw_offset_deg, lean)
            good = []
            for s in seeds:
                q, pos_err, _ = self._solve(s, T, 0.5)
                F = self.fk(q)
                ang_err = math.degrees(math.acos(float(np.clip(np.dot(F[:3, 2], T[:3, 2]), -1, 1))))
                if pos_err < 2e-3 and ang_err < 2.0 and self.within_limits(q, margin_deg):
                    good.append((float(np.max(np.abs(q - seed))), q))
                    if good[-1][0] < 15:  # close to where we are: good enough, stop searching
                        break
            if good:
                return min(good, key=lambda g: g[0])[1], abs(lean)
        raise IKError(
            f"the arm can't reach x={x * 100:.1f} cm, y={y * 100:.1f} cm, z={z * 100:.1f} cm "
            "with the gripper pointing down"
        )
