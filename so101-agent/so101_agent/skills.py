"""Everything the arm can do, as plain Python methods.

Both front ends use this: the MCP server (Claude app) and the terminal agent (and later,
voice). Each method returns a ToolResult with text and optional JPEG images, and never
raises for ordinary problems: the error goes back to the model as text so it can adapt.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from dataclasses import dataclass, field

from .arm import Arm, ArmError
from .calibration import CalibrationError, camera_moved, load_calibration
from .config import Config
from .kinematics import IKError
from .pointing import PointingError, gemini_point
from .vision import Camera, CameraError, MockCamera, TableMap, annotate

log = logging.getLogger(__name__)


class SkillError(RuntimeError):
    pass


@dataclass
class ToolResult:
    text: str
    images: list[bytes] = field(default_factory=list)
    is_error: bool = False


def tool(fn):
    """Serialize hardware access and turn expected failures into readable errors."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            try:
                return fn(self, *args, **kwargs)
            except (ArmError, IKError, CameraError, SkillError, PointingError, CalibrationError) as e:
                return ToolResult(f"Couldn't do that: {e}", is_error=True)
            except Exception as e:  # unexpected: still report instead of crashing the server
                log.exception("tool %s failed", fn.__name__)
                return ToolResult(f"Unexpected error in {fn.__name__}: {type(e).__name__}: {e}", is_error=True)

    return wrapper


class Skills:
    def __init__(self, cfg: Config, arm: Arm | None = None, camera=None, wrist_camera=None):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._arm = arm
        self._camera = camera
        self._wrist = wrist_camera
        self.calib = load_calibration(cfg.calibration_path)
        self.tmap = TableMap(self.calib["homography"]) if self.calib else None
        if self._arm is not None and self.calib:
            self._arm.apply_calibration(self.calib)

    # ------------------------------------------------------------ hardware
    @property
    def arm(self) -> Arm:
        if self._arm is None:
            self._arm = Arm(self.cfg)
            if self.calib:
                self._arm.apply_calibration(self.calib)
        return self._arm

    @property
    def camera(self):
        if self._camera is None:
            c = self.cfg.camera
            self._camera = MockCamera(width=c.width, height=c.height) if c.mock else Camera(c.index, c.width, c.height, c.rotate_180)
        return self._camera

    @property
    def wrist(self):
        if self._wrist is None:
            c = self.cfg.camera
            if c.wrist_index is None:
                raise SkillError("no wrist camera is set up (camera.wrist_index in config.yaml)")
            self._wrist = Camera(c.wrist_index, c.width, c.height)
        return self._wrist

    def request_stop(self) -> None:
        """Freeze the arm immediately, even mid-move. Deliberately doesn't take the lock:
        a move in progress holds it."""
        if self._arm is not None:
            self._arm.request_stop()

    def clear_stop(self) -> None:
        if self._arm is not None:
            self._arm.clear_stop()

    def shutdown(self) -> None:
        with self._lock:
            if self._arm is not None:
                try:
                    self._arm.close()
                except Exception:
                    log.exception("error while parking the arm")
            for cam in (self._camera, self._wrist):
                if cam is not None:
                    cam.close()

    # ------------------------------------------------------------ helpers
    def _need_map(self) -> TableMap:
        if self.tmap is None:
            raise SkillError("the camera-to-arm calibration hasn't been done yet: run `python calibrate.py`")
        return self.tmap

    def _xy(self, u: float, v: float) -> tuple[float, float]:
        tm = self._need_map()
        w, h = self.cfg.camera.width, self.cfg.camera.height
        if not (0 <= u < w and 0 <= v < h):
            raise SkillError(f"pixel ({u}, {v}) is outside the {w}x{h} image")
        return tm.to_table(u, v)

    def _reach(self):
        s = self.cfg.safety
        return (s.min_radius_m, s.max_radius_m, s.max_abs_angle_deg, self.arm.kin.pan_xy)

    def _in_reach(self, x: float, y: float) -> bool:
        s = self.cfg.safety
        r, ang = self.arm.kin.reach_polar(x, y)
        return s.min_radius_m <= r <= s.max_radius_m and abs(ang) <= s.max_abs_angle_deg

    def _state(self) -> str:
        arm = self.arm
        t = arm.tip()
        parts = [f"Gripper tip at x={t['x'] * 100:.1f} cm, y={t['y'] * 100:.1f} cm"]
        if "height" in t:
            parts.append(f"{t['height'] * 100:.1f} cm above the table")
        if self.tmap is not None:
            u, v = self.tmap.to_pixel(t["x"], t["y"])
            parts.append(f"over pixel ({u:.0f}, {v:.0f})")
        parts.append(f"jaws {arm.gripper_fraction() * 100:.0f}% open")
        if arm.holding():
            parts.append("holding something")
        if not arm.torque_on:
            parts.append("motors OFF (limp)")
        return ", ".join(parts) + "."

    def _pick_height(self, height_cm: float | None, default: float) -> float:
        return default if height_cm is None else float(height_cm) / 100.0

    # ------------------------------------------------------------ tools
    @tool
    def look(self, clear_view: bool = False, zoom_u: float | None = None, zoom_v: float | None = None,
             zoom: float = 1.0, camera: str = "overhead") -> ToolResult:
        if camera == "wrist":
            img = self.wrist.capture()
            jpg, _ = annotate(img)
            return ToolResult(f"Wrist camera view. {self._state()}", [jpg])
        if clear_view:
            self.arm.home()
        img = self.camera.capture()
        tip = None
        if self.tmap is not None:
            t = self.arm.tip()
            tip = (t["x"], t["y"])
        center = (zoom_u, zoom_v) if zoom_u is not None and zoom_v is not None else None
        jpg, (x0, y0, x1, y1) = annotate(img, self.tmap, self._reach(), center, zoom, tip_xy=tip)
        notes = [f"Overhead camera, full image is {img.shape[1]}x{img.shape[0]} px; grid labels are full-image pixel coordinates (u right, v down)."]
        if zoom > 1:
            notes.append(f"Showing the region u {x0:.0f}-{x1:.0f}, v {y0:.0f}-{y1:.0f} at {zoom:g}x.")
        if self.tmap is None:
            notes.append("Not calibrated yet, so the arm can't act on what it sees: run `python calibrate.py`.")
        else:
            notes.append("Yellow outline = where the gripper can reach.")
            moved, msg = camera_moved(self.calib, img)
            if moved:
                notes.append("WARNING: " + msg)
        notes.append(self._state())
        return ToolResult(" ".join(notes), [jpg])

    @tool
    def locate(self, description: str) -> ToolResult:
        img = self.camera.capture()
        pts = gemini_point(img, description, self.cfg.pointing.gemini_model, self.cfg.pointing.gemini_api_key_env)
        if not pts:
            return ToolResult(f"Gemini didn't find {description!r} in the image.")
        lines = []
        for u, v, label in pts:
            line = f"- {label}: pixel ({u:.0f}, {v:.0f})"
            if self.tmap is not None:
                x, y = self.tmap.to_table(u, v)
                line += f", table x={x * 100:.1f} cm y={y * 100:.1f} cm, {'reachable' if self._in_reach(x, y) else 'OUT OF REACH'}"
            lines.append(line)
        jpg, _ = annotate(img, self.tmap, self._reach(), points=pts)
        return ToolResult("Found:\n" + "\n".join(lines), [jpg])

    @tool
    def move_to(self, u: float, v: float, height_cm: float = 5.0) -> ToolResult:
        x, y = self._xy(u, v)
        info = self.arm.move_to(x, y, height_cm / 100.0)
        lean = f" (gripper leaning {info['lean_deg']:.0f}° to reach)" if info["lean_deg"] > 3 else ""
        return ToolResult(f"Moved over pixel ({u:.0f}, {v:.0f}){lean}. {self._state()}")

    @tool
    def nudge(self, du_px: float = 0.0, dv_px: float = 0.0, dz_cm: float = 0.0) -> ToolResult:
        tm = self._need_map()
        t = self.arm.tip()
        u, v = tm.to_pixel(t["x"], t["y"])
        x, y = tm.to_table(u + du_px, v + dv_px)
        self.arm.move_to(x, y, t["height"] + dz_cm / 100.0, speed_scale=0.5)
        return ToolResult(f"Nudged. {self._state()}")

    @tool
    def pick(self, u: float, v: float, grasp_angle_deg: float = 0.0, height_cm: float | None = None) -> ToolResult:
        arm, m = self.arm, self.cfg.motion
        if arm.holding():
            raise SkillError("already holding something; place() it or open the gripper first")
        x, y = self._xy(u, v)
        grasp = self._pick_height(height_cm, m.grasp_height_m)
        arm.move_to(x, y, m.hover_height_m, grasp_angle_deg)
        arm.open_gripper()
        arm.move_to(x, y, grasp, grasp_angle_deg, speed_scale=0.5)
        arm.close_gripper()
        arm.move_to(x, y, m.hover_height_m, grasp_angle_deg, speed_scale=0.6)
        if arm.holding():
            verdict = "The jaws stopped partway, so it's probably holding the object."
        else:
            verdict = "The jaws closed all the way: it probably MISSED. Look again and retry (adjust the position, angle or height)."
        return ToolResult(f"Tried to pick at pixel ({u:.0f}, {v:.0f}). {verdict} {self._state()}")

    @tool
    def place(self, u: float, v: float, height_cm: float | None = None) -> ToolResult:
        arm, m = self.arm, self.cfg.motion
        x, y = self._xy(u, v)
        release = self._pick_height(height_cm, m.place_height_m)
        travel = max(m.hover_height_m, release + 0.03)
        arm.move_to(x, y, travel)
        arm.move_to(x, y, release, speed_scale=0.5)
        arm.open_gripper()
        arm.move_to(x, y, travel, speed_scale=0.6)
        return ToolResult(f"Released at pixel ({u:.0f}, {v:.0f}), {release * 100:.1f} cm above the table. {self._state()}")

    @tool
    def gripper(self, open_percent: float) -> ToolResult:
        open_percent = max(0.0, min(100.0, float(open_percent)))
        if open_percent <= 1.0:
            self.arm.close_gripper()
        else:
            self.arm.set_gripper(open_percent / 100.0)
        return ToolResult(f"Gripper set. {self._state()}")

    @tool
    def home(self) -> ToolResult:
        self.arm.home()
        return ToolResult(f"Arm folded up out of the camera's way. {self._state()}")

    @tool
    def status(self) -> ToolResult:
        cal = "calibrated" if self.calib else "NOT calibrated (run calibrate.py)"
        return ToolResult(f"Camera-to-arm mapping: {cal}. {self._state()}")

    @tool
    def rest(self) -> ToolResult:
        self.arm.rest()
        return ToolResult("Arm is back in its rest pose with the motors off. Any move command wakes it up again.")
