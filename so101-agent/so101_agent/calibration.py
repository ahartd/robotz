"""Camera-to-arm "touch" calibration.

Printed ArUco markers are taped to the table. The camera finds them; then you guide the
limp arm by hand so the gripper tip touches each marker's center. Forward kinematics gives
where the tip was in the arm's own frame, so we get matched (pixel, table position) pairs
and fit a homography. Because the same arm model is used for calibration and for moving,
most model and assembly errors cancel out.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .vision import TableMap, detect_markers


class CalibrationError(RuntimeError):
    pass


def load_calibration(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def touch_calibration(arm, camera, ask: Callable[[str], None], say: Callable[[str], None] = print) -> dict:
    say("The motors are turning off now. Hold the arm so it doesn't drop.")
    ask("Support the arm with your hand, then press Enter.")
    arm.relax()

    ask("Close the gripper ALL the way by hand, then press Enter.")
    closed = arm.joints()["gripper"]
    ask("Now open the gripper ALL the way by hand, then press Enter.")
    opened = arm.joints()["gripper"]
    if abs(opened - closed) < 10:
        raise CalibrationError(f"gripper only moved from {closed:.1f} to {opened:.1f}; open and close it fully and try again")

    ask("Fold the arm back so it isn't covering any markers, then press Enter.")
    img = camera.capture()
    markers = detect_markers(img)
    if len(markers) < 4:
        raise CalibrationError(
            f"the camera can see {len(markers)} marker(s) {sorted(markers)}; at least 4 are needed. "
            "Check lighting and that the markers are flat and inside the camera view."
        )
    say(f"I can see markers {sorted(markers)}.")

    points = {}
    for mid in sorted(markers):
        ask(
            f"Touch the CENTER of marker {mid} with the gripper tip: jaws closed, gripper pointing "
            "straight down, tip just resting on the paper. Then press Enter."
        )
        T = arm.kin.fk(arm.q())
        tilt = math.degrees(math.acos(float(np.clip(-T[2, 2], -1, 1))))
        if tilt > 20:
            say(f"  (the gripper was leaning {tilt:.0f}° from vertical; try to keep it straighter next time)")
        points[mid] = {"pixel": list(markers[mid]), "robot_xyz": [float(v) for v in T[:3, 3]], "tilt_deg": tilt}

    px = [points[m]["pixel"] for m in sorted(points)]
    xy = [points[m]["robot_xyz"][:2] for m in sorted(points)]
    tmap, resid, loo = TableMap.fit(px, xy)
    zs = [points[m]["robot_xyz"][2] for m in points]
    if max(zs) - min(zs) > 0.02:
        say(f"Note: the touch heights varied by {(max(zs) - min(zs)) * 100:.1f} cm. Is the table flat and was the tip on the paper each time?")

    worst = max(loo) if loo else max(resid)
    say(f"Fit error: typical {np.median(loo or resid) * 1000:.1f} mm, worst {worst * 1000:.1f} mm.")
    if worst > 0.01:
        say("That's high. Re-run and touch the marker centers more carefully, or spread the markers out more.")

    ask("Put the arm back in its folded rest position, then press Enter.")
    h, w = img.shape[:2]
    return {
        "version": 1,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "homography": tmap.H.tolist(),
        "table_z": float(np.median(zs)),
        "image_size": [w, h],
        "markers": {str(k): v for k, v in points.items()},
        "gripper_closed": float(closed),
        "gripper_open": float(opened),
        "residual_mm": [r * 1000 for r in resid],
        "leave_one_out_mm": [r * 1000 for r in loo] if loo else None,
    }


def camera_moved(calib: dict, img: np.ndarray, tolerance_px: float = 8.0) -> tuple[bool | None, str]:
    """Compare marker positions now with calibration time. None = no marker visible to check."""
    seen = detect_markers(img)
    ref = {int(k): v["pixel"] for k, v in calib.get("markers", {}).items()}
    common = [m for m in seen if m in ref]
    if not common:
        return None, "no calibration marker visible, so I can't check whether the camera moved"
    shift = max(math.dist(seen[m], ref[m]) for m in common)
    if shift > tolerance_px:
        return True, f"marker(s) {common} moved {shift:.0f} px since calibration: the camera (or the markers) moved; re-run calibrate.py"
    return False, f"camera position OK (marker shift {shift:.1f} px)"
