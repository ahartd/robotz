"""Camera-to-arm calibration (about 2 minutes). Re-run whenever the camera moves.

    python calibrate.py            # calibrate
    python calibrate.py --verify   # arm hovers over each marker so you can check accuracy
    python calibrate.py --preview  # just show what the camera sees and which markers it finds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from so101_agent.arm import Arm, ArmError  # noqa: E402
from so101_agent.calibration import CalibrationError, load_calibration, save_calibration, touch_calibration  # noqa: E402
from so101_agent.config import load_config  # noqa: E402
from so101_agent.vision import Camera, CameraError, detect_markers  # noqa: E402


def ask(msg: str) -> None:
    input(f"\n{msg}\n> ")


def preview(cfg) -> None:
    cam = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.rotate_180)
    img = cam.capture()
    markers = detect_markers(img)
    for mid, (u, v) in markers.items():
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, str(mid), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out = cfg.resolve("camera_preview.jpg")
    cv2.imwrite(str(out), img)
    cam.close()
    print(f"Markers found: {sorted(markers) or 'none'}. Saved the image to {out}")


def verify(cfg) -> None:
    calib = load_calibration(cfg.calibration_path)
    if not calib:
        sys.exit("No calibration yet. Run `python calibrate.py` first.")
    arm = Arm(cfg)
    arm.apply_calibration(calib)
    try:
        ask("The arm is about to move. Keep hands clear, then press Enter.")
        for mid, m in sorted(calib["markers"].items(), key=lambda kv: int(kv[0])):
            x, y, _ = m["robot_xyz"]
            arm.move_to(x, y, 0.02)
            ask(f"The tip should be about 2 cm above the center of marker {mid}. Press Enter for the next one.")
        arm.home()
    finally:
        arm.close()


def calibrate(cfg) -> None:
    print(__doc__)
    print("Before starting: tape at least 4 markers (6 is better) flat on the table, spread across the area "
          "in front of the arm, all visible to the camera. Leave one in a corner permanently so the agent "
          "can tell if the camera gets bumped.")
    cam = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.rotate_180)
    arm = Arm(cfg)
    try:
        data = touch_calibration(arm, cam, ask)
    finally:
        arm.backend.close()  # motors are already off
        cam.close()
    data["camera_index"] = cfg.camera.index
    save_calibration(cfg.calibration_path, data)
    print(f"\nSaved {cfg.calibration_path}. Next: `python calibrate.py --verify`.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    try:
        if a.preview:
            preview(cfg)
        elif a.verify:
            verify(cfg)
        else:
            calibrate(cfg)
    except (ArmError, CameraError, CalibrationError) as e:
        sys.exit(f"\nStopped: {e}")


if __name__ == "__main__":
    main()
