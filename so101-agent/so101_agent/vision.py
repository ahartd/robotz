"""Overhead camera, ArUco markers, and the pixel <-> table mapping."""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

MARKER_DICT = cv2.aruco.DICT_4X4_50


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index, width: int = 640, height: int = 480, rotate_180: bool = False):
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        self.index, self.width, self.height, self.rotate_180 = index, width, height, rotate_180
        self._cap = None
        self._lock = threading.Lock()

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(
                f"can't open camera {self.index!r}; check camera.index in config.yaml "
                "(`lerobot-find-cameras opencv` lists the cameras)"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(5):  # let exposure settle
            cap.read()
        self._cap = cap

    def capture(self) -> np.ndarray:
        with self._lock:
            for attempt in range(2):
                if self._cap is None:
                    self._open()
                for _ in range(4):  # drop stale buffered frames
                    self._cap.grab()
                ok, frame = self._cap.retrieve()
                if ok and frame is not None:
                    break
                self._cap.release()
                self._cap = None
            else:
                raise CameraError(f"camera {self.index!r} isn't returning images")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


class MockCamera:
    """Returns a fixed or generated image; for tests."""

    def __init__(self, image: np.ndarray | None = None, render=None, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self.image, self.render = image, render

    def capture(self) -> np.ndarray:
        if self.render is not None:
            return self.render()
        if self.image is not None:
            return self.image.copy()
        return np.full((self.height, self.width, 3), 200, np.uint8)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- markers
def detect_markers(img: np.ndarray) -> dict[int, tuple[float, float]]:
    """Centers (u, v) of the printed ArUco markers that are visible, by id."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): _diagonal_center(c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)}


def _diagonal_center(c: np.ndarray) -> tuple[float, float]:
    """Where the square's diagonals cross: its true center, even when seen at an angle."""
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        m = c.mean(axis=0)
        return float(m[0]), float(m[1])
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denom
    x = p + t * r
    return float(x[0]), float(x[1])


def marker_image(marker_id: int, size_px: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(MARKER_DICT), marker_id, size_px)


# --------------------------------------------------------------------------- table map
class TableMap:
    """Homography from image pixels to (x, y) on the table, in the arm's base frame (m)."""

    def __init__(self, H):
        self.H = np.asarray(H, dtype=float)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(H: np.ndarray, a: float, b: float) -> tuple[float, float]:
        p = H @ np.array([a, b, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_table(self, u: float, v: float) -> tuple[float, float]:
        return self._apply(self.H, u, v)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return self._apply(self.Hinv, x, y)

    @classmethod
    def fit(cls, pixels, table_xy) -> tuple["TableMap", list[float], list[float] | None]:
        """Fit from matched points. Returns (map, residuals_m, leave_one_out_errors_m)."""
        px = np.asarray(pixels, dtype=np.float64)
        xy = np.asarray(table_xy, dtype=np.float64)
        H, _ = cv2.findHomography(px, xy, 0)
        if H is None:
            raise ValueError("couldn't fit the camera mapping; are the markers spread out (not in a line)?")
        tm = cls(H)
        resid = [math.dist(tm.to_table(*p), q) for p, q in zip(px, xy)]
        loo = None
        if len(px) >= 5:
            loo = []
            for i in range(len(px)):
                keep = [j for j in range(len(px)) if j != i]
                Hi, _ = cv2.findHomography(px[keep], xy[keep], 0)
                loo.append(math.dist(cls(Hi).to_table(*px[i]), xy[i]) if Hi is not None else float("nan"))
        return tm, resid, loo


# --------------------------------------------------------------------------- drawing
def _nice_step(min_step: float) -> int:
    for s in (5, 10, 20, 25, 50, 100, 200):
        if s >= min_step:
            return s
    return 200


def _label(img, text, org, scale=0.4, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotate(
    img: np.ndarray,
    tmap: TableMap | None = None,
    reach: tuple | None = None,
    center: tuple[float, float] | None = None,
    zoom: float = 1.0,
    points: list[tuple[float, float, str]] | None = None,
    tip_xy: tuple[float, float] | None = None,
) -> tuple[bytes, tuple[float, float, float, float]]:
    """Draw a labeled pixel grid (in full-image coordinates) plus the reachable area.

    Returns (jpeg_bytes, (x0, y0, x1, y1) = region of the full image shown).
    """
    H, W = img.shape[:2]
    zoom = float(max(1.0, min(zoom, 8.0)))
    cw, ch = W / zoom, H / zoom
    cx, cy = center if center is not None else (W / 2, H / 2)
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)
    if zoom > 1.0:
        M = np.array([[zoom, 0, -x0 * zoom], [0, zoom, -y0 * zoom]], dtype=np.float64)
        view = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    else:
        view = img.copy()

    def to_view(u, v):
        return int(round((u - x0) * zoom)), int(round((v - y0) * zoom))

    # reachable area outline
    if tmap is not None and reach is not None:
        rmin, rmax, max_ang, (px, py) = reach
        angs = np.radians(np.linspace(-max_ang, max_ang, 40))
        outline = [(px + rmax * math.cos(a), py + rmax * math.sin(a)) for a in angs]
        outline += [(px + rmin * math.cos(a), py + rmin * math.sin(a)) for a in angs[::-1]]
        pts = np.array([to_view(*tmap.to_pixel(x, y)) for x, y in outline], np.int32)
        cv2.polylines(view, [pts], True, (0, 220, 255), 1, cv2.LINE_AA)
        bu, bv = to_view(*tmap.to_pixel(0.0, 0.0))
        if 0 <= bu < W and 0 <= bv < H:
            cv2.circle(view, (bu, bv), 6, (0, 220, 255), 2)
            _label(view, "arm base", (bu + 8, bv - 8), color=(0, 220, 255))

    # grid in full-image pixel coordinates
    step = _nice_step(60 / zoom)
    overlay = view.copy()
    for u in range(0, W + 1, step):
        if x0 <= u <= x0 + cw:
            vu = to_view(u, 0)[0]
            cv2.line(overlay, (vu, 0), (vu, H), (255, 255, 255), 1)
            _label(view, str(u), (vu + 2, 12))
    for v in range(0, H + 1, step):
        if y0 <= v <= y0 + ch:
            vv = to_view(0, v)[1]
            cv2.line(overlay, (0, vv), (W, vv), (255, 255, 255), 1)
            _label(view, str(v), (2, vv - 3))
    view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)

    if tip_xy is not None and tmap is not None:
        tu, tv = to_view(*tmap.to_pixel(*tip_xy))
        cv2.drawMarker(view, (tu, tv), (255, 80, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        _label(view, "gripper", (tu + 8, tv + 14), color=(255, 80, 255))
    for (u, v, text) in points or []:
        pu, pv = to_view(u, v)
        cv2.drawMarker(view, (pu, pv), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        _label(view, f"{text} ({u:.0f},{v:.0f})", (pu + 8, pv - 8), 0.45, (80, 80, 255))
    if zoom > 1.0:
        _label(view, f"zoom x{zoom:g} of region u {x0:.0f}-{x0 + cw:.0f}, v {y0:.0f}-{y0 + ch:.0f}", (8, H - 8), 0.45)

    ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise CameraError("couldn't encode the image")
    return jpg.tobytes(), (x0, y0, x0 + cw, y0 + ch)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraError("couldn't encode the image")
    return jpg.tobytes()
