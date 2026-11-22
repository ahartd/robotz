"""Make markers.pdf: six ArUco markers (ids 0-5) to print, cut out and tape to the table.

Print at any scale; only the marker centers are used. Keep a white border around each one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from so101_agent.vision import marker_image  # noqa: E402

DPI = 300
MM = DPI / 25.4


def main(out: str = "markers.pdf") -> None:
    W, H = int(8.5 * DPI), int(11 * DPI)  # US Letter
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 42)
        small = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((int(15 * MM), int(12 * MM)), "SO-101 calibration markers: cut along the gray lines, keep the white border.", font=small, fill=0)
    d.text((int(15 * MM), int(18 * MM)), "Tape them flat, spread across the area in front of the arm. Leave one in a corner for good.", font=small, fill=0)

    size = int(45 * MM)  # marker
    cell_w, cell_h = int(85 * MM), int(80 * MM)
    left, top = (W - 2 * cell_w) // 2, int(32 * MM)
    for i in range(6):
        col, row = i % 2, i // 2
        cx, cy = left + col * cell_w, top + row * cell_h
        d.rectangle([cx, cy, cx + cell_w, cy + cell_h], outline=170, width=3)
        m = Image.fromarray(np.asarray(marker_image(i, size)))
        mx, my = cx + (cell_w - size) // 2, cy + int(10 * MM)
        page.paste(m, (mx, my))
        d.text((cx + int(8 * MM), cy + cell_h - int(14 * MM)), f"marker {i}", font=font, fill=0)
    page.save(out, "PDF", resolution=DPI)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
