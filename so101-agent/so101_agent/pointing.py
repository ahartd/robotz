"""Optional: precise object pointing with Gemini Robotics-ER.

Claude can read coordinates off the gridded image itself; this is a second opinion that is
usually more precise for small objects. Needs `pip install google-genai` and an API key.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .vision import encode_jpeg


class PointingError(RuntimeError):
    pass


PROMPT = (
    "Point to {what} in the image. Answer with JSON only, no prose: a list like "
    '[{{"point": [y, x], "label": "<short name>"}}] where the point is the center of the object '
    "and coordinates are normalized to 0-1000. Include one entry per matching object, at most 10. "
    "If nothing matches, answer []."
)


def parse_points(text: str, width: int, height: int) -> list[tuple[float, float, str]]:
    """Turn Gemini's normalized [y, x] answer into pixel (u, v, label) tuples."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise PointingError(f"unexpected answer from Gemini: {text[:200]!r}")
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PointingError(f"couldn't read Gemini's answer: {e}") from None
    out = []
    for it in items:
        y, x = it["point"][:2]
        out.append((float(x) / 1000.0 * width, float(y) / 1000.0 * height, str(it.get("label", "object"))))
    return out


def gemini_point(img: np.ndarray, what: str, model: str, api_key_env: str = "GEMINI_API_KEY"):
    key = os.environ.get(api_key_env)
    if not key:
        raise PointingError(
            f"{api_key_env} isn't set, so precise pointing is off. Read the pixel coordinates "
            "from the grid in look() instead (zoom in for precision)."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise PointingError("install google-genai to use locate(): pip install google-genai") from None

    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=encode_jpeg(img), mime_type="image/jpeg"), PROMPT.format(what=what)],
        config=types.GenerateContentConfig(temperature=0.3),
    )
    h, w = img.shape[:2]
    return parse_points(resp.text or "", w, h)
