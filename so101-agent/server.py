"""MCP server: lets Claude (the Claude desktop app, or agent.py) see through the overhead
camera and move the SO-101.

Run by the Claude app over stdio (see README). Nothing here may print to stdout: stdout
carries the MCP protocol. Logs go to stderr.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Annotated, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import Field  # noqa: E402

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402
from so101_agent.config import load_config  # noqa: E402
from so101_agent.skills import Skills, ToolResult  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s: %(message)s")

INSTRUCTIONS = """\
You control a real SO-101 robot arm on a table, seen by a fixed overhead camera.

How to work:
1. Start with look(clear_view=true) so the arm is folded out of the way. Find objects in the image and read their
   pixel coordinates (u right, v down) from the grid. For small objects, zoom in with look(zoom_u, zoom_v, zoom=3)
   to read the center precisely. locate(description) gives precise points if it is set up.
2. Act with pixel coordinates: pick(u, v), place(u, v), move_to(u, v). Aim at the CENTER of the object's footprint
   on the table. Only use spots inside the yellow outline (the reachable area).
3. After each pick or place, check the result: pick() reports whether the jaws closed on something, and
   look(clear_view=true) shows what happened. If it missed, adjust and try again (at most 3 tries, then ask).
4. For long thin objects, set grasp_angle_deg so the jaws close across the narrow side (try 90 if 0 fails).
   For stacking, place(u, v, height_cm=<height of the stack> + 1).
5. Do one physical action at a time and explain briefly what you are doing.

Safety: this is a real arm, possibly around children. Never move toward a hand or a person in the image; if a hand
is in the workspace, stop and ask. Don't try to lift heavy, sharp, hot or fragile things. Call rest() when the
user is done.
"""

cfg = load_config()
skills = Skills(cfg)
mcp = MCPServer("so101-arm", instructions=INSTRUCTIONS)


def _out(r: ToolResult) -> list:
    return [r.text] + [Image(data=jpg, format="jpeg") for jpg in r.images]


@mcp.tool()
def look(
    clear_view: Annotated[bool, Field(description="Fold the arm out of the way before taking the picture.")] = False,
    zoom_u: Annotated[float | None, Field(description="Pixel u (horizontal) to zoom in on.")] = None,
    zoom_v: Annotated[float | None, Field(description="Pixel v (vertical) to zoom in on.")] = None,
    zoom: Annotated[float, Field(ge=1, le=8, description="Zoom factor, 1 = whole image.")] = 1.0,
    camera: Annotated[Literal["overhead", "wrist"], Field(description="Which camera.")] = "overhead",
) -> list:
    """Take a picture with a labeled pixel grid, the reachable area outlined, and the gripper marked.
    Coordinates on the grid are always full-image pixels, even when zoomed."""
    return _out(skills.look(clear_view, zoom_u, zoom_v, zoom, camera))


@mcp.tool()
def locate(description: Annotated[str, Field(description="What to find, e.g. 'the red block'.")]) -> list:
    """Find objects precisely with Gemini Robotics-ER pointing (only if GEMINI_API_KEY is set).
    Returns pixel coordinates you can pass to pick/place."""
    return _out(skills.locate(description))


@mcp.tool()
def pick(
    u: Annotated[float, Field(description="Pixel u of the object's center.")],
    v: Annotated[float, Field(description="Pixel v of the object's center.")],
    grasp_angle_deg: Annotated[float, Field(ge=-90, le=90, description="Rotate the jaws around the vertical axis. 0 = default; for long thin objects or a missed grasp, try 90 or -45.")] = 0.0,
    height_cm: Annotated[float | None, Field(ge=0.5, le=10, description="Grasp height above the table (default about 1.2 cm).")] = None,
) -> list:
    """Pick up the object at pixel (u, v): open, come down from above, close, lift.
    Reports whether something seems to be held."""
    return _out(skills.pick(u, v, grasp_angle_deg, height_cm))


@mcp.tool()
def place(
    u: Annotated[float, Field(description="Pixel u where to put the held object.")],
    v: Annotated[float, Field(description="Pixel v where to put the held object.")],
    height_cm: Annotated[float | None, Field(ge=0.5, le=10, description="Release height above the table (default 2 cm). For stacking use the stack height + 1.")] = None,
) -> list:
    """Put down whatever the gripper is holding at pixel (u, v)."""
    return _out(skills.place(u, v, height_cm))


@mcp.tool()
def move_to(
    u: Annotated[float, Field(description="Pixel u.")],
    v: Annotated[float, Field(description="Pixel v.")],
    height_cm: Annotated[float, Field(ge=0.5, le=12, description="Height of the gripper tip above the table.")] = 5.0,
) -> list:
    """Move the gripper (pointing down) over pixel (u, v), e.g. to point at something."""
    return _out(skills.move_to(u, v, height_cm))


@mcp.tool()
def nudge(
    du_px: Annotated[float, Field(ge=-80, le=80, description="Move right (+) or left (-) in the image, pixels.")] = 0.0,
    dv_px: Annotated[float, Field(ge=-80, le=80, description="Move down (+) or up (-) in the image, pixels.")] = 0.0,
    dz_cm: Annotated[float, Field(ge=-5, le=5, description="Move up (+) or down (-), cm.")] = 0.0,
) -> list:
    """Small, slow correction of the gripper position relative to where it is now."""
    return _out(skills.nudge(du_px, dv_px, dz_cm))


@mcp.tool()
def gripper(open_percent: Annotated[float, Field(ge=0, le=100, description="0 = closed, 100 = fully open.")]) -> list:
    """Open or close the jaws without moving the arm."""
    return _out(skills.gripper(open_percent))


@mcp.tool()
def home() -> list:
    """Fold the arm up near its base, out of the camera's way."""
    return _out(skills.home())


@mcp.tool()
def status() -> list:
    """Where the gripper is, whether it's holding something, and whether calibration is done."""
    return _out(skills.status())


@mcp.tool()
def rest() -> list:
    """Return to the rest pose and turn the motors off. Use when finished."""
    return _out(skills.rest())


_shutting_down = False


def _shutdown(*_):
    """Park the arm in its rest pose before the server goes away."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # don't get killed halfway through a move
    skills.shutdown()


def _on_sigterm(*_):
    _shutdown()
    os._exit(0)


atexit.register(_shutdown)
signal.signal(signal.SIGTERM, _on_sigterm)

if __name__ == "__main__":
    mcp.run("stdio")
