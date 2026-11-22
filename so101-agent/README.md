# SO-101 agent: tell the arm what to do, no training

Claude looks through a fixed overhead camera and drives your SO-101 with a small set of safe,
pre-built moves: `look`, `pick`, `place`, `move_to`, `nudge`, `gripper`, `home`, `rest`.
After each action it checks the result and adjusts, so you can say "put the red block on the
blue one" and it works out the steps.

Two ways to talk to it, sharing the same tools:
- **Claude desktop app:** through the MCP server (`server.py`).
- **Terminal:** `agent.py` (needs an Anthropic API key). This is also where voice will plug in later.

## What you need

- The SO-101 follower arm, already set up with `lerobot-calibrate` (you know its port and id).
- A USB camera fixed above the table, looking down at the area in front of the arm. It must not
  move: if it gets bumped, recalibrate (takes about 2 minutes).
- A printer, for `markers.pdf`.

## 1. Install

Use the Python environment where LeRobot already runs your arm:

```bash
cd so101-agent
pip install -r requirements.txt
python -m pytest tests -q      # simulated arm and camera, no hardware: "13 passed"
```

## 2. Configure

Edit `config.yaml`:
- `robot.port`: the follower's port, from `lerobot-find-port`.
- `robot.id`: the id you used with `lerobot-calibrate`.
- `camera.index`: the overhead camera, from `lerobot-find-cameras opencv`.

Then put your API keys in a `.env` file. It's gitignored, so the keys never end up in the repo:

```bash
cp .env.example .env    # then paste your keys into .env
```

- `ANTHROPIC_API_KEY`: needed for the terminal agent (`agent.py`). The Claude desktop app doesn't need it.
- `GEMINI_API_KEY`: optional. It turns on `locate()` for precise object pointing with Gemini Robotics-ER.

Keys you `export` in your shell take priority over `.env`.

## 3. Markers

Print `markers.pdf` (any scale), cut the markers out, and tape 4 to 6 of them flat on the table:
- Inside the arm's reach: about 10 to 25 cm from the shoulder (the vertical axis the arm turns around).
- Spread out left/right and near/far, not in a line.
- Leave one taped down for good, in a spot you won't use. `look()` uses it to warn you if the
  camera moved.

Check what the camera sees:

```bash
python calibrate.py --preview     # writes camera_preview.jpg with the markers it found
```

## 4. Calibrate

```bash
python calibrate.py
```

1. The motors switch off, so hold the arm.
2. Close the gripper fully by hand, then open it fully, when asked.
3. Touch the center of each marker with the gripper tip: jaws closed, gripper pointing straight
   down, tip resting on the paper. Press Enter each time.

This saves `calibration.json`. Then check the result:

```bash
python calibrate.py --verify      # the arm hovers 2 cm over each marker
```

If the tip is more than about 5 mm off the centers, run `calibrate.py` again.

## 5. Use it from the terminal

With `ANTHROPIC_API_KEY` in `.env`:

```bash
python agent.py
you> pick up the red block and put it next to the blue one
```

## 6. Use it from the Claude desktop app

Add the server to the app's local MCP server config, `claude_desktop_config.json`. Use full paths:
`which python` (inside your LeRobot environment) gives the Python path.

```json
{
  "mcpServers": {
    "so101-arm": {
      "command": "/full/path/to/python",
      "args": ["/full/path/to/so101-agent/server.py"]
    }
  }
}
```

The server reads `GEMINI_API_KEY` from `.env` itself. Restart the app, then try "Look at the
table and pick up the red block."

## How it works

- **Seeing:** `look()` returns the camera image with a labeled pixel grid, the reachable area
  outlined in yellow, and the gripper marked. Claude reads object positions off the grid and
  can zoom in to be precise. `locate()` optionally asks Gemini Robotics-ER for exact points.
- **Pixels to the table:** the calibration maps any pixel to a spot on the table, in the arm's
  own coordinates. It's measured with the arm itself, so small model errors cancel out.
- **Moving:** inverse kinematics on the official SO-101 model (via placo). Moves follow straight
  lines with the gripper pointing down, and come down vertically for picks and places. Near
  the edge of its reach, the gripper may lean up to 25°.
- **Checking:** after `pick()`, the gripper position shows whether the jaws closed on something.
  Claude then looks again and retries if needed.

## Safety

- Speed is capped (60°/s by default), targets outside the reachable area are refused, and the
  gripper tip is never sent lower than 5 mm above the table.
- The arm can't feel anything. It won't notice a hand in the way. Keep its power plug within reach:
  unplugging it stops the arm immediately.
- **Start with the arm in its rest pose.** LeRobot briefly turns the motors off while it connects.
- When you're done, say "rest" (or call `rest()`): the arm goes back to its rest pose and the motors
  turn off. Quitting the agent or the app does the same.

## Tuning and troubleshooting

| Problem | Try |
|---|---|
| Misses, always toward the same side | `calibrate.py --verify`; recalibrate if off |
| Misses tall objects | Parallax: put the camera closer to directly above the workspace. A higher camera helps too |
| "can't reach" | Stay inside the yellow outline. Grasps from above reach about 25 cm from the shoulder |
| Says "MISSED" on thin objects | Lower `motion.holding_margin` |
| Image upside down | `camera.rotate_180: true` |
| Arm too fast | Lower `motion.max_joint_speed_deg_s` |

## Later: wrist camera

Set `camera.wrist_index` and Claude can use `look(camera="wrist")` for close-up checks. If you
swap the wrist-roll part to mount it, rerun `lerobot-calibrate` and then `calibrate.py`.

## Later: voice (XVF3800)

`agent.py` already exposes the one call voice needs: `await ArmAgent().ask(text)` returns the
reply text. A voice loop wraps that:

1. The XVF3800 in USB mode is a plug-and-play mic and speaker output.
2. Wake word (e.g. openWakeWord).
3. Record until silence.
4. Speech-to-text (e.g. faster-whisper).
5. `ask(text)`.
6. Text-to-speech out the XVF3800's headphone jack.

The XVF3800's echo cancellation lets it hear you while the robot talks.

## Files

| File | What it is |
|---|---|
| `server.py` | MCP server and tool definitions (used by the Claude app and `agent.py`) |
| `agent.py` | Terminal agent; `ArmAgent.ask()` is the voice hook |
| `calibrate.py` | Camera-to-arm calibration, `--verify`, `--preview` |
| `make_markers.py` / `markers.pdf` | Printable markers |
| `so101_agent/` | Arm control, kinematics, vision, skills |
| `tests/` | Offline tests with a simulated arm and camera |

The arm model in `so101_agent/assets/` is a kinematics-only copy of `so101_new_calib.urdf` from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
