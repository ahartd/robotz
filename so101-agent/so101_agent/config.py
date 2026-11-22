"""Settings for the SO-101 agent, loaded from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent


@dataclass
class RobotCfg:
    port: str = ""
    id: str = "my_follower_arm"
    calibration_dir: str | None = None  # None = LeRobot's default calibration folder
    max_relative_target_deg: float | None = 30.0  # hardware-level clamp on any single command
    mock: bool = False  # simulated arm, for testing without hardware


@dataclass
class CameraCfg:
    index: int | str = 0  # from `lerobot-find-cameras opencv`
    width: int = 640
    height: int = 480
    rotate_180: bool = False  # set true if the image comes out upside down
    wrist_index: int | str | None = None  # optional wrist camera
    mock: bool = False


@dataclass
class MotionCfg:
    max_joint_speed_deg_s: float = 60.0
    control_hz: float = 50.0
    hover_height_m: float = 0.05  # travel height above the table between picks
    grasp_height_m: float = 0.012  # where the gripper tip closes, above the table
    place_height_m: float = 0.02  # where the gripper opens when placing
    open_fraction: float = 0.7  # how far to open the jaws, 0..1 of the calibrated range
    holding_margin: float = 4.0  # gripper units above "fully closed" that count as holding
    home_joints_deg: list[float] = field(default_factory=lambda: [0.0, -90.0, 60.0, 90.0, 0.0])
    max_tilt_deg: float = 25.0  # how far from straight-down the gripper may lean when needed


@dataclass
class SafetyCfg:
    min_radius_m: float = 0.08  # no targets closer than this to the shoulder (pan) axis
    max_radius_m: float = 0.28  # or farther than this
    max_abs_angle_deg: float = 80.0  # left/right limit around the base
    min_height_m: float = 0.005  # gripper tip never commanded below this above the table
    max_height_m: float = 0.12
    joint_limit_margin_deg: float = 3.0


@dataclass
class PointingCfg:
    gemini_model: str = "gemini-robotics-er-2-preview"
    gemini_api_key_env: str = "GEMINI_API_KEY"


@dataclass
class AgentCfg:
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    keep_images: int = 2  # older camera images are dropped from the conversation


@dataclass
class VoiceCfg:
    input_device: str = "auto"  # "auto" finds the XVF3800, else part of a device name, "" = system default
    input_channel: int = 1  # XVF3800 USB firmware: 0 = conference mix, 1 = speech-recognition beam
    output_device: str = "auto"  # play replies on the XVF3800 so its echo cancellation can hear them
    wake_word: str = "hey_jarvis"  # an openWakeWord model name, or a path to your own .onnx
    wake_threshold: float = 0.5
    stt_model: str = "small.en"  # faster-whisper model: tiny.en, base.en, small.en, medium.en
    tts: str = "auto"  # auto (macOS `say`), say, none (print only)
    say_voice: str | None = None  # a `say -v` voice name, e.g. "Samantha"
    end_silence_s: float = 0.8  # this much quiet ends a command
    max_command_s: float = 12.0
    followup_s: float = 6.0  # after Claude asks a question, listen this long without the wake word
    chimes: bool = True


@dataclass
class Config:
    robot: RobotCfg = field(default_factory=RobotCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    pointing: PointingCfg = field(default_factory=PointingCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    voice: VoiceCfg = field(default_factory=VoiceCfg)
    calibration_file: str = "calibration.json"
    base_dir: Path = field(default=PROJECT_DIR, repr=False)

    def resolve(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def calibration_path(self) -> Path:
        return self.resolve(self.calibration_file)


def _merge(obj: Any, data: dict, where: str) -> None:
    known = {f.name: f for f in fields(obj) if f.name != "base_dir"}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Unknown setting '{where}{key}' in config.yaml")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"'{where}{key}' should be a section, got {value!r}")
            _merge(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_env_file(path: str | Path) -> None:
    """Put KEY=value lines from a .env file into the environment.

    Variables that are already set (e.g. exported in your shell) win over the file.
    Blank values are skipped, so an untouched `GEMINI_API_KEY=` line changes nothing.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml (or $SO101_AGENT_CONFIG), plus API keys from the .env file next to it.
    Missing settings keep their defaults."""
    path = Path(path or os.environ.get("SO101_AGENT_CONFIG") or PROJECT_DIR / "config.yaml").expanduser()
    load_env_file(path.parent / ".env")
    cfg = Config(base_dir=path.parent.resolve())
    if path.exists():
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {}, "")
    return cfg
