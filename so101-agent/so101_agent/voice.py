"""Voice front end: wake word -> speech-to-text -> the arm agent -> spoken reply.

Built for the ReSpeaker XVF3800 in USB mode, but works with any microphone. Audio is 16 kHz
mono int16 in 80 ms frames (1280 samples), which is what openWakeWord expects.

While the arm is working, saying the wake word again freezes it at once and cancels the
request. Replies are played on the XVF3800 so its echo canceller can hear (and remove) them.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .config import VoiceCfg

log = logging.getLogger(__name__)

RATE = 16000
FRAME = 1280  # 80 ms
XVF_NAMES = ("xvf3800", "respeaker")

STOP_WORDS = {"stop", "stop stop", "freeze", "halt", "cancel", "never mind", "nevermind", "wait", "stop it", "stop now"}
# Things Whisper tends to "hear" in silence or noise.
HALLUCINATIONS = {"", "you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "okay", "so"}

VOICE_PROMPT = """\
The user is talking to you out loud. Their words come from speech recognition and may have small mistakes
(for example "read" for "red", "black" for "block", "cop" for "cup"); use the scene to work out what they meant.
Your reply is spoken aloud: one or two short sentences, plain words, no lists, no markdown, no pixel coordinates.
If a request is unclear, ask one short question. Children may be talking to you: keep it friendly and simple,
and never do anything near a person's hand.
"""

STT_PROMPT = ("Commands for a robot arm: pick up, put down, place, stack, move, point at, open, close, home, rest, stop. "
              "Objects: block, cube, cup, bowl, ball, marker. Colors: red, blue, green, yellow, orange, purple, pink, white, black.")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


# --------------------------------------------------------------------------- devices
def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def find_device(hint: str, kind: str) -> int | None:
    """Device index for `hint` ("auto" = the XVF3800 if present, "" = system default)."""
    import sounddevice as sd

    key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d[key] > 0]
    hints = XVF_NAMES if hint == "auto" else ((hint.lower(),) if hint else ())
    for h in hints:
        for i, d in devices:
            if h in d["name"].lower():
                return i
    if hint not in ("auto", ""):
        raise RuntimeError(f"no {kind} device matching {hint!r}. Devices:\n{list_devices()}")
    return None  # system default


def device_name(index: int | None, kind: str) -> str:
    import sounddevice as sd

    return sd.query_devices(index, kind)["name"]


# --------------------------------------------------------------------------- microphone
class Microphone:
    """Streams 80 ms int16 frames from one channel of the input device into a queue."""

    def __init__(self, cfg: VoiceCfg):
        import sounddevice as sd

        self.device = find_device(cfg.input_device, "input")
        info = sd.query_devices(self.device, "input")
        self.name = info["name"]
        self.channels = int(info["max_input_channels"])
        is_xvf = any(n in self.name.lower() for n in XVF_NAMES)
        self.channel = min(cfg.input_channel if is_xvf else 0, self.channels - 1)
        self.is_xvf = is_xvf
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=RATE, blocksize=FRAME, device=self.device, channels=self.channels,
            dtype="int16", callback=self._callback,
        )

    def _callback(self, indata, n, t, status):
        if status:
            log.debug("audio input: %s", status)
        self.frames.put(indata[:, self.channel].copy())

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None


# --------------------------------------------------------------------------- output
def tone(freqs: list[float], dur: float = 0.12, volume: float = 0.25) -> np.ndarray:
    parts = []
    for f in freqs:
        t = np.arange(int(RATE * dur)) / RATE
        env = np.minimum(1, np.minimum(t, dur - t) / 0.01)  # 10 ms fade in/out
        parts.append(np.sin(2 * np.pi * f * t) * env * volume)
    return (np.concatenate(parts) * 32767).astype(np.int16)


class Speaker:
    """Plays chimes and speech (macOS `say`) on the output device, without blocking."""

    def __init__(self, cfg: VoiceCfg):
        import sounddevice as sd

        self.sd = sd
        self.device = find_device(cfg.output_device, "output")
        info = sd.query_devices(self.device, "output")
        self.name = info["name"]
        self.channels = min(2, int(info["max_output_channels"]))
        self.echo_cancelled = any(n in self.name.lower() for n in XVF_NAMES)
        self.voice = cfg.say_voice
        self.chimes = cfg.chimes
        tts = cfg.tts
        if tts == "auto":
            tts = "say" if shutil.which("say") else "none"
        self.tts = tts
        self._until = 0.0

    @property
    def busy(self) -> bool:
        return time.monotonic() < self._until

    def play(self, samples: np.ndarray) -> None:
        if self.channels > 1:  # same sound on both sides; the echo canceller listens to the left one
            samples = np.column_stack([samples] * self.channels)
        self.sd.play(samples, RATE, device=self.device)
        self._until = time.monotonic() + len(samples) / RATE + 0.1

    def stop(self) -> None:
        self.sd.stop()
        self._until = 0.0

    def chime(self, kind: str) -> None:
        if not self.chimes:
            return
        notes = {"wake": [660, 990], "thinking": [520], "error": [440, 330], "stop": [880, 440]}[kind]
        self.play(tone(notes))

    def say(self, text: str) -> None:
        print(f"arm: {text}", flush=True)
        if self.tts != "say" or not text:
            return
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reply.wav"
            cmd = ["say", "-o", str(path), "--file-format=WAVE", f"--data-format=LEI16@{RATE}"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd, input=text, text=True, check=True, capture_output=True)  # text via stdin
            audio = read_wav_int16(path)
        self.play(audio)


def read_wav_int16(path) -> np.ndarray:
    """16-bit mono samples from a WAV file, including the WAVE_FORMAT_EXTENSIBLE kind that
    older Python `wave` modules refuse."""
    data = Path(path).read_bytes()
    try:
        with wave.open(str(path)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except wave.Error:
        pass
    pos = 12  # skip "RIFF" <size> "WAVE"
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if cid == b"data":
            return np.frombuffer(data[pos + 8:pos + 8 + size], dtype=np.int16)
        pos += 8 + size + (size & 1)
    raise ValueError(f"no audio in {path}")


# --------------------------------------------------------------------------- recognition
class WakeWord:
    def __init__(self, cfg: VoiceCfg):
        import openwakeword.utils
        from openwakeword.model import Model

        name = cfg.wake_word
        if not name.endswith((".onnx", ".tflite")):
            openwakeword.utils.download_models(model_names=[name])  # first run only; tiny
        self.model = Model(wakeword_models=[name], inference_framework="onnx")
        self.threshold = cfg.wake_threshold
        self.label = name
        self._cooldown = 0  # frames; counted in audio time, not wall-clock time

    def detect(self, frame: np.ndarray) -> bool:
        scores = self.model.predict(frame)
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if max(scores.values()) >= self.threshold:
            self.model.reset()
            self._cooldown = int(1.5 * RATE / FRAME)  # one wake word, one trigger
            return True
        return False


class Endpointer:
    """Collects audio after the wake word until the speaker goes quiet."""

    def __init__(self, cfg: VoiceCfg, vad=None, start_timeout_s: float = 5.0):
        if vad is None:
            import webrtcvad

            v = webrtcvad.Vad(2)
            vad = lambda chunk: v.is_speech(chunk.tobytes(), RATE)  # noqa: E731
        self.vad = vad
        self.end_silence = cfg.end_silence_s
        self.max_len = cfg.max_command_s
        self.start_timeout = start_timeout_s
        self.reset()

    def reset(self, start_timeout_s: float | None = None, skip_s: float = 0.0) -> None:
        """Start a new command. `skip_s` ignores the first bit of audio (our own chime)."""
        if start_timeout_s is not None:
            self.start_timeout = start_timeout_s
        self.skip = int(round(skip_s * RATE / FRAME))
        self.audio: list[np.ndarray] = []
        self.heard = 0.0
        self.quiet = 0.0
        self.elapsed = 0.0
        self.started = False

    def feed(self, frame: np.ndarray) -> str | None:
        """Returns None (keep listening), "done" (got a command) or "timeout" (nobody spoke)."""
        if self.skip > 0:
            self.skip -= 1
            return None
        self.audio.append(frame)
        dt = len(frame) / RATE
        self.elapsed += dt
        voiced = sum(bool(self.vad(frame[i:i + 320])) for i in range(0, len(frame) - 319, 320))  # 20 ms chunks
        if voiced >= 2:
            self.heard += dt
            self.quiet = 0.0
            if self.heard >= 0.2:
                self.started = True
        else:
            self.quiet += dt
        if self.started and (self.quiet >= self.end_silence or self.elapsed >= self.max_len):
            return "done"
        if not self.started and self.elapsed >= self.start_timeout:
            return "timeout"
        return None

    def samples(self) -> np.ndarray:
        return np.concatenate(self.audio) if self.audio else np.zeros(0, np.int16)


class Transcriber:
    def __init__(self, cfg: VoiceCfg):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(cfg.stt_model, device="cpu", compute_type="int8")

    def __call__(self, samples: np.ndarray) -> str:
        audio = samples.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, condition_on_previous_text=False,
            initial_prompt=STT_PROMPT, vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()


# --------------------------------------------------------------------------- the loop
class AgentRunner:
    """Runs ArmAgent.ask() on its own event-loop thread so the mic keeps being heard."""

    def __init__(self, agent):
        import asyncio

        self.agent = agent
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def submit(self, text: str):
        import asyncio

        return asyncio.run_coroutine_threadsafe(self.agent.ask(text), self.loop)

    def cancel(self) -> None:
        self.agent.cancel()


class VoiceLoop:
    IDLE, LISTENING, ACTING, SPEAKING = "idle", "listening", "acting", "speaking"

    def __init__(self, cfg: VoiceCfg, read_frame, wake, endpointer, transcribe, speaker, runner, echo_cancelled: bool):
        self.cfg = cfg
        self.read_frame = read_frame
        self.wake, self.ep, self.transcribe = wake, endpointer, transcribe
        self.speaker, self.runner = speaker, runner
        self.echo_cancelled = echo_cancelled
        self.state = self.IDLE
        self.future = None
        self.stopping = False
        self.last_reply = ""
        self.log: list[tuple[str, str]] = []  # (who, text), for tests

    def _status(self, msg: str) -> None:
        print(msg, flush=True)

    def _listen(self, followup: bool = False) -> None:
        chime = not followup and self.cfg.chimes
        self.ep.reset(self.cfg.followup_s if followup else 5.0, skip_s=0.25 if chime else 0.0)
        self.state = self.LISTENING
        if chime:
            self.speaker.chime("wake")
        self._status("listening…")

    def _handle(self, text: str) -> None:
        said = normalize(text)
        if said in HALLUCINATIONS:
            self._status("(didn't catch that)")
            self.speaker.chime("error")
            self.state = self.IDLE
            return
        self._status(f"you: {text}")
        self.log.append(("you", text))
        if said in STOP_WORDS:
            self.runner.cancel()
            self._reply("Okay.")
            return
        self.speaker.chime("thinking")
        self.stopping = False
        self.future = self.runner.submit(text)
        self.state = self.ACTING

    def _reply(self, text: str) -> None:
        self.last_reply = text
        self.log.append(("arm", text))
        self.speaker.say(text)
        self.state = self.SPEAKING

    def step(self, frame: np.ndarray) -> None:
        """Process one 80 ms frame of audio."""
        # The wake-word model hears every frame (it needs continuous audio), but a detection only
        # counts when we aren't recording a command and, without echo cancellation, aren't
        # talking ourselves (we'd hear our own voice).
        hit = self.wake.detect(frame)
        can_wake = self.state != self.LISTENING and (self.echo_cancelled or not self.speaker.busy)
        woke = hit and can_wake

        if self.state == self.IDLE:
            if woke:
                self._listen()
        elif self.state == self.LISTENING:
            result = self.ep.feed(frame)
            if result == "done":
                self._status("…")
                self._handle(self.transcribe(self.ep.samples()))
            elif result == "timeout":
                self.state = self.IDLE
        elif self.state == self.ACTING:
            if woke and not self.stopping:  # barge in: freeze the arm and drop the request
                self.stopping = True
                self.runner.cancel()
                self.speaker.chime("stop")
                self._status("(stopping)")
            if self.future is not None and self.future.done():
                try:
                    reply = self.future.result()
                except Exception as e:  # API down, bad key, ...
                    log.exception("agent failed")
                    reply = f"Sorry, something went wrong: {e}"
                self.future = None
                if self.stopping:
                    self._reply("Stopped. What should I do?")
                else:
                    self._reply(reply)
        elif self.state == self.SPEAKING:
            if woke:
                self.speaker.stop()
                self._listen()
            elif not self.speaker.busy:
                if self.last_reply.rstrip().endswith("?") and self.cfg.followup_s > 0:
                    self._listen(followup=True)
                else:
                    self.state = self.IDLE

    def run(self) -> None:
        while True:
            frame = self.read_frame()
            if frame is not None:
                self.step(frame)
            elif self.state == self.ACTING or self.state == self.SPEAKING:
                self.step(np.zeros(FRAME, np.int16))  # keep the state machine moving


def main(cfg, agent) -> None:
    """Wire up the real microphone, models and speaker, then listen forever."""
    v = cfg.voice
    print("Loading speech models (the first run downloads them)…", flush=True)
    wake = WakeWord(v)
    transcribe = Transcriber(v)
    speaker = Speaker(v)
    with Microphone(v) as mic:
        print(f"Mic: {mic.name} (channel {mic.channel})   Speaker: {speaker.name}", flush=True)
        if not mic.is_xvf:
            print("Note: the XVF3800 wasn't found, using the default mic.", flush=True)
        if not speaker.echo_cancelled:
            print("Note: replies aren't going through the XVF3800, so it can't cancel their echo. "
                  "You can't interrupt while it's talking.", flush=True)
        words = v.wake_word.replace("_", " ").removesuffix(".onnx")
        print(f'Say "{words}", then a command. Say "{words}" again to stop the arm. Ctrl-C to quit.', flush=True)
        loop = VoiceLoop(v, mic.read, wake, Endpointer(v), transcribe, speaker, AgentRunner(agent),
                         echo_cancelled=speaker.echo_cancelled)
        try:
            loop.run()
        except KeyboardInterrupt:
            agent.cancel()
            print("\nBye.", file=sys.stderr)
