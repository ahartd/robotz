"""Voice loop and stop tests with fake audio, models and agent. No microphone needed."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import so101_agent.arm as arm_mod  # noqa: E402
from so101_agent.arm import Arm, ArmStopped, MockBackend  # noqa: E402
from so101_agent.config import Config, VoiceCfg  # noqa: E402
from so101_agent.voice import FRAME, Endpointer, VoiceLoop, normalize, tone  # noqa: E402

arm_mod.time.sleep = lambda s: None

SILENCE = np.zeros(FRAME, np.int16)
SPEECH = (np.random.default_rng(0).normal(0, 3000, FRAME)).astype(np.int16)
fake_vad = lambda chunk: float(np.abs(chunk).mean()) > 500  # noqa: E731


class FakeWake:
    def __init__(self):
        self.fire = False

    def detect(self, frame):
        hit, self.fire = self.fire, False
        return hit


class FakeSpeaker:
    busy = False

    def __init__(self):
        self.said, self.chimes, self.stopped = [], [], 0

    def say(self, text):
        self.said.append(text)

    def chime(self, kind):
        self.chimes.append(kind)

    def stop(self):
        self.stopped += 1


class FakeRunner:
    def __init__(self, replies):
        self.replies = list(replies)
        self.asked, self.cancels = [], 0
        self.pending: concurrent.futures.Future | None = None

    def submit(self, text):
        self.asked.append(text)
        f = concurrent.futures.Future()
        reply = self.replies.pop(0)
        if reply is None:
            self.pending = f  # finishes when the test says so
        else:
            f.set_result(reply)
        return f

    def cancel(self):
        self.cancels += 1


def make_loop(transcripts, replies, cfg=None):
    cfg = cfg or VoiceCfg()
    wake, speaker, runner = FakeWake(), FakeSpeaker(), FakeRunner(replies)
    texts = list(transcripts)
    loop = VoiceLoop(cfg, lambda: None, wake, Endpointer(cfg, vad=fake_vad), lambda samples: texts.pop(0),
                     speaker, runner, echo_cancelled=True)
    loop._status = lambda msg: None
    return loop, wake, speaker, runner


def feed(loop, frames):
    for f in frames:
        loop.step(f)


def say_command(loop, wake):
    wake.fire = True
    feed(loop, [SILENCE])  # wake word frame
    feed(loop, [SPEECH] * 10 + [SILENCE] * 12)  # 0.8 s speech, then 0.96 s quiet


def test_wake_command_and_spoken_reply():
    loop, wake, speaker, runner = make_loop(["Pick up the red block."], ["Got it, it's in my gripper."])
    feed(loop, [SILENCE] * 5)
    assert loop.state == loop.IDLE and not runner.asked
    say_command(loop, wake)
    feed(loop, [SILENCE] * 2)
    assert runner.asked == ["Pick up the red block."]
    assert speaker.said == ["Got it, it's in my gripper."]
    assert speaker.chimes[:2] == ["wake", "thinking"]
    assert loop.state == loop.IDLE  # no question, so no follow-up listening


def test_question_opens_a_follow_up_without_wake_word():
    loop, wake, speaker, runner = make_loop(["Pick up the block.", "The blue one."],
                                            ["Which one, the red or the blue block?", "Done."])
    say_command(loop, wake)
    feed(loop, [SILENCE] * 2)
    assert loop.state == loop.LISTENING  # answering the question
    feed(loop, [SPEECH] * 10 + [SILENCE] * 14)
    assert runner.asked == ["Pick up the block.", "The blue one."]
    assert speaker.said[-1] == "Done."


def test_wake_word_while_acting_stops_the_arm():
    loop, wake, speaker, runner = make_loop(["Stack all the blocks."], [None])
    say_command(loop, wake)
    feed(loop, [SILENCE] * 3)
    assert loop.state == loop.ACTING
    wake.fire = True
    feed(loop, [SILENCE])
    assert runner.cancels == 1 and "stop" in speaker.chimes
    runner.pending.set_result("Stopped.")
    feed(loop, [SILENCE])
    assert speaker.said == ["Stopped. What should I do?"]
    feed(loop, [SILENCE])
    assert loop.state == loop.LISTENING  # ready for the next instruction


def test_stop_word_and_noise_never_reach_the_agent():
    loop, wake, speaker, runner = make_loop(["Stop!", "Thank you."], [])
    say_command(loop, wake)
    assert runner.cancels == 1 and not runner.asked and speaker.said == ["Okay."]
    feed(loop, [SILENCE] * 2)
    say_command(loop, wake)
    assert not runner.asked and "error" in speaker.chimes


def test_endpointer_gives_up_if_nobody_speaks():
    ep = Endpointer(VoiceCfg(), vad=fake_vad, start_timeout_s=1.0)
    results = [ep.feed(SILENCE) for _ in range(14)]
    assert "timeout" in results and "done" not in results


def test_reads_extensible_wav(tmp_path):
    from so101_agent.voice import read_wav_int16

    samples = (np.arange(100) * 100).astype(np.int16)
    body = (b"WAVE" + b"fmt " + (40).to_bytes(4, "little") + (0xFFFE).to_bytes(2, "little") + bytes(38)
            + b"data" + len(samples.tobytes()).to_bytes(4, "little") + samples.tobytes())
    (tmp_path / "x.wav").write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)
    assert np.array_equal(read_wav_int16(tmp_path / "x.wav"), samples)


def test_normalize_and_tone():
    assert normalize("  Stop!! ") == "stop"
    t = tone([440, 880])
    assert t.dtype == np.int16 and len(t) == int(16000 * 0.12) * 2


# --------------------------------------------------------------------------- stopping the arm
def test_request_stop_freezes_a_move_and_blocks_new_ones(tmp_path):
    cfg = Config(base_dir=tmp_path)
    backend = MockBackend()
    arm = Arm(cfg, backend=backend)
    arm.table_z, arm.gripper_closed, arm.gripper_open = -0.0024, 1.5, 96.0
    writes = {"n": 0}
    orig = backend.write

    def write(targets):
        writes["n"] += 1
        if writes["n"] == 20:
            arm.request_stop()  # someone says "stop" partway through
        orig(targets)

    backend.write = write
    with pytest.raises(ArmStopped):
        arm.move_to(0.25, 0.05, 0.05)
    frozen = dict(backend.state)
    with pytest.raises(ArmStopped):
        arm.move_to(0.20, 0.0, 0.05)
    assert backend.state == frozen
    arm.clear_stop()
    arm.move_to(0.20, 0.0, 0.05)
    assert backend.state != frozen


def test_agent_cancel_ends_the_request_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_AGENT_CONFIG", str(tmp_path / "missing.yaml"))
    for mod in ("server", "agent"):
        sys.modules.pop(mod, None)
    import agent
    import server

    from so101_agent.skills import Skills
    from so101_agent.vision import MockCamera

    cfg = Config(base_dir=tmp_path)
    cfg.robot.mock = cfg.camera.mock = True
    server.skills = Skills(cfg, arm=Arm(cfg, backend=MockBackend()), camera=MockCamera())

    holder = {}

    class Messages:
        calls = 0

        async def create(self, **kw):
            Messages.calls += 1
            if Messages.calls == 2:
                holder["agent"].cancel()  # "stop" arrives while Claude is thinking
            content = [SimpleNamespace(type="tool_use", id=f"t{Messages.calls}", name="status", input={}),
                       SimpleNamespace(type="tool_use", id=f"u{Messages.calls}", name="home", input={})]
            return SimpleNamespace(content=content, stop_reason="tool_use")

    a = agent.ArmAgent(client=SimpleNamespace(messages=Messages()), on_event=lambda m: None)
    holder["agent"] = a
    reply = asyncio.run(a.ask("tidy up the table"))
    assert reply == agent.STOPPED
    assert server.skills.arm.stop_event.is_set()
    # every tool_use got a tool_result and the turn ends with the assistant
    for i, msg in enumerate(a.messages):
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
            ids = {b.id for b in msg["content"] if getattr(b, "type", "") == "tool_use"}
            results = {r["tool_use_id"] for r in a.messages[i + 1]["content"]}
            assert ids == results
    assert a.messages[-1]["role"] == "assistant"
    # the next request clears the stop
    Messages.calls = 10
    a.client = SimpleNamespace(messages=SimpleNamespace(create=_final_text))
    assert asyncio.run(a.ask("hello")) == "Hi!"
    assert not server.skills.arm.stop_event.is_set()


async def _final_text(**kw):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="Hi!")], stop_reason="end_turn")
