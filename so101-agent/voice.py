"""Talk to the arm: say the wake word ("hey jarvis" by default), then a command.

    python voice.py                 # listen on the XVF3800 and answer out loud
    python voice.py --list-devices  # see microphones and speakers
    python voice.py --type          # type commands but hear spoken replies (tests the speaker)

Saying the wake word while the arm is moving stops it immediately. Needs ANTHROPIC_API_KEY
(in .env). Settings are in the `voice:` section of config.yaml (see the README).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--type", action="store_true", help="type commands instead of speaking them")
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    from so101_agent import voice

    if a.list_devices:
        print(voice.list_devices())
        return

    import asyncio

    import server
    from agent import ArmAgent

    agent = ArmAgent(extra_system=voice.VOICE_PROMPT, on_event=lambda m: print(m, file=sys.stderr))
    try:
        if a.type:
            speaker = voice.Speaker(server.cfg.voice)
            loop = asyncio.new_event_loop()
            while True:
                try:
                    text = input("\nyou> ").strip()
                except EOFError:
                    break
                if text:
                    speaker.say(loop.run_until_complete(agent.ask(text)))
        else:
            voice.main(server.cfg, agent)
    finally:
        server.skills.shutdown()


if __name__ == "__main__":
    main()
