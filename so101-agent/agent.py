"""Terminal agent: type a command, Claude looks through the camera and moves the arm.

    export ANTHROPIC_API_KEY=...
    python agent.py

It uses exactly the same tools as the MCP server, in-process. `ArmAgent.ask(text)` is the
one function a voice front end needs to call: speech-to-text in, reply text out.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading

import anthropic

import server
from server import INSTRUCTIONS, cfg, mcp

VOICE_NOTE = "\nReplies are short and conversational, one or two sentences, because they may be read aloud."


def _to_blocks(result) -> list[dict]:
    blocks = []
    for c in result.content:
        if c.type == "text":
            blocks.append({"type": "text", "text": c.text})
        elif c.type == "image":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": c.mime_type, "data": c.data}})
    return blocks or [{"type": "text", "text": "(no output)"}]


STOPPED = "Stopped."


class ArmAgent:
    def __init__(self, model: str | None = None, keep_images: int | None = None, client=None, on_event=None,
                 extra_system: str = ""):
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model or cfg.agent.model
        self.keep_images = cfg.agent.keep_images if keep_images is None else keep_images
        self.messages: list[dict] = []
        self.on_event = on_event or (lambda msg: print(msg, file=sys.stderr))
        self.system = INSTRUCTIONS + VOICE_NOTE + (("\n" + extra_system) if extra_system else "")
        self._tools = None
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Stop the current request: the arm freezes right away and no further steps run.
        Safe to call from another thread (e.g. the voice loop hearing "stop")."""
        self._cancel.set()
        server.skills.request_stop()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _finish_cancelled(self, calls, results: list[dict]) -> str:
        """Close out the turn so the conversation stays valid for the next request."""
        done = {r["tool_use_id"] for r in results}
        for call in calls:
            if call.id not in done:
                results.append({"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                                "content": [{"type": "text", "text": "Cancelled: the user said stop."}]})
        if results:
            self.messages.append({"role": "user", "content": results})
        self.messages.append({"role": "assistant", "content": "I stopped, as you asked."})
        return STOPPED

    async def tools(self) -> list[dict]:
        if self._tools is None:
            self._tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in await mcp.list_tools()
            ]
        return self._tools

    def _prune_images(self) -> None:
        """Keep only the most recent camera images; older ones become a short note."""
        seen = 0
        for msg in reversed(self.messages):
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for part in msg["content"]:
                if not isinstance(part, dict) or part.get("type") != "tool_result":
                    continue
                for i, b in enumerate(part["content"]):
                    if b.get("type") == "image":
                        seen += 1
                        if seen > self.keep_images:
                            part["content"][i] = {"type": "text", "text": "[older camera image removed]"}

    async def ask(self, text: str, max_steps: int = 25) -> str:
        self._cancel.clear()
        server.skills.clear_stop()
        self.messages.append({"role": "user", "content": text})
        tools = await self.tools()
        for _ in range(max_steps):
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=cfg.agent.max_tokens,
                system=self.system,
                tools=tools,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if self.cancelled:
                return self._finish_cancelled(calls, [])
            if resp.stop_reason != "tool_use" or not calls:
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            results = []
            for call in calls:
                if self.cancelled:
                    return self._finish_cancelled(calls, results)
                self.on_event(f"  → {call.name}({json.dumps(call.input)})")
                try:
                    r = await mcp.call_tool(call.name, call.input)
                    content, is_error = _to_blocks(r), bool(r.is_error)
                except Exception as e:  # bad arguments etc.
                    content, is_error = [{"type": "text", "text": f"Tool error: {e}"}], True
                if content and content[0].get("type") == "text":
                    self.on_event("    " + content[0]["text"][:160])
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
            if self.cancelled:
                return self._finish_cancelled(calls, results)
            self.messages.append({"role": "user", "content": results})
            self._prune_images()
        return "I stopped because that took too many steps. Tell me how you'd like to continue."


async def main() -> None:
    agent = ArmAgent()
    print("SO-101 agent. Type a command (e.g. 'pick up the red block and put it in the bowl'). Ctrl-D to quit.")
    try:
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
            if text:
                print("arm>", await agent.ask(text))
    finally:
        server.skills.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
