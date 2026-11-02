"""Shared analysis tool loop with budgets covering every stage and retry of a run."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .client import ChatReply

PYTHON_TOOL = {"type": "function", "function": {
    "name": "python", "description": (
        "Analyze /input/audio.wav using Python. numpy, scipy, librosa, soundfile, matplotlib "
        "and pillow are installed. Network and host files are inaccessible. Each call starts "
        "fresh: variables/files do not persist. Print findings; save up to two plots as "
        "/work/*.png to see them. You choose every analysis; no BPM/onsets are precomputed."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"], "additionalProperties": False}}}


class ToolClient:
    def __init__(self, client, sandbox, raw_dir: Path):
        self.client, self.sandbox, self.raw_dir = client, sandbox, raw_dir
        self.api_calls = 0
        self.tool_index = 0
        self.max_api_calls = 36

    def chat(self, messages, response_format=None, reasoning_effort=None):
        conversation = list(messages)
        started = time.monotonic()
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        reasoning = []
        reasoning_sources = []
        while True:
            if self.api_calls >= self.max_api_calls:
                raise RuntimeError("run-wide model request budget exhausted")
            self.api_calls += 1
            reply = self.client.chat(conversation, response_format=response_format,
                                     reasoning_effort=reasoning_effort, tools=[PYTHON_TOOL])
            payload = {"messages": conversation, "reply_content": reply.content,
                       "tool_calls": reply.tool_calls, "usage": reply.usage,
                       "elapsed_s": reply.elapsed_s, "reasoning": reply.reasoning}
            payload["reasoning_metadata"] = reply.reasoning_metadata
            (self.raw_dir / f"model_step_{self.api_calls:03d}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += reply.usage.get(key, 0) or 0
            if "cached_input_tokens" in reply.usage:
                usage["cached_input_tokens"] = usage.get("cached_input_tokens", 0) + (reply.usage["cached_input_tokens"] or 0)
            if "cost_usd" in reply.usage:
                usage["cost_usd"] = usage.get("cost_usd", 0) + reply.usage["cost_usd"]
            if reply.reasoning:
                reasoning.append(reply.reasoning)
            if reply.reasoning_metadata:
                reasoning_sources.append(reply.reasoning_metadata)
            if not reply.tool_calls:
                return ChatReply(reply.content, usage, time.monotonic() - started,
                                 "\n\n".join(reasoning), reasoning_metadata={"sources": reasoning_sources,
                                     "kind": "provider_reasoning_summary" if reasoning_sources else "provider_text",
                                     "availability": "captured" if reasoning else "not_returned",
                                     "raw_internal_cot_available": False})
            conversation.append({"role": "assistant", "content": reply.content or None,
                                 "tool_calls": reply.tool_calls})
            for call in reply.tool_calls:
                self.tool_index += 1
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments", ""))
                    if function.get("name") != "python" or not isinstance(arguments, dict) or set(arguments) != {"code"}:
                        raise ValueError("only python(code) is available")
                    result = self.sandbox.execute(arguments["code"])
                except (ValueError, TypeError) as exc:
                    result = {"ok": False, "error": str(exc)}
                (self.raw_dir / f"python_{self.tool_index:03d}.json").write_text(
                    json.dumps({"request": call, "result": result}), encoding="utf-8")
                images = result.get("images", [])
                text_result = {key: value for key, value in result.items() if key != "images"}
                text_result["plots"] = [im["name"] for im in images]
                conversation.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(text_result)})
                if images:
                    conversation.append({"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + im["data"], "detail": "high"}}
                        for im in images]})
