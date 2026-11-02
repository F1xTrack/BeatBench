"""OpenAI-compatible chat client and robust JSON extraction."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class ChatReply:
    content: str
    usage: dict
    elapsed_s: float
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_metadata: dict = field(default_factory=dict)


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return ""


def _split_reasoning(msg, content: str) -> tuple[str, str]:
    blocks = [b.strip() for b in _THINK_RE.findall(content) if b.strip()]
    if blocks:
        return "\n\n".join(blocks), _THINK_RE.sub("", content).strip()
    for attr in ("reasoning", "reasoning_content"):
        text = _stringify(getattr(msg, attr, None))
        if text.strip():
            return text, content.strip()
    extra = getattr(msg, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_details"):
        text = _stringify(extra.get(key))
        if text.strip():
            return text, content.strip()
    return "", content.strip()


class ModelClient:
    def __init__(self, endpoint: str, key: str, model: str, timeout: float | None = None,
                 live_dir: Path | None = None):
        self.model = model
        self.live_dir = live_dir
        self.call_index = 0
        options = {"timeout": timeout, "max_retries": 0} if timeout is not None else {}
        self._client = OpenAI(base_url=endpoint, api_key=key, **options)

    def chat(self, messages: list[dict], response_format: dict | None = None,
             reasoning_effort: str | None = None,
             tools: list[dict] | None = None) -> ChatReply:
        start = time.monotonic()
        kwargs = {"model": self.model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False
        live = None
        if self.live_dir is not None:
            self.call_index += 1
            live = _LiveStreamLog(self.live_dir, self.call_index, self.model)
            kwargs.update(stream=True, stream_options={"include_usage": True})
        try:
            resp = self._client.chat.completions.create(**kwargs) if live is None else None
            if live is not None:
                resp = self._collect_stream(kwargs, live)
        except Exception as exc:
            if live is not None:
                live.finish("error", f"{type(exc).__name__}: {exc}")
            raise
        elapsed = time.monotonic() - start
        if not resp.choices:
            error = (getattr(resp, "model_extra", None) or {}).get("error")
            raise RuntimeError("provider returned no completion choices: " + json.dumps(error, ensure_ascii=False))
        msg = resp.choices[0].message
        raw_usage = resp.usage
        usage = (raw_usage.model_dump(exclude_none=True)
                 if raw_usage is not None and hasattr(raw_usage, "model_dump") else {})
        if raw_usage is not None:
            def usage_field(obj, name):
                return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

            prompt_tokens = usage_field(raw_usage, "prompt_tokens")
            completion_tokens = usage_field(raw_usage, "completion_tokens")
            if prompt_tokens is not None:
                usage["prompt_tokens"] = prompt_tokens
                usage["input_tokens"] = prompt_tokens
            if completion_tokens is not None:
                usage["completion_tokens"] = completion_tokens
                usage["output_tokens"] = completion_tokens
            prompt_details = usage_field(raw_usage, "prompt_tokens_details")
            completion_details = usage_field(raw_usage, "completion_tokens_details")
            cached_tokens = usage_field(prompt_details, "cached_tokens")
            reasoning_tokens = usage_field(completion_details, "reasoning_tokens")
            if cached_tokens is not None:
                usage["cached_input_tokens"] = cached_tokens
            if reasoning_tokens is not None:
                usage["reasoning_output_tokens"] = reasoning_tokens
        if getattr(resp, "model", None) is not None:
            usage["resolved_model"] = resp.model
        if raw_usage is not None:
            extra_usage = getattr(raw_usage, "model_extra", None) or {}
            cost = getattr(raw_usage, "cost", None) or extra_usage.get("cost")
            if cost is not None:
                usage["cost_usd"] = float(cost)
        reasoning, content = _split_reasoning(msg, msg.content or "")
        message_extra = getattr(msg, "model_extra", None) or {}
        detail_items = message_extra.get("reasoning_details")
        reasoning_metadata = {
            "provider_reasoning_returned": bool(reasoning),
            "reasoning_field": ("reasoning_details" if detail_items else
                                "reasoning" if message_extra.get("reasoning") else None),
            "reasoning_detail_types": [item.get("type") for item in detail_items
                                       if isinstance(item, dict)] if isinstance(detail_items, list) else [],
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        }
        return ChatReply(content=content, usage=usage, elapsed_s=elapsed,
                         reasoning=reasoning,
                         tool_calls=[t.model_dump(exclude_none=True) for t in (getattr(msg, "tool_calls", None) or [])],
                         reasoning_metadata=reasoning_metadata)

    def _collect_stream(self, kwargs: dict, live: "_LiveStreamLog"):
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = model = finish_reason = None
        try:
            for chunk in self._client.chat.completions.create(**kwargs):
                model = getattr(chunk, "model", None) or model
                usage = getattr(chunk, "usage", None) or usage
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                text = _stringify(getattr(delta, "content", None))
                if text:
                    content.append(text)
                    live.add("answer", text)
                thought = ""
                for attr in ("reasoning", "reasoning_content"):
                    thought = _stringify(getattr(delta, attr, None))
                    if thought:
                        break
                if not thought:
                    extra = getattr(delta, "model_extra", None) or {}
                    thought = _stringify(extra.get("reasoning") or extra.get("reasoning_content"))
                if thought:
                    reasoning.append(thought)
                    live.add("reasoning", thought)
                for item in getattr(delta, "tool_calls", None) or []:
                    index = getattr(item, "index", 0)
                    call = tool_calls.setdefault(index, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    call["id"] = getattr(item, "id", None) or call["id"]
                    call["type"] = getattr(item, "type", None) or call["type"]
                    function = getattr(item, "function", None)
                    if function is not None:
                        call["function"]["name"] += getattr(function, "name", None) or ""
                        call["function"]["arguments"] += getattr(function, "arguments", None) or ""
            calls = [SimpleNamespace(model_dump=lambda exclude_none=True, data=data: data)
                     for _, data in sorted(tool_calls.items())]
            message = SimpleNamespace(content="".join(content), reasoning="".join(reasoning),
                                      tool_calls=calls, model_extra={})
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                usage=usage, model=model, model_extra={})
            live.finish("complete")
            return response
        except Exception:
            live.finish("error")
            raise


class _LiveStreamLog:
    """Write exposed provider stream deltas as they arrive; never stores prompts."""

    def __init__(self, raw_dir: Path, index: int, model: str):
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = raw_dir / f"live_call_{index:03d}.jsonl"
        self.current_path = raw_dir / "live_current.md"
        self.index, self.model = index, model
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.reasoning: list[str] = []
        self.answer: list[str] = []
        self.last_snapshot = 0.0
        self.finished = False
        self._event("started", status="streaming", model=model, started_at=self.started_at)
        self._snapshot("streaming")

    def _event(self, event: str, **fields) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    def add(self, channel: str, text: str) -> None:
        (self.reasoning if channel == "reasoning" else self.answer).append(text)
        self._event("delta", channel=channel, text=text)
        if time.monotonic() - self.last_snapshot >= 0.5:
            self._snapshot("streaming")

    def _snapshot(self, status: str, error: str | None = None) -> None:
        body = [f"# Live model response — call {self.index}", "", f"- Model: `{self.model}`",
                f"- Status: **{status}**", f"- Started: {self.started_at}",
                "", "## Provider-exposed reasoning", "", "".join(self.reasoning) or "_(пока пусто)_",
                "", "## Ответ модели", "", "".join(self.answer) or "_(пока пусто)_"]
        if error:
            body.extend(["", "## Error", "", error])
        self.current_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        self.last_snapshot = time.monotonic()

    def finish(self, status: str, error: str | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        self._event("finished", status=status, ended_at=datetime.now(timezone.utc).isoformat(), error=error)
        self._snapshot(status, error)


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output, tolerating fences and prose."""
    text = text.strip()
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip().removeprefix("json").strip()
            if chunk.startswith("{"):
                text = chunk
                break
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for end in range(start, len(text)):
        ch = text[end]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:end + 1])
    raise ValueError("unbalanced JSON")
