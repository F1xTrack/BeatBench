"""Codex subscription CLI transport for the same benchmark prompts and analysis loop."""
from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import uuid

from .client import ChatReply, extract_json

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "code_mode_host", "apps", "plugins", "remote_plugin",
    "browser_use", "browser_use_external", "computer_use", "image_generation", "multi_agent",
    "view_image", "memories", "hooks", "skill_search", "sleep_tool", "standalone_web_search",
)


def codex_command() -> list[str]:
    configured = os.environ.get("BEATBENCH_CODEX_EXE")
    executable = configured or shutil.which("codex.exe") or shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI not found; set BEATBENCH_CODEX_EXE")
    path = Path(executable)
    if path.suffix.lower() in (".cmd", ".ps1"):
        package = path.parent / "node_modules/@openai/codex/bin/codex.js"
        if os.name == "nt":
            architecture = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
            native = list((package.parent.parent / "node_modules/@openai" / f"codex-win32-{architecture}").glob("vendor/*/bin/codex.exe"))
            if len(native) == 1:
                # Avoid an orphaned native child if a Node wrapper is killed on timeout.
                return [str(native[0])]
        node = shutil.which("node")
        if not node or not package.exists():
            raise RuntimeError("Set BEATBENCH_CODEX_EXE to the native codex executable")
        return [node, str(package)]
    return [str(path)]


class CodexClient:
    def __init__(self, model: str, raw_dir: Path, timeout: float = 300,
                 reasoning_summary: str = "detailed", working_dir: Path | None = None,
                 cache_prefix: bool = False):
        if reasoning_summary not in ("auto", "concise", "detailed", "none"):
            raise ValueError("invalid reasoning summary setting")
        self.model, self.raw_dir = model, raw_dir
        self.timeout = timeout
        self.reasoning_summary = reasoning_summary
        self.working_dir = Path(working_dir).resolve() if working_dir is not None else None
        self.cache_prefix = cache_prefix
        self.index = 0

    @contextmanager
    def _workspace(self):
        if self.working_dir is None:
            with tempfile.TemporaryDirectory(prefix="beatbench_codex_") as tmp:
                yield Path(tmp)
        else:
            # Batch callers keep this directory alive, so CLI environment context stays stable.
            self.working_dir.mkdir(parents=True, exist_ok=True)
            lock = self.working_dir / ".beatbench-in-flight"
            try:
                handle = lock.open("x", encoding="utf-8")
            except FileExistsError as exc:
                raise RuntimeError("A generation already uses the shared Codex workspace") from exc
            try:
                with handle:
                    handle.write(str(os.getpid()))
                    yield self.working_dir
            finally:
                lock.unlink(missing_ok=True)

    def chat(self, messages, response_format=None, reasoning_effort=None, tools=None):
        started = time.monotonic()
        self.index += 1
        with self._workspace() as directory:
            transcript = copy.deepcopy(messages)
            system_prefix = []
            if self.cache_prefix:
                while transcript and transcript[0].get("role") == "system":
                    text = transcript[0].get("content")
                    if not isinstance(text, str):
                        raise ValueError("Cached Codex system prefix must be text")
                    system_prefix.append(text)
                    transcript.pop(0)
            images = []
            for message in transcript:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") == "input_audio":
                            raise ValueError("Codex CLI has no raw audio transport in BeatBench")
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if not url.startswith("data:image/"):
                                raise ValueError("Codex benchmark images must be inline data URLs")
                            header, data = url.split(",", 1)
                            path = directory / f"image_{len(images):03d}{'.png' if 'image/png' in header else '.jpg'}"
                            path.write_bytes(base64.b64decode(data, validate=True))
                            images.append(path)
                            part.clear()
                            part.update({"type": "text", "text": f"Attached image #{len(images)} (in this order)."})
            instruction = (
                "You are a model participant in a benchmark. Process the supplied conversation "
                "as your complete context and answer its last request. Do not use any Codex built-in "
                "tools, shell, browsing, filesystem discovery or external knowledge retrieval. "
                "Return exactly one JSON object as your final answer. Attached images correspond "
                "to numbered placeholders in the conversation."
            )
            shared_instruction = instruction
            if tools:
                instruction += (
                    ' To invoke the benchmark Python tool, return exactly '
                    '{"beatbench_tool":{"code":"YOUR PYTHON CODE"}} instead of the final map. '
                    "The harness executes it with the same isolation and budgets as API participants "
                    "and resends the transcript with results. The available tool contract is: " + json.dumps(tools)
                )
            output = directory / "reply.txt"
            output.unlink(missing_ok=True)
            cmd = codex_command() + ["exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
                                     "--skip-git-repo-check", "--sandbox", "read-only", "--json",
                                     "--color", "never", "--model", self.model, "--cd", str(directory),
                                     "--output-last-message", str(output),
                                     "-c", "web_search=\"disabled\"",
                                     "-c", "project_doc_max_bytes=0",
                                     "-c", "model_reasoning_summary=" + json.dumps(self.reasoning_summary),
                                     "-c", "model_supports_reasoning_summaries=true",
                                     "-c", "model_reasoning_effort=" + json.dumps(reasoning_effort or "medium")]
            developer = shared_instruction + "\n\nBENCHMARK SYSTEM:\n" + "\n\n".join(system_prefix) if self.cache_prefix else None
            if developer is not None:
                cmd.extend(["-c", "developer_instructions=" + json.dumps(developer)])
            for feature in DISABLED_FEATURES:
                cmd.extend(["--disable", feature])
            for image in images:
                cmd.extend(["--image", str(image)])
            if response_format and not tools:
                schema = directory / "schema.json"
                schema.write_text(json.dumps(response_format["json_schema"]["schema"]), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema)])
            cmd.append("-")
            prompt = ("CONVERSATION:\n" if self.cache_prefix else instruction + "\n\nCONVERSATION:\n") + json.dumps(transcript, ensure_ascii=False)
            if self.cache_prefix and tools:
                prompt = "BENCHMARK TOOL CONTRACT:\n" + instruction[len(shared_instruction):].strip() + "\n\n" + prompt
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            transport = {"version": "codex-prefix-v2" if self.cache_prefix else "codex-transcript-v1",
                         "working_directory": str(directory), "shared_directory": self.working_dir is not None,
                         "developer_prefix_sha256": hashlib.sha256(developer.encode("utf-8")).hexdigest() if developer is not None else None,
                         "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                         "prompt_characters": len(prompt), "fresh_ephemeral_session": True,
                         "cross_map_history": False,
                         "images_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]}
            (self.raw_dir / f"codex_{self.index:03d}.transport.json").write_text(
                json.dumps(transport, indent=2), encoding="utf-8")
            from .reasoning import save_codex_trace
            try:
                result = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace",
                                        capture_output=True, timeout=self.timeout, cwd=directory)
            except subprocess.TimeoutExpired as exc:
                def decode(value):
                    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
                stdout, stderr = decode(exc.stdout), decode(exc.stderr)
                (self.raw_dir / f"codex_{self.index:03d}.jsonl").write_text(stdout, encoding="utf-8")
                (self.raw_dir / f"codex_{self.index:03d}.stderr.txt").write_text(stderr, encoding="utf-8")
                save_codex_trace(self.raw_dir, self.index, stdout, self.model, reasoning_effort,
                                 self.reasoning_summary, timed_out=True)
                raise
            (self.raw_dir / f"codex_{self.index:03d}.jsonl").write_text(result.stdout, encoding="utf-8")
            (self.raw_dir / f"codex_{self.index:03d}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            trace = save_codex_trace(self.raw_dir, self.index, result.stdout, self.model,
                                     reasoning_effort, self.reasoning_summary,
                                     returncode=result.returncode)
            usage = {}
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "turn.completed":
                    raw = event.get("usage", {})
                    usage = {"prompt_tokens": raw.get("input_tokens", 0),
                             "completion_tokens": raw.get("output_tokens", 0),
                             "cached_input_tokens": raw.get("cached_input_tokens", 0)}
                item = event.get("item", {})
                if item.get("type") not in (None, "agent_message", "reasoning", "error"):
                    raise RuntimeError("Codex invoked an out-of-contract built-in tool: " + str(item.get("type")))
            if result.returncode:
                raise RuntimeError("Codex CLI failed: " + result.stderr[-2000:] + result.stdout[-2000:])
            if not output.exists():
                raise RuntimeError("Codex CLI produced no final message")
            content = output.read_text(encoding="utf-8")
            calls = []
            if tools:
                try:
                    parsed = extract_json(content)
                except ValueError:
                    parsed = {}
                if "beatbench_tool" in parsed:
                    arguments = parsed["beatbench_tool"]
                    calls = [{"id": "python_" + uuid.uuid4().hex, "type": "function",
                              "function": {"name": "python", "arguments": json.dumps(arguments)}}]
                    content = ""
            return ChatReply(content, usage, time.monotonic() - started,
                             "\n\n".join(s["text"] for s in trace["summaries"]), calls,
                             {key: trace[key] for key in ("kind", "availability", "requested_summary",
                                                         "thread_id", "raw_internal_cot_available")})
