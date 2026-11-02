"""Archive exposed reasoning summaries and observable actions, never infer hidden CoT."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

TOPICS = {
    "timing": r"\b(bpm|tempo|onsets?|timing|rhythm|beat tracking|offset)\b|ритм|темп",
    "movement": r"\b(parity|forehand|backhand|resets?|swings?|flow|wrist|hand alternation)\b|движени",
    "phrasing": r"\b(phrases?|sections?|motifs?|chorus|repetition|patterns?)\b|паттерн|фраз",
    "lighting": r"\b(light|lighting|brightness|chroma|environment)\b|освещ|свет",
    "palette": r"\b(portal|orange|blue|palette|colou?r)\b|портал|оранж|син",
    "validation": r"\b(validate|verify|schema|check|error|exception)\b|провер",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_codex_trace(stdout: str) -> dict:
    items = {}
    thread_id = None
    usage = {}
    errors = []
    malformed = 0
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        thread_id = event.get("thread_id", thread_id)
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        if event.get("type") in ("error", "turn.failed"):
            errors.append({"line": line_number, "error": event.get("message", event.get("error"))})
        item = event.get("item") or {}
        if not isinstance(item, dict):
            malformed += 1
            continue
        if item.get("type") == "reasoning":
            # Updated items are cumulative. Keep the last revision, not repeated copies.
            key = item.get("id") or f"line-{line_number}"
            text = item.get("text", "")
            if isinstance(text, str) and text.strip():
                items[key] = {"item_id": key, "event": event.get("type"),
                              "line": line_number, "text": text}
    summaries = list(items.values())
    return {"thread_id": thread_id, "usage": usage, "summaries": summaries,
            "summary_characters": sum(len(s["text"]) for s in summaries),
            "errors": errors, "malformed_lines": malformed,
            "availability": "captured" if summaries else "not_returned",
            "kind": "provider_reasoning_summary", "raw_internal_cot_available": False}


def save_codex_trace(raw_dir: Path, index: int, stdout: str, model: str,
                     effort: str | None, summary_setting: str, *,
                     timed_out: bool = False, returncode: int | None = None) -> dict:
    trace = parse_codex_trace(stdout)
    source = raw_dir / f"codex_{index:03d}.jsonl"
    trace.update({"version": "reasoning-capture-v2", "model": model,
                  "effort": effort or "medium", "requested_summary": summary_setting,
                  "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "timed_out": timed_out, "returncode": returncode,
                  "source": source.name,
                  "source_sha256": _sha(source) if source.exists() else None,
                  "normalized_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest()})
    (raw_dir / f"codex_{index:03d}.reasoning.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    if trace["summaries"]:
        text = "# Exposed reasoning summaries (not full internal CoT)\n\n" + "\n\n".join(
            f"## {s['item_id']}\n\n{s['text']}" for s in trace["summaries"])
        (raw_dir / f"codex_{index:03d}.reasoning.md").write_text(text, encoding="utf-8")
    return trace


def _body_characters(text: str) -> int:
    """Count text outside standalone Markdown headings; not a reasoning-depth rating."""
    return sum(len(line.strip()) for line in text.splitlines()
               if line.strip() and not re.fullmatch(r"\*\*[^*]+\*\*|#{1,6}\s+.+", line.strip()))


def analyze_reasoning(run_dir: Path, output_root: Path | None = None) -> tuple[dict, Path]:
    """Deterministic evidence index. Topic mentions are not proof of sound decisions."""
    raw = run_dir / "raw"
    warnings = []

    def read_record(path):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Expected a JSON object")
            return record
        except (OSError, ValueError) as exc:
            warnings.append({"source": str(path.resolve()), "error": str(exc)})
            return {}

    traces = []
    for path in sorted(raw.glob("codex_*.jsonl")):
        trace = parse_codex_trace(path.read_text(encoding="utf-8"))
        trace.update({"source": str(path.resolve()), "sha256": _sha(path)})
        sidecar = path.with_suffix(".reasoning.json")
        trace["capture_settings"] = read_record(sidecar) if sidecar.exists() else {}
        recorded_hash = trace["capture_settings"].get("source_sha256")
        if recorded_hash and recorded_hash != trace["sha256"]:
            warnings.append({"source": str(sidecar.resolve()),
                             "error": "Recorded source hash differs from file bytes; use freshly computed trace SHA256. Capture v1 hashed normalized stdout before Windows newline conversion."})
        traces.append(trace)
    evidence = []
    for trace in traces:
        for summary in trace["summaries"]:
            evidence.append({"source": trace["source"], "line": summary["line"],
                             "item_id": summary["item_id"], "kind": trace["kind"],
                             "text": summary["text"],
                             "topic_mentions": [name for name, pattern in TOPICS.items()
                                                if re.search(pattern, summary["text"], re.I)]})
    actions = []
    for path in sorted(raw.glob("python_*.json")):
        data = read_record(path)
        if not data:
            continue
        result = data.get("result", {})
        request = data.get("request", {})
        function = request.get("function", {}) if isinstance(request, dict) else None
        if not isinstance(result, dict) or not isinstance(function, dict):
            warnings.append({"source": str(path.resolve()), "error": "Malformed Python action record"})
            continue
        arguments = function.get("arguments", "{}")
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
            code = decoded.get("code", "") if isinstance(decoded, dict) else ""
        except (ValueError, TypeError):
            code = ""
        actions.append({"source": str(path.resolve()), "sha256": _sha(path),
                        "ok": result.get("ok"), "error": result.get("error"),
                        "code": code, "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "elapsed_s": result.get("elapsed_s"),
                        "truncated": result.get("truncated"),
                        "plots": [im.get("name") for im in result.get("images", [])]})
    api_calls = []
    for path in sorted(raw.glob("call_*.json")):
        data = read_record(path)
        if not data:
            continue
        reasoning = data.get("reasoning", "")
        if isinstance(reasoning, (dict, list)):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        if not isinstance(reasoning, str):
            reasoning = ""
        usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
        api_calls.append({"source": str(path.resolve()), "sha256": _sha(path),
                          "kind": data.get("kind"), "usage": usage,
                          "reasoning_characters": len(reasoning),
                          "provider_reasoning_returned": bool(reasoning)})
        if reasoning:
            evidence.append({"source": str(path.resolve()), "line": 1,
                             "item_id": data.get("call_index"), "kind": data.get("kind", "api"),
                             "text": reasoning,
                             "topic_mentions": [name for name, pattern in TOPICS.items()
                                                if re.search(pattern, reasoning, re.I)]})
    result_path = run_dir / "results.json"
    result = read_record(result_path) if result_path.exists() else {}
    manifest_path = run_dir / "experiment.json"
    manifest = read_record(manifest_path) if manifest_path.exists() else {}
    counted = [t["usage"] for t in traces if isinstance(t["usage"].get("input_tokens"), int)]
    reported = [u for u in counted if isinstance(u.get("cached_input_tokens"), int)]
    input_tokens = sum(u["input_tokens"] for u in reported)
    cached_tokens = sum(u["cached_input_tokens"] for u in reported)
    writes = [u for u in counted if isinstance(u.get("cache_write_input_tokens"), int)]
    reasoning_counts = [u for u in counted if isinstance(u.get("reasoning_output_tokens"), int)]
    cache_usage = {"reported_calls": len(reported), "calls_without_cache_usage": len(traces) - len(reported),
                   "input_tokens": input_tokens, "cached_input_tokens": cached_tokens,
                   "uncached_input_tokens": input_tokens - cached_tokens,
                   "cached_fraction": cached_tokens / input_tokens if input_tokens else None,
                   "cache_write_input_tokens": sum(u["cache_write_input_tokens"] for u in writes) if writes else None,
                   "calls_without_cache_write_usage": len(traces) - len(writes),
                   "output_tokens": sum(u.get("output_tokens", 0) for u in counted),
                   "reasoning_output_tokens": sum(u["reasoning_output_tokens"] for u in reasoning_counts) if reasoning_counts else None,
                   "calls_without_reasoning_usage": len(traces) - len(reasoning_counts),
                   "interpretation": "Provider-reported CLI counters, including failures with final usage. Newer CLI traces may expose cache-write/reasoning counts; missing counters stay null or are counted separately. Reuse is not guaranteed."}
    api_usages = [call["usage"] for call in api_calls]

    def detail_value(usage, group, name):
        details = usage.get(group) or {}
        return details.get(name) if isinstance(details, dict) else None

    def total_usage(primary, secondary=None):
        values = []
        for usage in api_usages:
            value = usage.get(primary)
            if value is None and secondary:
                value = detail_value(usage, *secondary)
            if isinstance(value, (int, float)):
                values.append(value)
        return sum(values) if values else None

    openrouter_usage = {
        "reported_calls": len(api_calls),
        "calls_with_reasoning": sum(call["provider_reasoning_returned"] for call in api_calls),
        "input_tokens": total_usage("input_tokens") or total_usage("prompt_tokens"),
        "cached_input_tokens": total_usage("cached_input_tokens", ("prompt_tokens_details", "cached_tokens")),
        "output_tokens": total_usage("output_tokens") or total_usage("completion_tokens"),
        "reasoning_output_tokens": total_usage("reasoning_output_tokens", ("completion_tokens_details", "reasoning_tokens")),
        "cost_usd": total_usage("cost_usd"),
        "resolved_models": sorted({str(usage["resolved_model"]) for usage in api_usages
                                    if usage.get("resolved_model")}),
    }
    payload = {"version": "reasoning-analysis-v2", "source_run": str(run_dir.resolve()),
               "recorded_at": datetime.now(timezone.utc).isoformat(),
               "capture_status": "captured" if evidence else "not_returned",
               "raw_internal_cot_available": False, "cli_turns": len(traces),
               "summary_items": len(evidence), "summary_characters": sum(len(x["text"]) for x in evidence),
               "summary_body_characters": sum(_body_characters(x["text"]) for x in evidence),
               "heading_only_summary_items": sum(_body_characters(x["text"]) == 0 for x in evidence),
                "provider_reasoning_calls": openrouter_usage["calls_with_reasoning"],
                "provider_reasoning_characters": sum(call["reasoning_characters"] for call in api_calls),
               "run_conditions": {key: manifest.get(key) for key in
                                  ("model", "reasoning_effort", "input_mode", "generation_prompt")},
               "call_capture": [{"source": t["source"], "availability": t["availability"],
                                 "source_sha256": t["sha256"],
                                 "malformed_lines": t["malformed_lines"], "errors": t["errors"],
                                 **{key: t["capture_settings"].get(key) for key in
                                    ("model", "effort", "requested_summary", "timed_out", "returncode")}}
                                for t in traces],
               "analysis_warnings": warnings,
               "summaries": evidence, "observable_python_actions": actions,
               "failed_python_actions": sum(a["ok"] is False for a in actions),
               "topic_mentions": {name: sum(name in x["topic_mentions"] for x in evidence) for name in TOPICS},
               "cli_usage": [t["usage"] for t in traces],
               "cache_usage": cache_usage,
                "openrouter_calls": api_calls,
                "openrouter_usage": openrouter_usage,
               "map_assessment": result.get("assessment"),
                "interpretation": "Provider-returned reasoning is not guaranteed to be a complete internal CoT. Topic mentions and tool code do not prove reasoning quality or musical correctness. No model judge and no effect on map scores."}
    diagnostics = result.get("diagnostics") or {}
    payload["artifact_observations"] = {
        name: {"notes": data.get("note_count"), "lighting": data.get("lighting"),
               "palette": data.get("palette"), "issues": data.get("issues"),
               "scope": diagnostics.get("scope")}
        for name, data in diagnostics.get("difficulty_reviews", {}).items()}
    root = output_root or run_dir / "reasoning-analysis"
    if "history" in root.parts and ".docs" in root.parts:
        raise ValueError("Cannot write analysis to append-only goal history")
    directory = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    md = ["# Reasoning evidence", "", f"Capture: {payload['capture_status']}",
          f"Provider reasoning calls: {payload['provider_reasoning_calls']}; Python actions: {len(actions)}",
          "", payload["interpretation"], ""]
    md.extend([f"Heading-only summary items: {payload['heading_only_summary_items']}; "
               f"characters outside heading lines: {payload['summary_body_characters']}",
               "This counts formatting, not reasoning depth or understanding.", ""])
    md.extend(["## Cache usage", "", "```json", json.dumps(cache_usage, indent=2), "```", ""])
    md.extend(["## Topic mentions", "", "| Topic | Summary items mentioning it |", "|---|---:|"])
    md.extend(f"| {name} | {count} |" for name, count in payload["topic_mentions"].items())
    md.extend(["", "## Observable artifact", "", "```json",
               json.dumps(payload["artifact_observations"], ensure_ascii=False, indent=2), "```", ""])
    for x in evidence:
        md.extend([f"## {Path(x['source']).name}:{x['line']}", "", x["text"], ""])
    (directory / "analysis.md").write_text("\n".join(md), encoding="utf-8")
    return payload, path
