import json
import hashlib
from pathlib import Path
import subprocess

import pytest

from beatbench.runner.reasoning import parse_codex_trace, save_codex_trace, analyze_reasoning
from beatbench.runner.codex_client import CodexClient


def test_updates_are_deduplicated_and_summary_is_not_internal_cot():
    lines = [
        {"type": "thread.started", "thread_id": "test-thread"},
        {"type": "item.updated", "item": {"type": "reasoning", "id": "r1", "text": "BPM"}},
        {"type": "item.completed", "item": {"type": "reasoning", "id": "r1", "text": "BPM and parity"}},
        {"type": "turn.completed", "usage": {"output_tokens": 90}},
    ]
    trace = parse_codex_trace("\n".join(json.dumps(x) for x in lines))
    assert trace["availability"] == "captured"
    assert [x["text"] for x in trace["summaries"]] == ["BPM and parity"]
    assert trace["raw_internal_cot_available"] is False


def test_tokens_without_summary_remain_explicitly_missing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    text = json.dumps({"type": "turn.completed", "usage": {"output_tokens": 4000}})
    (raw / "codex_001.jsonl").write_text(text)
    save_codex_trace(raw, 1, text, "gpt-6-luna", "high", "detailed")
    result, path = analyze_reasoning(tmp_path)
    assert result["capture_status"] == "not_returned"
    assert result["summary_items"] == 0
    assert path.exists()
    assert not (raw / "codex_001.reasoning.md").exists()


def test_timeout_preserves_partial_summary(monkeypatch, tmp_path):
    from beatbench.runner import codex_client
    monkeypatch.setattr(codex_client, "codex_command", lambda: ["codex"])
    text = json.dumps({"type": "item.completed", "item": {"type": "reasoning", "id": "r", "text": "Check timing"}})
    def fail(cmd, **kwargs):
        assert 'model_reasoning_summary="detailed"' in cmd
        assert "model_supports_reasoning_summaries=true" in cmd
        raise subprocess.TimeoutExpired(cmd, 1, output=text.encode(), stderr=b"partial")
    monkeypatch.setattr(codex_client.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        CodexClient("gpt-6-luna", tmp_path, timeout=1).chat([], reasoning_effort="high")
    data = json.loads((tmp_path / "codex_001.reasoning.json").read_text())
    assert data["timed_out"] and data["availability"] == "captured"
    assert (tmp_path / "codex_001.jsonl").read_text() == text
    assert data["source_sha256"] == hashlib.sha256((tmp_path / "codex_001.jsonl").read_bytes()).hexdigest()


def test_reasoning_does_not_enter_next_model_context(monkeypatch, tmp_path):
    from beatbench.runner import codex_client
    monkeypatch.setattr(codex_client, "codex_command", lambda: ["codex"])
    def run(cmd, **kwargs):
        assert "prior secret summary" not in kwargs["input"]
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text('{"value": 1}')
        events = [{"type": "item.completed", "item": {"type": "reasoning", "id": "r", "text": "prior secret summary"}}]
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": "\n".join(map(json.dumps, events))})()
    monkeypatch.setattr(codex_client.subprocess, "run", run)
    client = CodexClient("gpt-6-luna", tmp_path)
    reply = client.chat([{"role": "user", "content": "map"}])
    assert reply.reasoning_metadata["kind"] == "provider_reasoning_summary"
    client.chat([{"role": "user", "content": "next map"}])


def test_truncated_trace_and_malformed_item_keep_previous_summary():
    events = [
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "id": "r", "text": "Check rhythm"}}),
        json.dumps({"type": "item.updated", "item": "invalid"}),
        '{"type":"unfinished',
    ]
    trace = parse_codex_trace("\n".join(events))
    assert trace["malformed_lines"] == 2
    assert trace["availability"] == "captured"
    assert trace["summaries"][0]["text"] == "Check rhythm"


def test_analysis_distinguishes_heading_only_text_and_keeps_capture_settings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    events = [
        {"type": "item.completed", "item": {"type": "reasoning", "id": "r1", "text": "**Estimating BPM**"}},
        {"type": "item.completed", "item": {"type": "reasoning", "id": "r2", "text": "**Comparing sections**\nThe pattern changes at the chorus."}},
    ]
    text = "\n".join(map(json.dumps, events))
    path = raw / "codex_001.jsonl"
    path.write_text(text, encoding="utf-8")
    original = path.read_bytes()
    save_codex_trace(raw, 1, text, "gpt-6-luna", "high", "detailed", timed_out=True)
    (raw / "python_001.json").write_text('{"unfinished', encoding="utf-8")
    result, output = analyze_reasoning(tmp_path)
    assert result["heading_only_summary_items"] == 1
    assert result["summary_body_characters"] == len("The pattern changes at the chorus.")
    assert result["call_capture"][0]["requested_summary"] == "detailed"
    assert result["call_capture"][0]["timed_out"] is True
    assert result["analysis_warnings"] and result["topic_mentions"]["phrasing"] == 1
    assert path.read_bytes() == original
    assert output.is_file()


def test_source_hash_uses_original_file_bytes_with_windows_newlines(tmp_path):
    text = '{"type":"turn.completed","usage":{}}\n'
    path = tmp_path / "codex_001.jsonl"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    trace = save_codex_trace(tmp_path, 1, text, "gpt-6-luna", "high", "detailed")
    assert trace["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert trace["normalized_stdout_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert trace["source_sha256"] != trace["normalized_stdout_sha256"]
