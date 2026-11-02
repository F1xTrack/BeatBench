"""Shared-input scenarios: representations, continuations, isolated tools and CLI transport."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beatbench.runner import pipeline, prompts
from beatbench.runner.client import ChatReply
from beatbench.runner.codex_client import CodexClient
from beatbench.runner.representations import build_spectrograms, stage_signal
from beatbench.runner.tool_client import ToolClient
from beatbench.types import TrackMeta


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "test.wav"
    sr = 44100
    y = np.zeros(sr * 8, dtype=np.float32)
    for sec in (.5, 1.5, 3.5, 6.5, 7.5):
        start = int(sec * sr)
        y[start:start + 441] = .5 * np.sin(np.arange(441) * 2 * np.pi * 440 / sr)
    sf.write(path, y, sr)
    return path


def test_deterministic_spectrogram_covers_signal_without_audio_or_annotations(audio, tmp_path):
    staged, signal = stage_signal(audio, tmp_path / "signal")
    parts, manifest = build_spectrograms(staged, tmp_path / "a")
    _, second = build_spectrograms(staged, tmp_path / "b")
    assert signal["duration_seconds"] == 8
    assert manifest == second
    assert [(r["start_seconds"], r["end_seconds"]) for r in manifest["images"]] == [(0,8),(0,4),(4,8)]
    assert manifest["settings"]["sample_rate"] == 44100
    assert manifest["settings"]["version"] == 2
    assert not any(p["type"] == "input_audio" for p in parts)
    assert not {"onsets", "bpm", "beats", "notes"} & manifest.keys()
    from PIL import Image
    with Image.open(tmp_path / "a/segment_000.png") as im:
        assert im.size == (1400, 480)


def test_all_stages_use_the_same_supplied_signal_without_loading_audio(monkeypatch, audio):
    def forbidden(*args, **kwargs):
        raise AssertionError("native audio leaked into common condition")
    monkeypatch.setattr(prompts, "_audio_part", forbidden)
    parts = [{"type": "text", "text": "shared signal"}]
    meta = TrackMeta(title="T", bpm=120)
    info = {"beatsPerMinute": 120}
    stages = [prompts.build_info_messages(meta, None, audio, signal_parts=parts),
              prompts.build_difficulty_messages(meta, info, "ExpertPlus", None, audio, signal_parts=parts),
              prompts.build_continuation_messages(meta, info, "ExpertPlus", {"colorNotes": []}, None, audio, signal_parts=parts)]
    for messages in stages:
        assert messages[1]["content"][0] == parts[0]
        assert not any(p.get("type") == "input_audio" for p in messages[1]["content"])


def test_common_mode_pipeline_continues_scores_and_preserves_previous_runs(monkeypatch, tmp_path, audio):
    monkeypatch.chdir(tmp_path)
    previous = Path("results/example/test/raw/call_1.json")
    previous.parent.mkdir(parents=True)
    previous.write_text("keep")
    info = {"version":"2.2.0", "songName":"T", "songAuthorName":"A", "beatsPerMinute":120,
            "songFilename":"song.wav", "coverImageFilename":"cover.png",
            "difficultyBeatmapSets":[{"_beatmapCharacteristicName":"Standard", "_difficultyBeatmaps":[
                {"_difficulty":"ExpertPlus", "_beatmapFilename":"ExpertPlus.dat"}]}]}
    dat = {"version":"3.3.0", "colorNotes":[{"b":1,"x":0,"y":0,"c":0,"d":1}],"basicBeatmapEvents":[]}
    tail = {"colorNotes":[{"b":15,"x":1,"y":0,"c":1,"d":0}], "basicBeatmapEvents":[]}
    class Client:
        def __init__(self,*args,**kwargs):
            assert kwargs["timeout"] == 900
            self.replies = iter([{"info":info}, dat, tail])
        def chat(self,messages,**kwargs):
            assert not any(p.get("type")=="input_audio" for m in messages for p in (m.get("content") if isinstance(m.get("content"),list) else []))
            return ChatReply(json.dumps(next(self.replies)), {}, .01)
    monkeypatch.setattr(pipeline,"ModelClient",Client)
    cfg={"weights":{"coverage":1}}
    out=pipeline.evaluate_track(audio,"ep","key","example",TrackMeta(title="T",bpm=120),cfg,input_mode="spectrogram",v2=True,model_call_timeout=900)
    result=json.loads((out/"results.json").read_text())
    manifest=json.loads((out/"experiment.json").read_text())
    assert result["status"]=="ok" and result["gate_passed"]
    assert manifest["input_mode"]=="spectrogram" and manifest["status"]=="ok"
    assert manifest["model_call_timeout_seconds"] == 900
    assert (out/"test.zip").exists() and (out/"test-v2.zip").exists()
    assert previous.read_text()=="keep"
    assert json.loads((out/"raw/call_3.json").read_text())["kind"]=="difficulty_continue"


def test_tool_loop_preserves_tool_protocol_usage_and_run_wide_budget(tmp_path):
    seen=[]
    class Client:
        def chat(self,messages,**kwargs):
            seen.append(messages)
            if len(seen)==1:
                return ChatReply("",{"prompt_tokens":10,"completion_tokens":2,"cost_usd":.01},.01,tool_calls=[
                    {"id":"p1","type":"function","function":{"name":"python","arguments":'{"code":"print(42)"}'}}])
            return ChatReply('{"answer":42}',{"prompt_tokens":20,"completion_tokens":3,"cost_usd":.02},.01)
    class Sandbox:
        calls=0
        def execute(self,code):
            self.calls+=1
            assert code=="print(42)"
            return {"ok":True,"stdout":"42","images":[]}
    sandbox=Sandbox()
    c=ToolClient(Client(),sandbox,tmp_path)
    reply=c.chat([{"role":"user","content":"analyze"}])
    assert reply.content=='{"answer":42}' and reply.usage=={"prompt_tokens":30,"completion_tokens":5,"cost_usd":.03}
    assert seen[1][-1]["role"]=="tool" and seen[1][-1]["tool_call_id"]=="p1"
    assert sandbox.calls==1 and c.api_calls==2
    c.max_api_calls=2
    with pytest.raises(RuntimeError,match="budget"):
        c.chat([])


def test_codex_transports_images_and_rejects_builtin_tool_activity(monkeypatch,tmp_path):
    from beatbench.runner import codex_client
    from beatbench.runner.representations import image_part
    image=tmp_path/"image.png"
    from PIL import Image
    Image.new("RGB",(12,12)).save(image)
    monkeypatch.setattr(codex_client,"codex_command",lambda:["codex"])
    def run(cmd,**kwargs):
        assert kwargs["timeout"] == 900
        assert "--ignore-user-config" in cmd and "--ephemeral" in cmd
        assert "shell_tool" in cmd and "--image" in cmd
        assert "data:image" not in kwargs["input"]
        out=Path(cmd[cmd.index("--output-last-message")+1])
        out.write_text('{"beatbench_tool":{"code":"print(1)"}}')
        return type("Result",(),{"returncode":0,"stderr":"","stdout":json.dumps({"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}})})()
    monkeypatch.setattr(codex_client.subprocess,"run",run)
    reply=CodexClient("gpt-5.6-luna",tmp_path,timeout=900).chat([{"role":"user","content":[image_part(image)]}],tools=[{}])
    assert reply.tool_calls[0]["function"]["name"]=="python"
    assert reply.usage["prompt_tokens"]==5
    def bad_run(cmd,**kwargs):
        return type("Result",(),{"returncode":0,"stderr":"","stdout":'{"item":{"type":"command_execution"}}'})()
    monkeypatch.setattr(codex_client.subprocess,"run",bad_run)
    with pytest.raises(RuntimeError,match="out-of-contract"):
        CodexClient("gpt-5.6-luna",tmp_path).chat([])


def test_api_surfaces_http_200_provider_error_and_bounds_requests(monkeypatch):
    from beatbench.runner import client as module
    captured = {}
    class Completions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": None, "model_extra": {
                "error": {"message": "upstream quota exhausted", "code": 502}}})()
    class API:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = type("Chat", (), {"completions": Completions()})()
    monkeypatch.setattr(module, "OpenAI", API)
    client = module.ModelClient("http://ep", "key", "free", timeout=300)
    with pytest.raises(RuntimeError, match="upstream quota exhausted"):
        client.chat([])
    assert captured["timeout"] == 300 and captured["max_retries"] == 0


@pytest.mark.parametrize("backend", ["codex", "local"])
def test_subscription_and_local_cli_need_no_api_key(monkeypatch, audio, tmp_path, backend):
    import beatbench.cli as module
    from typer.testing import CliRunner
    monkeypatch.delenv("BEATBENCH_KEY", raising=False)
    seen = {}
    def evaluate(*args, **kwargs):
        seen.update(kwargs)
        assert args[2] == ""
        out = tmp_path / "output"
        out.mkdir()
        (out / "results.json").write_text('{"status":"ok","final":0.5}')
        return out
    monkeypatch.setattr(module, "evaluate_track", evaluate)
    result = CliRunner().invoke(module.app, ["evaluate", str(audio), "--backend", backend,
                               "--endpoint", "http://127.0.0.1:8097/v1",
                               "--model", "gpt-5.6-luna", "--input-mode", "python"])
    assert result.exit_code == 0
    assert seen["backend"] == backend and seen["input_mode"] == "python"


def test_status_reader_ignores_older_failure_file(tmp_path):
    import os
    from beatbench.cli import _load_status
    failure = tmp_path / "status.json"
    success = tmp_path / "results.json"
    failure.write_text('{"status":"failed_api"}')
    success.write_text('{"status":"ok"}')
    os.utime(failure, ns=(100,100))
    os.utime(success, ns=(200,200))
    assert _load_status(tmp_path)["status"] == "ok"


def test_sandbox_freezes_worker_and_stages_only_signal(audio):
    from beatbench.runner.sandbox import PythonSandbox
    with_source = PythonSandbox(audio, None)
    try:
        source = Path(__file__).parents[1] / "src/beatbench/runner/sandbox_worker.py"
        assert with_source.worker_path != source
        assert with_source.worker_path.read_bytes() == source.read_bytes()
        assert [p.name for p in with_source.input_dir.iterdir()] == ["audio.wav"]
        assert with_source.worker_path.parent == with_source.input_dir.parent
    finally:
        with_source.close()


def test_local_backend_rejects_external_endpoint_before_creating_run(audio, tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        pipeline.evaluate_track(audio, "https://openrouter.ai/api/v1", "", "gemma",
                                TrackMeta(title="T"), {}, backend="local", input_mode="python",
                                output_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()
