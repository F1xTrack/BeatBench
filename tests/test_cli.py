"""CLI tests: key resolution from env, exit codes per status."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import beatbench.cli as cli_mod
from beatbench.cli import app

runner = CliRunner()


@pytest.fixture()
def audio(tmp_path):
    wav = tmp_path / "track.wav"
    wav.write_bytes(b"RIFF")
    return wav


def stub_track(monkeypatch, tmp_path, status="ok"):
    captured = {}

    def fake_evaluate(audio_path, endpoint, key, model, meta, cfg,
                      constrained=False, v2=False, reasoning_effort=None,
                      audio_fullrate=False):
        captured["key"] = key
        captured["audio"] = audio_path
        captured["v2"] = v2
        captured["reasoning_effort"] = reasoning_effort
        captured["audio_fullrate"] = audio_fullrate
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        (out / "results.json").write_text(json.dumps({"status": status}),
                                          encoding="utf-8")
        return out

    monkeypatch.setattr(cli_mod, "evaluate_track", fake_evaluate)
    return captured


BASE_ARGS = ["--endpoint", "http://ep", "--model", "m1",
             "--title", "t", "--bpm", "120"]


@pytest.mark.parametrize("status", ["failed_info", "invalid", "partial"])
def test_nonzero_exit_for_failure_statuses(audio, tmp_path, monkeypatch, status):
    stub_track(monkeypatch, tmp_path, status=status)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k"])
    assert result.exit_code == 1


def test_zero_exit_for_ok(audio, tmp_path, monkeypatch):
    stub_track(monkeypatch, tmp_path, status="ok")
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k"])
    assert result.exit_code == 0


def test_v2_flag_forwarded_to_pipeline(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k", "--v2"])
    assert result.exit_code == 0
    assert captured["v2"] is True


def test_v2_flag_defaults_to_false(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS, "--key", "k"])
    assert captured["v2"] is False


def test_reasoning_effort_flag_forwarded_to_pipeline(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k", "--reasoning-effort", "high"])
    assert result.exit_code == 0
    assert captured["reasoning_effort"] == "high"


def test_reasoning_effort_defaults_to_none(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS, "--key", "k"])
    assert result.exit_code == 0
    assert captured["reasoning_effort"] is None


def test_audio_fullrate_flag_forwarded_to_pipeline(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k", "--audio-fullrate"])
    assert result.exit_code == 0
    assert captured["audio_fullrate"] is True


def test_audio_fullrate_defaults_to_false(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS,
                                 "--key", "k"])
    assert result.exit_code == 0
    assert captured["audio_fullrate"] is False


def test_key_from_flag_used(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    monkeypatch.setenv("BEATBENCH_KEY", "from-env")
    runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS, "--key", "from-flag"])
    assert captured["key"] == "from-flag"


def test_key_from_env_fallback(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    monkeypatch.setenv("BEATBENCH_KEY", "from-env")
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS])
    assert result.exit_code == 0
    assert captured["key"] == "from-env"


def test_missing_key_errors_clearly(audio, tmp_path, monkeypatch):
    stub_track(monkeypatch, tmp_path)
    monkeypatch.delenv("BEATBENCH_KEY", raising=False)
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS])
    assert result.exit_code != 0
    assert "BEATBENCH_KEY" in result.output


def test_env_key_takes_precedence_over_missing_flag_only(audio, tmp_path, monkeypatch):
    captured = stub_track(monkeypatch, tmp_path)
    monkeypatch.setenv("BEATBENCH_KEY", "")
    result = runner.invoke(app, ["evaluate", str(audio), *BASE_ARGS])
    assert result.exit_code != 0
