"""Tests for prompt builders in beatbench.runner.prompts."""
import base64
import json

import numpy as np
import pytest
import soundfile as sf

from beatbench.runner import prompts
from beatbench.types import TrackMeta


@pytest.fixture()
def wav_file(tmp_path):
    path = tmp_path / "song.wav"
    sf.write(path, np.zeros(1000, dtype="float32"), 44100, format="WAV")
    return path


def meta(bpm=None, nps=None):
    return TrackMeta(
        title="Neon Drift",
        author="Synthwave Kid",
        bpm=bpm,
        nps=nps,
        difficulties=["ExpertPlus", "Expert"],
    )


def input_audio_parts(messages):
    parts = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            parts.extend(p for p in msg["content"] if p.get("type") == "input_audio")
    return parts


def text_parts(messages):
    out = ""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for p in msg["content"]:
                if p.get("type") == "text":
                    out += p["text"]
    return out


def test_info_messages_structure(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    audios = input_audio_parts(messages)
    assert len(audios) == 1
    assert audios[0]["input_audio"]["format"] == "wav"
    decoded = base64.b64decode(audios[0]["input_audio"]["data"])
    assert decoded[:4] == wav_file.read_bytes()[:4]
    text = text_parts(messages)
    assert "Neon Drift" in text
    assert "Synthwave Kid" in text
    assert 'info' in text
    assert '"difficultyBeatmapSets": [' in text
    assert '"_beatmapCharacteristicName": "Standard"' in text


def test_info_messages_skeleton_is_valid_json(wav_file):
    text = text_parts(prompts.build_info_messages(meta(), None, wav_file))
    start = text.index('{"info": {"version": "2.2.0"')
    json.loads(text[start:])


def test_system_prompt_keywords(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    system = messages[0]["content"]
    for token in ("JSON", "colorNotes", "sliders", "burstSliders", "EACH hand", "Lighting", "sceneColors"):
        assert token in system


def test_system_prompt_requests_contextual_spawn_settings(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    system = messages[0]["content"]
    assert "_noteJumpMovementSpeed" in system
    assert "_noteJumpStartBeatOffset" in system
    assert "readable approach" in system


def test_context_text_mentions_cover_art_palette():
    text = prompts._context_text(meta())
    assert "cover art palette" in text
    assert "saber colors" in text.lower()


def test_info_messages_require_saber_colors(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    text = text_parts(messages)
    assert '"saberColors"' in text
    assert '"left"' in text
    assert '"right"' in text
    skeleton_start = text.index('{"info": {"version": "2.2.0"')
    skeleton_text = text[skeleton_start:]
    assert '"saberColors"' in skeleton_text
    parsed = json.loads(skeleton_text)
    colors = parsed["saberColors"]
    assert "sceneColors" in parsed
    for side in ("left", "right"):
        assert len(colors[side]) == 3
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in colors[side])


def test_bpm_present_only_if_set(wav_file):
    without = text_parts(prompts.build_info_messages(meta(), None, wav_file))
    with_bpm = text_parts(prompts.build_info_messages(meta(bpm=128.0), None, wav_file))
    assert "Desired BPM" not in without
    assert "Desired BPM: 128.0" in with_bpm


def test_difficulty_messages(wav_file):
    info_json = {"version": "2.2.0", "songName": "Neon Drift"}
    messages = prompts.build_difficulty_messages(meta(), info_json, "ExpertPlus", None, wav_file)
    assert len(input_audio_parts(messages)) == 1
    text = text_parts(messages)
    assert "ExpertPlus" in text
    assert '"songName"' in text
    assert '"2.2.0"' in text
    assert '"colorNotes"' in text
    assert "NOT Info.dat" in text
    skeleton_start = text.index('{"version": "3.3.0"')
    json.loads(text[skeleton_start:])


def test_cover_part_included(tmp_path, wav_file):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8fakejpeg")
    messages = prompts.build_info_messages(meta(), cover, wav_file)
    urls = [
        p for p in messages[1]["content"]
        if p.get("type") == "image_url"
    ]
    assert len(urls) == 1
    assert urls[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    payload = urls[0]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(payload) == cover.read_bytes()


def test_no_cover_no_image_part(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    assert all(p.get("type") != "image_url" for p in messages[1]["content"])


def test_ogg_triggers_conversion(monkeypatch, tmp_path, wav_file):
    ogg = tmp_path / "song.ogg"
    ogg.write_bytes(wav_file.read_bytes())
    fake_mp3 = tmp_path / "fake.mp3"
    fake_mp3.write_bytes(b"ID3fake")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        target = cmd[-1]
        with open(target, "wb") as fh:
            fh.write(fake_mp3.read_bytes())

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(prompts.subprocess, "run", fake_run)
    messages = prompts.build_info_messages(meta(), None, ogg)
    audios = input_audio_parts(messages)
    assert len(audios) == 1
    assert audios[0]["input_audio"]["format"] == "mp3"
    assert base64.b64decode(audios[0]["input_audio"]["data"]) == b"ID3fake"
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "-t" in cmd
    assert float(cmd[cmd.index("-t") + 1]) == prompts.MAX_AUDIO_SECONDS


def test_long_wav_routed_to_conversion(monkeypatch, tmp_path):
    long_wav = tmp_path / "long.wav"
    sf.write(long_wav, np.zeros(241 * 8000, dtype="float32"), 8000, format="WAV")
    fake_mp3 = tmp_path / "fake.mp3"
    fake_mp3.write_bytes(b"ID3trimmed")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        target = cmd[-1]
        with open(target, "wb") as fh:
            fh.write(fake_mp3.read_bytes())

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(prompts.subprocess, "run", fake_run)
    messages = prompts.build_info_messages(meta(), None, long_wav)
    audios = input_audio_parts(messages)
    assert len(audios) == 1
    assert audios[0]["input_audio"]["format"] == "mp3"
    assert base64.b64decode(audios[0]["input_audio"]["data"]) == b"ID3trimmed"
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-t" in cmd


def test_short_wav_stays_passthrough(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file)
    audios = input_audio_parts(messages)
    assert audios[0]["input_audio"]["format"] == "wav"


def test_ffmpeg_failure_raises(monkeypatch, tmp_path, wav_file):
    bad = tmp_path / "song.flac"
    bad.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        class Proc:
            returncode = 1
            stderr = "boom: invalid data"

        return Proc()

    monkeypatch.setattr(prompts.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid data"):
        prompts.build_info_messages(meta(), None, bad)


def test_nps_line_present_when_set(wav_file):
    msgs = prompts.build_info_messages(meta(nps=2.5), None, wav_file)
    text = "".join(p.get("text", "") for p in msgs[1]["content"])
    assert "2.50 notes per second" in text


def test_nps_line_absent_when_none(wav_file):
    msgs = prompts.build_info_messages(meta(nps=None), None, wav_file)
    text = "".join(p.get("text", "") for p in msgs[1]["content"])
    assert "notes per second" not in text


def _capture_ffmpeg(monkeypatch, tmp_path):
    """Stub subprocess.run; returns dict capturing the ffmpeg argv."""
    fake_mp3 = tmp_path / "fake.mp3"
    fake_mp3.write_bytes(b"ID3fake")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        target = cmd[-1]
        with open(target, "wb") as fh:
            fh.write(fake_mp3.read_bytes())

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(prompts.subprocess, "run", fake_run)
    return captured


def _ogg_for_conversion(tmp_path, wav_file):
    ogg = tmp_path / "song.ogg"
    ogg.write_bytes(wav_file.read_bytes())
    return ogg


def test_convert_to_mp3_default_uses_mono_16k(monkeypatch, tmp_path, wav_file):
    ogg = _ogg_for_conversion(tmp_path, wav_file)
    captured = _capture_ffmpeg(monkeypatch, tmp_path)
    prompts.build_info_messages(meta(), None, ogg)
    cmd = captured["cmd"]
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"


def test_convert_to_mp3_fullrate_uses_stereo_44k1(monkeypatch, tmp_path, wav_file):
    ogg = _ogg_for_conversion(tmp_path, wav_file)
    captured = _capture_ffmpeg(monkeypatch, tmp_path)
    prompts.build_info_messages(meta(), None, ogg, audio_fullrate=True)
    cmd = captured["cmd"]
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-ar") + 1] == "44100"
    # trim cap unchanged in both modes
    assert float(cmd[cmd.index("-t") + 1]) == prompts.MAX_AUDIO_SECONDS


def test_difficulty_messages_fullrate_threads_through(
        monkeypatch, tmp_path, wav_file):
    ogg = _ogg_for_conversion(tmp_path, wav_file)
    captured = _capture_ffmpeg(monkeypatch, tmp_path)
    prompts.build_difficulty_messages(
        meta(), {"songName": "Neon Drift"}, "ExpertPlus", None, ogg,
        audio_fullrate=True)
    cmd = captured["cmd"]
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-ar") + 1] == "44100"


def test_difficulty_messages_default_keeps_legacy_args(
        monkeypatch, tmp_path, wav_file):
    ogg = _ogg_for_conversion(tmp_path, wav_file)
    captured = _capture_ffmpeg(monkeypatch, tmp_path)
    prompts.build_difficulty_messages(
        meta(), {"songName": "Neon Drift"}, "ExpertPlus", None, ogg)
    cmd = captured["cmd"]
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"


def test_fullrate_does_not_affect_passthrough(wav_file):
    messages = prompts.build_info_messages(meta(), None, wav_file,
                                           audio_fullrate=True)
    audios = input_audio_parts(messages)
    assert audios[0]["input_audio"]["format"] == "wav"
