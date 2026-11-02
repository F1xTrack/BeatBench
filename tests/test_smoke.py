"""Integration smoke: full pipeline on FakeClient with synthetic audio and cover."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beatbench.config import DEFAULT_WEIGHTS
from beatbench.runner import pipeline
from beatbench.runner.client import ChatReply
from beatbench.types import TrackMeta

SR = 44100
DURATION_S = 20
BPM = 120

FAKE_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xd9"
)


def make_click_wav(path: Path) -> None:
    samples = np.zeros(SR * DURATION_S, dtype="float32")
    beat_interval = 60.0 / BPM
    click = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(200) / SR)
    for i in range(int(DURATION_S / beat_interval)):
        start = int(i * beat_interval * SR)
        end = min(start + len(click), len(samples))
        samples[start:end] += click[: end - start]
    sf.write(path, samples, SR, format="WAV")


class FakeClient:
    last_instance = None

    def __init__(self, endpoint, key, model, replies):
        self.endpoint, self.key, self.model = endpoint, key, model
        self.replies = list(replies)
        self.calls: list[tuple] = []
        FakeClient.last_instance = self

    @classmethod
    def factory(cls, replies):
        def make(endpoint, key, model):
            return cls(endpoint, key, model, replies)
        return make

    def chat(self, messages, response_format=None, reasoning_effort=None):
        self.calls.append((messages, response_format))
        content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return ChatReply(content=content, usage={"prompt_tokens": 5,
                                                 "completion_tokens": 7},
                         elapsed_s=0.02)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav = tmp_path / "track.wav"
    make_click_wav(wav)
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(FAKE_JPEG)
    return wav


def meta():
    return TrackMeta(title="Smoke Song", author="Smoke Artist", bpm=BPM,
                     difficulties=["ExpertPlus"])


def info_reply():
    info = {
        "version": "3.3.0",
        "songName": "Smoke Song",
        "songSubName": "",
        "songAuthorName": "Smoke Artist",
        "levelAuthorName": "Bench",
        "beatsPerMinute": BPM,
        "shuffle": 0,
        "shufflePeriod": 0.5,
        "previewStartTime": 12,
        "songFilename": "song.ogg",
        "coverImageFilename": "cover.jpg",
        "environmentName": "TriangleEnvironment",
        "difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": "ExpertPlus",
                "_difficultyRank": 9,
                "_beatmapFilename": "ExpertPlus.dat",
                "_noteJumpMovementSpeed": 18,
            }],
        }],
    }
    return json.dumps({"info": info})


def difficulty_reply():
    from tests.fixtures_helpers import FIXTURES
    diff_path = FIXTURES / "maps" / "valid" / "ExpertPlus.dat"
    return diff_path.read_text(encoding="utf-8")


def continuation_reply():
    return json.dumps({
        "colorNotes": [
            {"b": 40, "x": 1, "y": 0, "c": 0, "d": 1},
            {"b": 40.5, "x": 2, "y": 0, "c": 1, "d": 0},
        ],
        "bombNotes": [],
        "obstacles": [],
        "basicBeatmapEvents": [{"b": 41, "et": 0, "i": 1}],
    })


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_smoke_full_pipeline(env, monkeypatch):
    monkeypatch.setattr(
        pipeline, "ModelClient",
        FakeClient.factory([info_reply(), difficulty_reply(),
                            continuation_reply()]))
    out = pipeline.evaluate_track(env, "http://ep", "key", "m1", meta(),
                                  {"weights": DEFAULT_WEIGHTS})

    expected = Path("results") / "m1" / "track"
    assert out == expected
    assert out.is_dir() and (out / "raw").is_dir()

    calls = sorted((out / "raw").glob("call_*.json"))
    assert [load_json(c)["kind"] for c in calls] == [
        "info", "difficulty", "difficulty_continue"]
    for c in calls:
        payload = load_json(c)
        assert set(payload) >= {"call_index", "kind", "messages",
                                "reply_content", "usage", "elapsed_s"}
        assert payload["usage"] != {}

    fake = FakeClient.last_instance
    assert len(fake.calls) == 3
    assert '"image_url"' in json.dumps(fake.calls[0][0])

    results = load_json(out / "results.json")
    assert results["status"] == "ok"
    assert results["gate_passed"] is True
    assert not results["errors"]
    names = [s["name"] for s in results["subscores"]]
    assert set(names) == {"playability", "sync", "density", "mod_usage", "coverage"}
    for s in results["subscores"]:
        assert 0.0 <= s["score"] <= 1.0, f"{s['name']}={s['score']}"
    assert results["final"] is not None and 0.0 <= results["final"] <= 1.0

    assert (out / "track.zip").exists()
    with zipfile.ZipFile(out / "track.zip") as z:
        assert z.testzip() is None
        names_in_zip = z.namelist()
        assert any(n.endswith("/Info.dat") for n in names_in_zip)
        assert any(n.endswith("/ExpertPlus.dat") for n in names_in_zip)
        assert any(n.endswith("/song.ogg") for n in names_in_zip)
        assert any(n.endswith("/cover.jpg") for n in names_in_zip)
