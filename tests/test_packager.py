"""Packager tests: saberColors sidecar extraction."""
from __future__ import annotations
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from beatbench.runner.client import ChatReply
import beatbench.runner.pipeline as pipeline_mod
from beatbench.runner.pipeline import evaluate_track
from beatbench.types import TrackMeta
from tests.fixtures_helpers import FIXTURES

INFO = json.loads((FIXTURES / "maps" / "valid" / "info.json").read_text(encoding="utf-8"))
DIFF = json.loads((FIXTURES / "maps" / "valid" / "ExpertPlus.dat").read_text(encoding="utf-8"))

SABER_COLORS = {"left": [255, 80, 80], "right": [40, 120, 255]}


class FakeClient:
    def __init__(self, endpoint, key, model, replies):
        self.endpoint, self.key, self.model = endpoint, key, model
        self.replies = list(replies)

    @classmethod
    def factory(cls, replies):
        def make(endpoint, key, model):
            return cls(endpoint, key, model, replies)
        return make

    def chat(self, messages, response_format=None, reasoning_effort=None):
        content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return ChatReply(content=content,
                         usage={"prompt_tokens": 5, "completion_tokens": 7},
                         elapsed_s=0.02)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav = tmp_path / "track.wav"
    sf.write(wav, np.zeros(2000, dtype="float32"), 44100, format="WAV")
    return wav


def meta():
    return TrackMeta(title="Test", author="Artist", bpm=120.0,
                     difficulties=["ExpertPlus"])


def run(env, monkeypatch, info):
    monkeypatch.setattr(pipeline_mod, "ModelClient",
                        FakeClient.factory([json.dumps({"info": info}),
                                            json.dumps(DIFF)]))
    return evaluate_track(env, "http://ep", "key", "m1", meta(),
                          {"weights": {"playability": 0.0, "sync": 0.0,
                                       "density": 0.0, "mod_usage": 0.0,
                                       "coverage": 0.0}})


def test_saber_colors_moved_to_sidecar(env, monkeypatch):
    info = {**INFO, "saberColors": SABER_COLORS}
    out = run(env, monkeypatch, info)

    with zipfile.ZipFile(out / "track.zip") as z:
        assert "Artist - Test/saber_colors.json" in z.namelist()
        sidecar = json.loads(z.read("Artist - Test/saber_colors.json")
                             .decode("utf-8"))
        assert sidecar == SABER_COLORS
        info_dat = json.loads(z.read("Artist - Test/Info.dat").decode("utf-8"))
        assert "saberColors" not in info_dat


def test_no_sidecar_without_saber_colors(env, monkeypatch):
    out = run(env, monkeypatch, dict(INFO))

    with zipfile.ZipFile(out / "track.zip") as z:
        assert "Artist - Test/saber_colors.json" not in z.namelist()
        info_dat = json.loads(z.read("Artist - Test/Info.dat").decode("utf-8"))
        expected = {**INFO, "levelAuthorName": "m1"}
        assert info_dat == expected


def test_build_zip_does_not_mutate_pkg_info(tmp_path):
    from beatbench.export.packager import build_zip
    from beatbench.types import GeneratedPackage

    audio = tmp_path / "track.wav"
    sf.write(audio, np.zeros(1000, dtype="float32"), 44100, format="WAV")
    pkg = GeneratedPackage(
        info={"songName": "Song", "saberColors": SABER_COLORS},
        difficulties={"Easy.dat": DIFF},
        audio_path=audio,
        cover_path=None,
    )
    build_zip(pkg, tmp_path, "track")

    assert pkg.info["saberColors"] == SABER_COLORS
