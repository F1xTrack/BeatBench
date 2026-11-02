import numpy as np
import pytest
import soundfile as sf

from beatbench.scoring.sync import validate_sync
from beatbench.types import GeneratedPackage

INFO = {"version": "2.2.0", "beatsPerMinute": 120, "songName": "t", "songAuthorName": "a",
        "songFilename": "s.wav", "coverImageFilename": "c.jpg",
        "difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
          "_difficultyBeatmaps": [{"_difficulty": "X", "_beatmapFilename": "X.dat"}]}]}


@pytest.fixture(scope="module")
def click_audio(tmp_path_factory):
    sr, dur = 22050, 10.0
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for k in np.arange(0, dur, 0.5):
        idx = int(k * sr)
        y[idx:idx + 400] += np.exp(-np.arange(400) / 60) * np.sin(2 * np.pi * 1000 * t[:400])
    p = tmp_path_factory.mktemp("aud") / "s.wav"
    sf.write(p, y, sr)
    return p


def test_on_beat_notes_score_high(click_audio):
    notes = [{"b": float(k), "x": i % 4, "y": 0, "c": i % 2, "d": 1}
             for i, k in enumerate(range(1, 17))]
    pkg = GeneratedPackage(INFO, {"X.dat": {"version": "3", "colorNotes": notes,
        "basicBeatmapEvents": []}}, audio_path=click_audio, cover_path=None)
    s = validate_sync(pkg)
    assert s.name == "sync"
    assert s.score > 0.7


def test_off_beat_notes_score_low(click_audio):
    notes = [{"b": float((k * 0.5 + 0.23) * 2), "x": 1, "y": 0, "c": 0, "d": 1}
             for k in range(1, 18)]
    pkg = GeneratedPackage(INFO, {"X.dat": {"version": "3", "colorNotes": notes,
        "basicBeatmapEvents": []}}, audio_path=click_audio, cover_path=None)
    s = validate_sync(pkg)
    assert s.score < 0.5


def test_no_notes_scores_zero(click_audio):
    pkg = GeneratedPackage(INFO, {"X.dat": {"version": "3", "colorNotes": [],
        "basicBeatmapEvents": []}}, audio_path=click_audio, cover_path=None)
    s = validate_sync(pkg)
    assert s.score == 0.0


def test_details_present(click_audio):
    notes = [{"b": float(k), "x": 1, "y": 0, "c": 0, "d": 1}
             for k in range(1, 17)]
    pkg = GeneratedPackage(INFO, {"X.dat": {"version": "3", "colorNotes": notes,
        "basicBeatmapEvents": []}}, audio_path=click_audio, cover_path=None)
    s = validate_sync(pkg)
    for key in ("within_tolerance", "median_deviation_ms", "accent_coverage"):
        assert key in s.details


def test_arcs_and_chain_slices_participate_in_sync(click_audio):
    dat = {"colorNotes": [], "basicBeatmapEvents": [],
           "sliders": [{"b": 2, "c": 0, "x": 1, "y": 0, "d": 1,
                        "tb": 2.5, "tx": 2, "ty": 0, "tc": 0, "m": 1}],
           "burstSliders": [{"b": 3, "c": 1, "x": 2, "y": 0, "d": 1,
                             "tb": 4, "tx": 3, "ty": 0, "sc": 3, "s": 0.9}]}
    pkg = GeneratedPackage(INFO, {"X.dat": dat}, audio_path=click_audio, cover_path=None)
    result = validate_sync(pkg)
    assert result.details["within_tolerance"] > 0.5
