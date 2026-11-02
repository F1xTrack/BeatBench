"""Tests for analytics.metrics — synthetic fixtures only, no src imports."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from analytics.metrics import (
    density_series,
    energy_density_corr,
    energy_envelope,
    light_stats,
    load_audio,
    load_map,
    njs_stats,
    nps_stats,
    onset_alignment,
    quiet_violation_rate,
    vocal_alignment,
)

SR = 44100
DUR = 20.0
WIN_S = 2.0


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _click_audio() -> np.ndarray:
    """20 s click track: loud (amp 0.9) first half, near-silent (0.01) second.

    Clicks every 0.5 s at exact sample offsets so each 2 s window contains
    four bit-identical bursts -> identical RMS per window within each half.
    """
    total = int(SR * DUR)
    audio = np.zeros(total)
    click_len = int(0.02 * SR)
    t = np.arange(click_len) / SR
    burst_full = np.sin(2 * np.pi * 1000.0 * t) * np.exp(-t * 80.0)
    for start in range(0, total, SR // 2):
        amp = 0.9 if start < total // 2 else 0.01
        end = min(start + click_len, total)
        audio[start:end] += amp * burst_full[: end - start]
    return audio


@pytest.fixture(scope="module")
def click_audio():
    return _click_audio()


@pytest.fixture(scope="module")
def wav_path(tmp_path_factory, click_audio):
    path = tmp_path_factory.mktemp("audio") / "song.wav"
    sf.write(str(path), click_audio, SR)
    return path


def _loud_note_times() -> list[float]:
    """Two clicks per loud window, strictly inside window bounds."""
    return [off + 2.0 * w for w in range(5) for off in (0.5, 1.5)]


def _silent_note_times() -> list[float]:
    return [10.0 + 0.5 * k for k in range(1, 20)]  # 10.5 .. 19.5 s


# --------------------------------------------------------------------------
# energy_envelope / load_audio
# --------------------------------------------------------------------------

def test_energy_envelope_tiles_duration(click_audio):
    env = energy_envelope(click_audio, SR, win_s=WIN_S)
    assert len(env) == int(DUR / WIN_S)
    assert np.allclose(env[:5], env[0])          # loud half uniform
    assert np.allclose(env[5:], env[5])          # silent half uniform
    assert env[0] > 10 * env[5]                  # loud >> near-silent


def test_load_audio_mono_22050(wav_path):
    y, sr = load_audio(wav_path)
    assert sr == 22050
    assert y.ndim == 1
    assert len(y) == int(DUR * 22050)


# --------------------------------------------------------------------------
# density_series
# --------------------------------------------------------------------------

def test_density_series_counts_per_window():
    notes = [{"b": t} for t in (0.5, 1.5, 2.5, 3.5)]
    dens = density_series(notes, DUR, win_s=1.0)
    assert list(dens) == [1, 1, 1, 1] + [0] * 16


def test_density_series_bpm_conversion():
    assert list(density_series([{"b": 2}], 4.0, win_s=1.0, bpm=60)) == [0, 0, 1, 0]
    assert list(density_series([{"b": 2}], 4.0, win_s=1.0, bpm=120)) == [0, 1, 0, 0]


# --------------------------------------------------------------------------
# energy_density_corr / quiet_violation_rate
# --------------------------------------------------------------------------

def test_corr_loud_notes_positive(click_audio):
    env = energy_envelope(click_audio, SR)
    dens = density_series([{"b": t} for t in _loud_note_times()], DUR)
    corr = energy_density_corr(env, dens)
    assert corr["pearson"] is not None and corr["pearson"] > 0.5
    assert corr["spearman"] == pytest.approx(1.0)


def test_energy_density_corr_constant_series_safe():
    corr = energy_density_corr(np.ones(10), np.arange(10))
    assert corr == {"pearson": None, "spearman": None}


def test_quiet_violation_loud_half_zero(click_audio):
    env = energy_envelope(click_audio, SR)
    rate = quiet_violation_rate(_loud_note_times(), env, q=0.25, win_s=WIN_S)
    assert rate == 0.0


def test_quiet_violation_silent_half_one(click_audio):
    env = energy_envelope(click_audio, SR)
    rate = quiet_violation_rate(_silent_note_times(), env, q=0.25, win_s=WIN_S)
    assert rate == 1.0


# --------------------------------------------------------------------------
# nps_stats
# --------------------------------------------------------------------------

def test_nps_stats_exact_arithmetic():
    notes = [{"b": float(i)} for i in range(50)]
    res = nps_stats(notes, 20.0, target=2.0)
    assert res["nps"] == pytest.approx(2.5)
    assert res["target"] == 2.0
    assert res["error"] == pytest.approx(0.25)
    assert res["error_rel"] == pytest.approx(0.25)


def test_nps_stats_no_target():
    res = nps_stats([], 10.0)
    assert res["nps"] == 0.0 and res["target"] is None and res["error"] is None


# --------------------------------------------------------------------------
# onset_alignment / vocal_alignment
# --------------------------------------------------------------------------

def test_onset_alignment_notes_on_clicks(click_audio):
    res = onset_alignment(_loud_note_times(), click_audio, SR)
    assert res["share_within_50ms"] >= 0.95
    assert res["median_abs_dt"] <= 0.03


def test_vocal_alignment_positive(click_audio):
    res = vocal_alignment(_loud_note_times(), click_audio, SR, win_s=WIN_S)
    assert res["pearson"] is not None and res["pearson"] > 0


# --------------------------------------------------------------------------
# light_stats / njs_stats
# --------------------------------------------------------------------------

def test_light_stats_exact_counts():
    events = [
        {"b": 0.0, "etype": 1, "value": 0},
        {"b": 2.5, "etype": 1, "value": 0},
        {"b": 5.0, "etype": 1000, "value": 5},
        {"b": 6.0, "etype": 4096, "value": 7},
    ]
    res = light_stats(events, DUR, win_s=WIN_S)
    assert res["count"] == 4
    assert res["temporal_coverage"] == pytest.approx(0.4)   # windows 0,1,2,3 of 10
    assert res["unique_pairs"] == 3                          # (1,0),(1000,5),(4096,7)
    assert res["chroma_custom"] == 2                         # etype 1000 and 4096


def test_light_stats_empty():
    res = light_stats([], DUR)
    assert res == {"count": 0, "temporal_coverage": 0.0, "unique_pairs": 0,
                   "chroma_custom": 0}


def test_njs_stats_passthrough():
    assert njs_stats({"njs": 14.0, "njs_offset": 1.0}) == {"njs": 14.0, "offset": 1.0}
    assert njs_stats({}) == {"njs": None, "offset": None}


# --------------------------------------------------------------------------
# load_map roundtrips
# --------------------------------------------------------------------------

def test_load_map_v2_zip(tmp_path):
    info = {
        "_version": "2.0.0",
        "_beatsPerMinute": 120.0,
        "_songName": "fixture",
        "_difficultyBeatmapSets": [{
            "_difficultyBeatmaps": [{
                "_difficulty": "ExpertPlus",
                "_beatmapFilename": "ExpertPlus.dat",
            }],
        }],
    }
    diff = {
        "_version": "2.0.0",
        "_notes": [
            {"_time": 4.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 0,
             "_cutDirection": 1},
            {"_time": 8.0, "_lineIndex": 2, "_lineLayer": 1, "_type": 1,
             "_cutDirection": 0},
            {"_time": 9.0, "_lineIndex": 0, "_lineLayer": 0, "_type": 3,
             "_cutDirection": 0},  # bomb -> excluded
        ],
        "_events": [{"_time": 2.0, "_type": 5, "_value": 1}],
        "_noteJumpMovementSpeed": 14.0,
        "_noteJumpStartBeatOffset": 1.5,
    }
    zip_path = tmp_path / "map.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Info.dat", json.dumps(info))
        zf.writestr("ExpertPlus.dat", json.dumps(diff))

    m = load_map(zip_path)
    assert m["format"] == "v2"
    assert m["bpm"] == 120.0
    assert m["njs"] == 14.0
    assert m["njs_offset"] == 1.5
    assert m["notes"] == [
        {"b": 4.0, "x": 1, "y": 0, "c": 0, "d": 1},
        {"b": 8.0, "x": 2, "y": 1, "c": 1, "d": 0},
    ]
    assert m["events"] == [{"b": 2.0, "etype": 5, "value": 1}]


def test_load_map_v3_folder(tmp_path):
    info = {
        "version": "3.0.0",
        "beatsPerMinute": 128.0,
        "difficultyBeatmaps": [{
            "difficulty": "ExpertPlus",
            "beatmapFilename": "ExpertStandard.dat",
            "noteJumpMovementSpeed": 18.0,
            "noteJumpStartBeatOffset": 0.0,
        }],
    }
    diff = {
        "version": "3.0.0",
        "colorNotes": [
            {"b": 1.0, "x": 1, "y": 0, "c": 0, "d": 1},
            {"b": 2.0, "x": 0, "y": 1, "c": 1, "d": 0},
        ],
        "bombNotes": [{"b": 3.0, "x": 2, "y": 0}],
        "obstacles": [{"b": 0.0, "x": 0, "y": 0, "duration": 4.0, "width": 1}],
        "basicBeatmapEvents": [
            {"b": 0.0, "e": 1, "f": 5.0, "i": 0},
            {"b": 4.0, "e": 1000, "f": 1.0, "i": 7},
        ],
    }
    folder = tmp_path / "mapdir"
    folder.mkdir()
    (folder / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    (folder / "ExpertStandard.dat").write_text(json.dumps(diff), encoding="utf-8")

    m = load_map(folder)
    assert m["format"] == "v3"
    assert m["bpm"] == 128.0
    assert m["njs"] == 18.0
    assert m["njs_offset"] == 0.0
    assert m["notes"] == [
        {"b": 1.0, "x": 1, "y": 0, "c": 0, "d": 1},
        {"b": 2.0, "x": 0, "y": 1, "c": 1, "d": 0},
    ]
    assert m["events"] == [
        {"b": 0.0, "etype": 1, "value": 5.0},
        {"b": 4.0, "etype": 1000, "value": 1.0},
    ]


def test_load_map_v3_projects_arcs_and_multiblock_slices(tmp_path):
    info = {
        "version": "3.0.0", "beatsPerMinute": 120.0,
        "difficultyBeatmaps": [{"difficulty": "ExpertPlus",
                                 "beatmapFilename": "ExpertStandard.dat"}],
    }
    diff = {
        "version": "3.0.0",
        "colorNotes": [{"b": 1.0, "x": 0, "y": 0, "c": 0, "d": 1}],
        "sliders": [{"b": 2.0, "x": 1, "y": 0, "c": 0, "d": 1,
                     "tb": 6.0, "tx": 2, "ty": 1, "tc": 0, "m": 1}],
        "burstSliders": [{"b": 8.0, "x": 2, "y": 1, "c": 1, "d": 0,
                          "tb": 10.0, "tx": 3, "ty": 1, "sc": 3, "s": 0.8}],
        "basicBeatmapEvents": [],
    }
    folder = tmp_path / "arc-map"
    folder.mkdir()
    (folder / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    (folder / "ExpertStandard.dat").write_text(json.dumps(diff), encoding="utf-8")

    loaded = load_map(folder)
    assert loaded["arcs"] == 1
    assert loaded["chains"] == 1
    assert loaded["chain_slices"] == 3
    assert loaded["arc_duration_beats"] == pytest.approx(4.0)
    assert loaded["chain_duration_beats"] == pytest.approx(2.0)
    assert loaded["invalid_elements"] == 0
    assert loaded["notes"] == [
        {"b": 1.0, "x": 0, "y": 0, "c": 0, "d": 1},
        {"b": 2.0, "x": 1, "y": 0, "c": 0, "d": 1, "kind": "arc_head"},
        {"b": 8.0, "x": 2, "y": 1, "c": 1, "d": 0, "kind": "chain_slice"},
        {"b": 9.0, "x": 2, "y": 1, "c": 1, "d": 8, "kind": "chain_slice"},
        {"b": 10.0, "x": 3, "y": 1, "c": 1, "d": 8, "kind": "chain_slice"},
    ]
    assert nps_stats(loaded["notes"], 5.0)["nps"] == pytest.approx(1.0)


def test_load_map_missing_path_raises(tmp_path):
    with pytest.raises(ValueError):
        load_map(tmp_path / "does_not_exist.zip")
