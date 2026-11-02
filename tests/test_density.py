import numpy as np
import pytest
import soundfile as sf

from beatbench.types import GeneratedPackage
from beatbench.scoring.density import validate_density
from tests.fixtures_helpers import INFO

AUD_INFO = {"version": "2.2.0", "beatsPerMinute": 120, "songName": "t",
            "songAuthorName": "a", "songFilename": "s.wav",
            "coverImageFilename": "c.jpg",
            "difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
              "_difficultyBeatmaps": [{"_difficulty": "ExpertPlus",
                                       "_beatmapFilename": "X.dat"}]}]}


def _pkg(notes):
    return GeneratedPackage(INFO, {"X.dat": {"version": "3", "colorNotes": notes,
        "basicBeatmapEvents": []}}, audio_path=None, cover_path=None)


def _audio_pkg(notes, audio_path):
    return GeneratedPackage(AUD_INFO, {"X.dat": {"version": "3", "colorNotes": notes,
        "basicBeatmapEvents": []}}, audio_path=audio_path, cover_path=None)


def test_target_nps_scores_high():
    notes = [{"b": i * 0.125 * 2, "x": i % 4, "y": 0, "c": i % 2, "d": 1}
             for i in range(int(8.0 * 8))]
    s = validate_density(_pkg(notes), "ExpertPlus", 120.0)
    assert s.score > 0.75


def test_sparse_map_penalized():
    notes = [{"b": i * 16, "x": 1, "y": 0, "c": 0, "d": 1} for i in range(4)]
    s = validate_density(_pkg(notes), "ExpertPlus", 120.0)
    assert s.score < 0.5
    assert s.details["dead_zones_count"] >= 2


def test_easy_difficulty_lower_target():
    dense = [{"b": i * 0.0625 * 2, "x": i % 4, "y": 0, "c": 0, "d": 1}
             for i in range(int(4 * 16))]
    s = validate_density(_pkg(dense), "Easy", 120.0)
    assert s.details["target_nps"] == 1.0


def _multi_info():
    return {"difficultyBeatmapSets": [{"_difficultyBeatmaps": [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"},
        {"_difficulty": "Easy", "_beatmapFilename": "Easy.dat"},
    ]}]}


def test_only_requested_difficulty_file_used():
    ep_notes = [{"b": 0, "x": 0, "y": 0, "c": 0, "d": 1}]
    easy_notes = [{"b": i * 2, "x": 0, "y": 0, "c": 0, "d": 1} for i in range(8)]
    pkg = GeneratedPackage(
        _multi_info(), {"ExpertPlus.dat": {"colorNotes": ep_notes},
                        "Easy.dat": {"colorNotes": easy_notes}},
        audio_path=None, cover_path=None)
    s = validate_density(pkg, "ExpertPlus", 120.0)
    assert s.details["mean_nps"] == round(1 / (4 * 0.5), 3)


SR = 22050
DUR = 9.0


def _sine(envelope_fn):
    t = np.arange(int(SR * DUR)) / SR
    return 0.9 * envelope_fn(t) * np.sin(2 * np.pi * 440.0 * t)


@pytest.fixture(scope="module")
def silent_middle_audio(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "silent_mid.wav"
    sf.write(p, _sine(lambda t: np.where((t >= 3.0) & (t < 6.0), 0.0, 1.0)), SR)
    return p


@pytest.fixture(scope="module")
def ramp_audio(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "ramp.wav"
    sf.write(p, _sine(lambda t: 0.2 + 0.8 * t / DUR), SR)
    return p


def test_notes_in_silent_section_penalized(silent_middle_audio, ramp_audio):
    notes = [{"b": 6.0 + i * 0.5, "x": i % 4, "y": 0, "c": i % 2, "d": 1}
             for i in range(12)]
    s_quiet = validate_density(_audio_pkg(notes, silent_middle_audio),
                               "ExpertPlus", 120.0)
    s_ref = validate_density(_audio_pkg(notes, ramp_audio),
                             "ExpertPlus", 120.0)
    assert s_quiet.details["quiet_ratio"] > 0.9
    assert s_quiet.score < s_ref.score
    assert s_ref.details["quiet_ratio"] == 0.0


def test_missing_audio_skips_quiet_penalty():
    notes = [{"b": i * 0.5, "x": 0, "y": 0, "c": 0, "d": 1} for i in range(8)]
    s = validate_density(_pkg(notes), "ExpertPlus", 120.0)
    assert "quiet_ratio" not in s.details
    assert "energy_note" in s.details


def test_unreadable_audio_skips_quiet_penalty(tmp_path):
    notes = [{"b": i * 0.5, "x": 0, "y": 0, "c": 0, "d": 1} for i in range(8)]
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    s = validate_density(_audio_pkg(notes, bad), "ExpertPlus", 120.0)
    assert "quiet_ratio" not in s.details
    assert "energy_note" in s.details


def test_meta_nps_overrides_table_target():
    notes = [{"b": i * 0.875, "x": i % 4, "y": 0, "c": i % 2, "d": 1}
             for i in range(24)]
    s = validate_density(_pkg(notes), "ExpertPlus", 120.0, target_nps=2.5)
    assert s.details["target_nps"] == 2.5
    assert s.score > 0.75


def test_arcs_and_chain_slices_contribute_to_density():
    pkg = _pkg([{"b": 0, "x": 0, "y": 0, "c": 0, "d": 1}])
    pkg.difficulties["X.dat"].update({
        "sliders": [{"b": 2, "c": 0, "x": 1, "y": 0, "d": 1,
                     "tb": 3, "tx": 2, "ty": 0, "tc": 0, "m": 1}],
        "burstSliders": [{"b": 4, "c": 1, "x": 2, "y": 0, "d": 1,
                          "tb": 5, "tx": 3, "ty": 0, "sc": 4, "s": 0.9}],
    })
    result = validate_density(pkg, "ExpertPlus", 120.0, target_nps=1.0)
    assert result.details["mean_nps"] > 0
