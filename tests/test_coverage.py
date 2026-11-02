"""Coverage validator tests: notes must span the whole (capped) audio duration."""
from __future__ import annotations
from pathlib import Path

from beatbench.scoring.coverage import validate_coverage
from beatbench.types import GeneratedPackage


def make_pkg(bpm: float, files: dict[str, list[float]]):
    difficulties = {fname: {"colorNotes": [{"b": b} for b in beats]} for fname, beats in files.items()}
    info = {"beatsPerMinute": bpm} if bpm else {}
    return GeneratedPackage(info=info, difficulties=difficulties, audio_path=Path("song.ogg"), cover_path=None)


def test_full_coverage_is_one():
    pkg = make_pkg(120.0, {"ExpertPlus.dat": [0, 60, 240]})
    res = validate_coverage(pkg, 120.0)
    assert res.name == "coverage"
    assert abs(res.score - 1.0) < 1e-6
    assert res.details["coverage_per_file"]["ExpertPlus.dat"] == 1.0


def test_notes_stopping_halfway_scores_about_half():
    pkg = make_pkg(120.0, {"ExpertPlus.dat": [0, 50, 200]})
    res = validate_coverage(pkg, 200.0)
    assert abs(res.score - 0.5) < 1e-6


def test_no_notes_scores_zero():
    pkg = make_pkg(120.0, {"ExpertPlus.dat": []})
    res = validate_coverage(pkg, 100.0)
    assert res.score == 0.0
    assert res.details["coverage_per_file"]["ExpertPlus.dat"] == 0.0


def test_final_takes_minimum_across_files():
    pkg = make_pkg(120.0, {
        "ExpertPlus.dat": [0, 200],
        "Expert.dat": [0, 50],
    })
    res = validate_coverage(pkg, 200.0)
    assert abs(res.score - 0.125) < 1e-6
    assert res.details["coverage_per_file"] == {"ExpertPlus.dat": 0.5, "Expert.dat": 0.125}


def test_duration_capped_at_240_seconds():
    pkg = make_pkg(120.0, {"ExpertPlus.dat": [0, 480]})
    res = validate_coverage(pkg, 480.0)
    assert abs(res.score - 1.0) < 1e-6
    assert res.details["audio_duration_sec"] == 480.0


def test_missing_bpm_scores_zero():
    pkg = make_pkg(None, {"ExpertPlus.dat": [0, 10]})
    res = validate_coverage(pkg, 100.0)
    assert res.score == 0.0


def test_missing_duration_scores_zero():
    pkg = make_pkg(120.0, {"ExpertPlus.dat": [0, 10]})
    res = validate_coverage(pkg, None)
    assert res.score == 0.0


def test_arc_or_chain_tail_extends_coverage():
    pkg = GeneratedPackage(
        info={"beatsPerMinute": 120.0},
        difficulties={"ExpertPlus.dat": {
            "colorNotes": [{"b": 0}],
            "sliders": [{"b": 20, "tb": 240}],
            "burstSliders": [{"b": 240, "tb": 480}],
        }},
        audio_path=Path("song.ogg"), cover_path=None)
    result = validate_coverage(pkg, 240.0)
    assert result.score == 1.0
