"""Adversarial checks of scoring behavior, independent of model names and today's maps."""
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf

from beatbench.types import GeneratedPackage
from beatbench.scoring.review import review_package
from beatbench.scoring.assessment import assess_package, geometric_score


@pytest.fixture
def package(tmp_path):
    rate, duration = 22050, 32
    audio = np.zeros(rate * duration, dtype=np.float32)
    times = np.arange(0.4, duration - 0.4, 0.4)
    for i, t in enumerate(times):
        start = int(t * rate)
        rng = np.random.default_rng(i)
        burst = rng.standard_normal(650) * np.exp(-np.arange(650) / 90)
        audio[start:start + 650] += burst * (0.12 if i < 40 else 0.4)
    path = tmp_path / "audio.wav"
    sf.write(path, audio, rate)
    notes = [{"b": float(t * 2), "x": 0 if i % 2 == 0 else 3, "y": 0,
              "c": i % 2, "d": 1 if (i // 2) % 2 == 0 else 0, "a": 0} for i, t in enumerate(times)]
    info = {"beatsPerMinute": 120, "difficultyBeatmapSets": [{"_difficultyBeatmaps": [
        {"_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 14, "_noteJumpStartBeatOffset": 0}]}]}
    return GeneratedPackage(info, {"ExpertPlus.dat": {"colorNotes": notes, "basicBeatmapEvents": []}}, path, None)


def assess(pkg, *, audio=True, nps=2.5):
    result = assess_package(pkg, review_package(pkg, analyze_audio=audio, target_nps=nps), nps)
    return result, result["difficulty_assessments"]["ExpertPlus.dat"]


def test_shifted_timing_scores_lower(package):
    _, original = assess(package)
    shifted = deepcopy(package)
    for note in shifted.difficulties["ExpertPlus.dat"]["colorNotes"]:
        note["b"] += 0.3
    _, bad = assess(shifted)
    assert bad["axes"]["musicality"]["score"] < original["axes"]["musicality"]["score"] - 0.2


def test_same_downstrokes_reduce_movement(package):
    _, original = assess(package)
    bad = deepcopy(package)
    for note in bad.difficulties["ExpertPlus.dat"]["colorNotes"]:
        note["d"] = 1
    _, mutated = assess(bad)
    assert mutated["axes"]["movement"]["score"] < original["axes"]["movement"]["score"]


def test_fast_returns_trigger_noncompensating_cap(package):
    bad = deepcopy(package)
    for i, note in enumerate(bad.difficulties["ExpertPlus.dat"]["colorNotes"]):
        note.update(b=i * 0.08, d=1)
    _, mutated = assess(bad)
    assert any(x["code"] == "pervasive_fast_return_candidates" for x in mutated["caps"])
    assert mutated["experimental_score"] <= 0.39


def test_single_tail_note_cannot_get_high_coverage_score(package):
    bad = deepcopy(package)
    bad.difficulties["ExpertPlus.dat"]["colorNotes"] = [{"b": 63, "x": 0, "y": 0, "c": 0, "d": 1}]
    _, mutated = assess(bad)
    assert mutated["experimental_score"] is None
    assert any(x["code"] == "major_unmapped_audible_sections" for x in mutated["caps"])


def test_no_audio_or_unknown_geometry_cannot_inflate_remaining_axes(package):
    result, data = assess(package, audio=False)
    assert result["experimental_score"] is None
    assert "musicality" in data["unrated_axes"]
    bad = deepcopy(package)
    bad.difficulties["ExpertPlus.dat"]["colorNotes"][0]["customData"] = {"animation": {"position": []}}
    assert assess(bad)[0]["experimental_score"] is None


def test_mod_flag_and_exact_target_nps_earn_no_bonus(package):
    a, _ = assess(package)
    other = deepcopy(package)
    other.info["customData"] = {"_requirements": ["Chroma", "Noodle Extensions"]}
    b, _ = assess(other, nps=8)
    assert a["experimental_score"] == b["experimental_score"]


def test_empty_lighting_does_not_change_gameplay_but_changes_presentation(package):
    a, _ = assess(package)
    lit = deepcopy(package)
    lit.difficulties["ExpertPlus.dat"]["basicBeatmapEvents"] = [{"b": i * 0.8, "et": 0, "i": 1} for i in range(79)]
    b, data = assess(lit)
    assert a["experimental_score"] == b["experimental_score"]
    assert data["presentation"]["score"] > 0


def test_bpm_reencoding_preserves_audio_and_movement_axes(package):
    _, a = assess(package)
    recoded = deepcopy(package)
    recoded.info["beatsPerMinute"] = 240
    for note in recoded.difficulties["ExpertPlus.dat"]["colorNotes"]:
        note["b"] *= 2
    _, b = assess(recoded)
    for axis in ("musicality", "movement", "pattern_development"):
        assert a["axes"][axis]["score"] == b["axes"][axis]["score"]


def test_geometric_mean_penalizes_core_failure_and_missing_stays_unknown():
    good = {k: {"score": 0.9} for k in ("musicality", "movement", "pattern_development", "readability")}
    bad = deepcopy(good)
    bad["musicality"]["score"] = 0.1
    assert geometric_score(bad) < 0.45
    bad["musicality"]["score"] = None
    assert geometric_score(bad) is None


def test_arcs_and_chains_are_reviewed_without_disabling_assessment(package):
    enriched = deepcopy(package)
    dat = enriched.difficulties["ExpertPlus.dat"]
    dat["sliders"] = [{"b": 2, "c": 0, "x": 0, "y": 0, "d": 1,
                        "tb": 2.5, "tx": 1, "ty": 1, "tc": 0, "m": 1}]
    dat["burstSliders"] = [{"b": 4, "c": 1, "x": 3, "y": 0, "d": 1,
                            "tb": 4.5, "tx": 2, "ty": 1, "sc": 4, "s": 0.9}]
    review = review_package(enriched, target_nps=2.5)
    evidence = review["difficulty_reviews"]["ExpertPlus.dat"]
    assert evidence["elements"]["valid"] is True
    assert evidence["elements"]["arcs"] == 1
    assert evidence["elements"]["chains"] == 1
    assert evidence["playable_hit_count"] == len(dat["colorNotes"]) + 1 + 4
    result, assessed = assess(enriched)
    assert assessed["experimental_score"] is not None
    assert assessed["axes"]["movement"]["score"] is not None
