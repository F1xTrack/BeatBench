"""Golden tests: synthetic V3 positive and real human-mapped V2 negative."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from beatbench.types import GeneratedPackage
from beatbench.scoring.density import validate_density
from beatbench.scoring.mod_usage import validate_mod_usage
from beatbench.scoring.playability import validate_playability
from beatbench.scoring.schema_check import check_schema
from beatbench.scoring.sync import validate_sync

NOTE_COUNT = 200
TOTAL_BEATS = 64


def build_synthetic_v3(audio_path: Path) -> GeneratedPackage:
    info = {
        "version": "3.3.0",
        "songName": "Golden Synthetic",
        "songSubName": "",
        "songAuthorName": "Bench",
        "levelAuthorName": "Bench",
        "beatsPerMinute": 120,
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
    color_notes = []
    for i in range(NOTE_COUNT):
        note = {
            "b": round(i * (TOTAL_BEATS / NOTE_COUNT), 3),
            "x": i % 4,
            "y": (i // 4) % 3,
            "c": i % 2,
            "d": i % 9,
            "a": 0,
        }
        if i % 10 == 0:
            note["customData"] = {
                "animation": {"position": [[0, 0, 0], 0, "easeOutCubic"]}}
        color_notes.append(note)
    events = [
        {"b": float(i) * 4.0, "et": 1, "i": 7,
         "customData": {"color": [1.0, 0.2, 0.2, 1.0]}}
        for i in range(16)
    ]
    dat = {
        "version": "3.3.0",
        "colorNotes": color_notes,
        "bombNotes": [],
        "obstacles": [
            {"b": 20.0, "x": 2, "y": 0, "d": 4.0, "w": 2, "h": 5},
            {"b": 40.0, "x": 0, "y": 2, "d": 2.0, "w": 4, "h": 1},
        ],
        "basicBeatmapEvents": events,
        "useNormalEventsAsCompatibleEvents": True,
    }
    return GeneratedPackage(info=info,
                            difficulties={"ExpertPlus.dat": dat},
                            audio_path=audio_path,
                            cover_path=None)


def make_quiet_wav(tmp_path: Path) -> Path:
    audio = tmp_path / "song.ogg"
    sf.write(audio, np.zeros(1000, dtype="float32"), 44100, format="OGG")
    return audio


def load_v2_pkg(tmp_path: Path) -> GeneratedPackage:
    audio = make_quiet_wav(tmp_path)
    info = {"_version": "2.1.0", "_songName": "Synthetic V2",
            "_songFilename": "song.ogg", "_difficultyBeatmapSets": []}
    difficulties = {
        "ExpertStandard.dat": {"_version": "2.0.0", "_notes": [], "_obstacles": []},
        "HardStandard.dat": {"_version": "2.0.0", "_notes": [], "_obstacles": []},
    }
    return GeneratedPackage(info=info, difficulties=difficulties,
                            audio_path=audio, cover_path=None)


def assert_score_in_range(sub):
    assert 0.0 <= sub.score <= 1.0, f"{sub.name}={sub.score}"


def test_golden_positive_synthetic_v3_passes_gate_and_scores(tmp_path):
    pkg = build_synthetic_v3(make_quiet_wav(tmp_path))
    ok, errors = check_schema(pkg)
    assert ok is True
    assert errors == []

    for validator in (validate_playability, validate_mod_usage,
                      validate_sync):
        assert_score_in_range(validator(pkg))
    assert_score_in_range(validate_density(pkg, "ExpertPlus", 120.0))

    mod_usage = validate_mod_usage(pkg)
    assert mod_usage.details["chroma"] > 0
    assert mod_usage.details["noodle"] > 0

    playability = validate_playability(pkg)
    assert playability.details["total_notes"] == NOTE_COUNT


def test_golden_negative_real_v2_map_fails_gate_without_raising(tmp_path):
    pkg = load_v2_pkg(tmp_path)
    ok, errors = check_schema(pkg)
    assert ok is False
    assert len(errors) > 0

    for validator in (validate_playability, validate_mod_usage,
                      validate_sync):
        assert_score_in_range(validator(pkg))
    for difficulty in ("Expert", "Hard"):
        assert_score_in_range(validate_density(pkg, difficulty, 90.0))
