"""V3→V2 converter tests: saber_colors sidecar injection."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from beatbench.export.v2 import convert_difficulty_v3_to_v2, convert_folder

SABER_COLORS = {"left": [255, 80, 80], "right": [40, 120, 255]}

V3_INFO = {
    "songName": "T",
    "songSubName": "",
    "songAuthorName": "A",
    "levelAuthorName": "M",
    "beatsPerMinute": 128,
    "songFilename": "song.ogg",
    "coverImageFilename": "cover.png",
    "environmentName": "DefaultEnvironment",
    "difficultyBeatmapSets": [{
        "_beatmapCharacteristicName": "Standard",
        "_difficultyBeatmaps": [{
            "_difficulty": "ExpertPlus",
            "_difficultyRank": 9,
            "_beatmapFilename": "ExpertPlus.dat",
            "_noteJumpMovementSpeed": 18,
            "_noteJumpStartBeatOffset": 0,
        }],
    }],
}

V3_DIFF = {
    "colorNotes": [{"b": 4.0, "x": 1, "y": 0, "c": 0, "d": 1}],
    "bombNotes": [],
    "obstacles": [],
    "basicBeatmapEvents": [],
}


@pytest.fixture()
def src_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "Info.dat").write_text(json.dumps(V3_INFO), encoding="utf-8")
    (src / "ExpertPlus.dat").write_text(json.dumps(V3_DIFF), encoding="utf-8")
    return src


def test_sidecar_colors_injected(src_dir):
    (src_dir / "saber_colors.json").write_text(json.dumps(SABER_COLORS), encoding="utf-8")
    dst = src_dir.parent / "dst"
    convert_folder(src_dir, dst)
    info = json.loads((dst / "Info.dat").read_text(encoding="utf-8"))
    profile = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    cd = profile["_customData"]
    left = {"r": 1.0, "g": 80 / 255, "b": 80 / 255, "a": 1.0}
    right = {"r": 40 / 255, "g": 120 / 255, "b": 1.0, "a": 1.0}
    assert cd["_colorLeft"] == pytest.approx(left)
    assert cd["_colorRight"] == pytest.approx(right)
    assert cd["_envColorLeft"] == pytest.approx(left)
    assert cd["_envColorRight"] == pytest.approx(right)
    assert info["_colorSchemes"][profile["_beatmapColorSchemeIdx"]]["useOverride"] is True


def test_no_sidecar_no_custom_data(src_dir):
    dst = src_dir.parent / "dst"
    convert_folder(src_dir, dst)
    info = json.loads((dst / "Info.dat").read_text(encoding="utf-8"))
    assert "_customData" not in info


def test_invalid_sidecar_ignored(src_dir):
    (src_dir / "saber_colors.json").write_text('{"left": [1, 2]}', encoding="utf-8")
    dst = src_dir.parent / "dst"
    convert_folder(src_dir, dst)
    info = json.loads((dst / "Info.dat").read_text(encoding="utf-8"))
    assert "_customData" not in info


def test_v2_conversion_refuses_to_drop_arcs_or_chains():
    with pytest.raises(ValueError, match="arcs.*chains"):
        convert_difficulty_v3_to_v2({
            "colorNotes": [], "bombNotes": [], "obstacles": [], "basicBeatmapEvents": [],
            "sliders": [{"b": 1, "c": 0, "x": 0, "y": 0, "d": 1,
                         "tb": 2, "tx": 1, "ty": 0, "tc": 0, "m": 0}],
        })
