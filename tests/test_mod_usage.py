from copy import deepcopy
from beatbench.types import GeneratedPackage
from beatbench.scoring.mod_usage import validate_mod_usage
from tests.fixtures_helpers import INFO

BASE_DIFF = {"version": "3", "colorNotes": [{"b": 0, "x": 1, "y": 0, "c": 0, "d": 1}],
             "basicBeatmapEvents": []}


def _pkg(diff):
    return GeneratedPackage(INFO, {"X.dat": diff}, audio_path=None, cover_path=None)


def test_both_mods_full():
    d = deepcopy(BASE_DIFF)
    d["basicBeatmapEvents"] = [{"b": 0, "et": 1, "i": 5, "customData": {"colorData": [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]]}}]
    d["colorNotes"][0]["customData"] = {"animation": {"position": [[[0, 0, 0, 0]], [[0.5, 1, 0, 0]]]}}
    s = validate_mod_usage(_pkg(d))
    assert s.score == 1.0
    assert s.name == "mod_usage"
    assert s.details == {"chroma": 1.0, "noodle": 1.0}


def test_no_mods_zero():
    s = validate_mod_usage(_pkg(deepcopy(BASE_DIFF)))
    assert s.score == 0.0


def test_chroma_only_half():
    d = deepcopy(BASE_DIFF)
    d["lightColorEventBoxGroups"] = []
    d["useNormalEventsAsCompatibleEvents"] = True
    d["basicBeatmapEvents"] = [{"b": 0, "et": 2, "i": 3}]
    s = validate_mod_usage(_pkg(d))
    assert s.score == 0.5
    assert s.details == {"chroma": 1.0, "noodle": 0.0}


def test_noodle_animation_on_arc_or_chain_is_detected():
    d = deepcopy(BASE_DIFF)
    animated = {"animation": {"offsetPosition": [[0, 0, 0, 0]]}}
    d["sliders"] = [{"b": 1, "c": 0, "x": 0, "y": 0, "d": 1,
                      "tb": 2, "tx": 1, "ty": 0, "tc": 0, "m": 1,
                      "customData": animated}]
    s = validate_mod_usage(_pkg(d))
    assert s.details["noodle"] == 1.0
