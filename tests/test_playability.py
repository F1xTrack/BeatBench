from beatbench.types import GeneratedPackage
from beatbench.scoring.playability import validate_playability

INFO = {"version": "2.2.0", "songName": "t", "songAuthorName": "a", "beatsPerMinute": 120,
        "songFilename": "s.ogg", "coverImageFilename": "c.jpg",
        "difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
          "_difficultyBeatmaps": [{"_difficulty": "X", "_beatmapFilename": "X.dat"}]}]}

def _pkg(notes, bombs=(), obstacles=()):
    return GeneratedPackage(INFO, {"X.dat": {
        "version": "3.3.0", "colorNotes": list(notes),
        "bombNotes": list(bombs), "basicBeatmapEvents": [], "obstacles": list(obstacles)}},
        audio_path=None, cover_path=None)

N = lambda b, x, y, c=0, d=1, cd=None: (
    {"b": b, "x": x, "y": y, "c": c, "d": d} if cd is None else
    {"b": b, "x": x, "y": y, "c": c, "d": d, "customData": cd})


def test_clean_map_full_score():
    s = validate_playability(_pkg([N(0, 1, 0), N(1, 2, 0, c=1)]))
    assert s.score == 1.0


def test_alternating_colors_no_resets():
    s = validate_playability(_pkg([
        N(0.0, 1, 0), N(0.5, 2, 0, c=1),
        N(1.0, 1, 0), N(1.5, 2, 0, c=1)]))
    assert s.details["reset_rate"] == 0
    assert s.score == 1.0


def test_same_color_consecutive_resets_lower_score():
    notes = [N(i * 0.5, 1, min(i, 2)) for i in range(6)]
    alt = [N(i * 0.5, 1, min(i, 2), c=i % 2) for i in range(6)]
    s = validate_playability(_pkg(notes))
    ref = validate_playability(_pkg(alt))
    assert ref.details["reset_rate"] == 0
    assert s.details["reset_rate"] > 0.3
    assert s.score < ref.score - 0.15


def test_duplicate_note_different_colors_no_reset_penalty():
    s = validate_playability(_pkg([N(0, 1, 0), N(0, 1, 0, c=1)]))
    assert s.score == 0.5
    assert s.details["violations"]["duplicate_cell"] == 1
    assert s.details["reset_rate"] == 0


def test_animated_notes_exempt_from_duplicates_and_resets():
    anim = {"animation": {"position": [[0, 0, 0, 0], [0.5, 1, 0, 0]]}}
    s = validate_playability(_pkg([N(0, 1, 0, cd=anim), N(0, 1, 0, c=1, cd=anim)]))
    assert s.score == 1.0


def test_bomb_on_note_is_dirty():
    s = validate_playability(_pkg([N(0, 1, 0)], bombs=[{"b": 0, "x": 1, "y": 0}]))
    assert s.score < 1.0


def test_rule_overlap_counts_once_no_reset_double_penalty():
    s = validate_playability(
        _pkg([N(0, 1, 0), N(0, 1, 0, c=1)], bombs=[{"b": 0, "x": 1, "y": 0}]))
    assert s.score == 0.5
    assert s.details["violations"]["duplicate_cell"] == 1
    assert s.details["violations"]["X.dat:bomb_overlap"] == 1
    assert s.details["total_notes"] == 2


def test_center_wall_is_dirty():
    wall = {"b": 0, "x": 1, "y": 0, "d": 0, "w": 2, "h": 1}
    s = validate_playability(_pkg([N(0, 0, 0)], obstacles=[wall]))
    assert s.score < 1.0


def test_arcs_and_multiblocks_are_counted_as_playable_elements():
    pkg = _pkg([N(0, 1, 0)])
    pkg.difficulties["X.dat"].update({
        "sliders": [{"b": 1, "c": 0, "x": 0, "y": 0, "d": 1,
                     "tb": 2, "tx": 1, "ty": 0, "tc": 0, "m": 1}],
        "burstSliders": [{"b": 2.5, "c": 1, "x": 2, "y": 0, "d": 1,
                          "tb": 3, "tx": 3, "ty": 0, "sc": 3, "s": 0.9}],
    })
    result = validate_playability(pkg)
    assert result.details["total_notes"] == 5
