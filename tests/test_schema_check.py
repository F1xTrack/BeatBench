import json
from pathlib import Path
from beatbench.types import GeneratedPackage
from beatbench.scoring.schema_check import check_schema

FIX = Path(__file__).parent / "fixtures" / "maps"


def _pkg(which: str) -> GeneratedPackage:
    d = FIX / which
    info = json.loads((d / "info.json").read_text())
    diffs = {"ExpertPlus.dat": json.loads((d / "ExpertPlus.dat").read_text())}
    return GeneratedPackage(info, diffs, audio_path=d / "song.ogg", cover_path=None)


def test_valid_map_passes_gate():
    ok, errs = check_schema(_pkg("valid"))
    assert ok, errs


def test_broken_map_fails_with_errors():
    ok, errs = check_schema(_pkg("broken"))
    assert not ok
    assert any("beatsPerMinute" in e for e in errs)
    assert any("x" in e for e in errs)


def test_v3_arcs_and_chains_pass_schema():
    pkg = _pkg("valid")
    dat = pkg.difficulties["ExpertPlus.dat"]
    dat["sliders"] = [{"b": 8, "c": 0, "x": 1, "y": 0, "d": 1,
                        "tb": 8.5, "tx": 2, "ty": 0, "tc": 0, "m": 1}]
    dat["burstSliders"] = [{"b": 9, "c": 1, "x": 2, "y": 0, "d": 0,
                            "tb": 9.5, "tx": 3, "ty": 1, "sc": 4, "s": 0.9}]
    ok, errors = check_schema(pkg)
    assert ok, errors


def test_v3_arc_tail_geometry_is_checked():
    pkg = _pkg("valid")
    pkg.difficulties["ExpertPlus.dat"]["sliders"] = [{
        "b": 8, "c": 0, "x": 1, "y": 0, "d": 1,
        "tb": 7.5, "tx": 2, "ty": 0, "tc": 0, "m": 1,
    }]
    ok, errors = check_schema(pkg)
    assert not ok
    assert any("tail beat must be after head beat" in error for error in errors)


def test_v3_multiblock_slice_count_is_checked():
    pkg = _pkg("valid")
    pkg.difficulties["ExpertPlus.dat"]["burstSliders"] = [{
        "b": 9, "c": 1, "x": 2, "y": 0, "d": 0,
        "tb": 9, "tx": 3, "ty": 1, "sc": 0, "s": 0.9,
    }]
    ok, errors = check_schema(pkg)
    assert not ok
    assert any("slice count must be positive" in error for error in errors)
