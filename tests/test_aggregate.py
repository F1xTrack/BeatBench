"""Aggregator tests: schema gate blocks scoring; final is a weighted mean."""
from __future__ import annotations
import json
from unittest.mock import patch

from beatbench.config import DEFAULT_WEIGHTS
from beatbench.scoring import aggregate as aggregate_mod
from beatbench.scoring.aggregate import run_scoring
from beatbench.types import Subscore, TrackMeta
from tests.fixtures_helpers import valid_pkg


def test_gate_failure_blocks_scoring(tmp_path):
    res = run_scoring(valid_pkg(broken=True), TrackMeta(title="t"), "m1", tmp_path,
                      {"weights": DEFAULT_WEIGHTS})
    assert res.status == "invalid"
    assert res.gate_passed is False
    assert res.final_score is None
    assert res.errors
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["gate_passed"] is False
    assert data["errors"]
    assert data["final"] is None


def _fake_scores(**overrides):
    base = {
        "playability": Subscore("playability", 1.0),
        "sync": Subscore("sync", 0.8),
        "density": Subscore("density", 0.5),
        "mod_usage": Subscore("mod_usage", 0.2),
        "coverage": Subscore("coverage", 0.6),
    }
    base.update(overrides)
    return base


def test_final_is_weighted_mean(tmp_path):
    with patch("beatbench.scoring.aggregate._run_validators",
               return_value=_fake_scores()):
        res = run_scoring(
            valid_pkg(), TrackMeta(title="t", difficulties=["ExpertPlus"]),
            "m1", tmp_path, {"weights": DEFAULT_WEIGHTS})
    expected = (0.30 * 1.0 + 0.20 * 0.8 + 0.17 * 0.5 + 0.15 * 0.2 + 0.18 * 0.6) / 1.0
    assert abs(res.final_score - expected) < 1e-6
    assert res.status == "ok" and res.gate_passed is True
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in data["subscores"]] == [
        "playability", "sync", "density", "mod_usage", "coverage"]
    assert data["model"] == "m1"


def test_subscore_details_persisted_in_results_json(tmp_path):
    with (
        patch.object(aggregate_mod, "validate_playability",
                     return_value=Subscore("playability", 0.9,
                                           {"reset_rate": 0.25})),
        patch.object(aggregate_mod, "validate_sync",
                     return_value=Subscore("sync", 1.0,
                                           {"median_deviation_ms": 12.5})),
        patch.object(aggregate_mod, "validate_density",
                     return_value=Subscore("density", 0.7,
                                           {"target_nps": 2.5,
                                            "quiet_ratio": 0.1})),
        patch.object(aggregate_mod, "validate_mod_usage",
                     return_value=Subscore("mod_usage", 0.4)),
        patch.object(aggregate_mod, "validate_coverage",
                     return_value=Subscore(
                         "coverage", 0.95,
                         {"coverage_per_file": {"ExpertPlus.dat": 0.95},
                          "audio_duration_sec": 10.0})),
    ):
        res = run_scoring(valid_pkg(),
                          TrackMeta(title="t", difficulties=["ExpertPlus"]),
                          "m1", tmp_path, {"weights": DEFAULT_WEIGHTS})

    assert res.gate_passed is True
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    by_name = {s["name"]: s for s in data["subscores"]}
    assert by_name["playability"]["details"] == {"reset_rate": 0.25}
    assert by_name["sync"]["details"] == {"median_deviation_ms": 12.5}
    assert by_name["density"]["details"] == {"target_nps": 2.5,
                                             "quiet_ratio": 0.1}
    assert by_name["coverage"]["details"]["coverage_per_file"] == {
        "ExpertPlus.dat": 0.95}
    for entry in data["subscores"]:
        assert set(entry) >= {"name", "score", "details"}


def test_final_normalized_by_used_weights(tmp_path):
    with patch("beatbench.scoring.aggregate._run_validators",
               return_value={"playability": Subscore("playability", 1.0),
                             "sync": Subscore("sync", 0.5)}):
        res = run_scoring(
            valid_pkg(), TrackMeta(title="t"), "m1", tmp_path,
            {"weights": {**DEFAULT_WEIGHTS, "density": 9.9, "mod_usage": 9.9}})
    expected = (0.30 * 1.0 + 0.20 * 0.5) / 0.50
    assert abs(res.final_score - expected) < 1e-6


def test_song_stem_threaded_through(tmp_path):
    with patch("beatbench.scoring.aggregate._run_validators",
               return_value=_fake_scores()):
        res = run_scoring(valid_pkg(), TrackMeta(title="t", difficulties=["ExpertPlus"]),
                          "m1", tmp_path, {"weights": DEFAULT_WEIGHTS}, song_stem="track")
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res.song_stem == "track"
    assert data["song"] == "track"


def test_song_stem_defaults_to_audio_path_stem(tmp_path):
    res = run_scoring(valid_pkg(broken=True), TrackMeta(title="t"), "m1",
                      tmp_path, {"weights": DEFAULT_WEIGHTS})
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res.song_stem == "song"
    assert data["song"] == "song"


def test_precomputed_gate_errors_skip_second_validation(tmp_path):
    from unittest.mock import patch as mock_patch

    with mock_patch("beatbench.scoring.aggregate.check_schema") as mocked:
        res = run_scoring(valid_pkg(broken=True), TrackMeta(title="t"), "m1",
                          tmp_path, {"weights": DEFAULT_WEIGHTS},
                          gate_errors=["Info.dat: bad shape"])
    mocked.assert_not_called()
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res.status == "invalid"
    assert data["errors"] == ["Info.dat: bad shape"]


def test_run_validators_calls_all_subvalidators():
    from beatbench.scoring import aggregate
    from beatbench.types import Subscore
    pkg = valid_pkg()
    meta = TrackMeta(title="t", difficulties=["ExpertPlus"])
    with (
        patch.object(aggregate, "validate_playability", return_value=Subscore("playability", 1.0)) as p,
        patch.object(aggregate, "validate_sync", return_value=Subscore("sync", 0.9)) as s,
        patch.object(aggregate, "validate_density",
                     return_value=Subscore("density", 0.5,
                                           {"target_nps": 9.0})) as d,
        patch.object(aggregate, "validate_mod_usage", return_value=Subscore("mod_usage", 0.3)) as m,
        patch.object(aggregate, "validate_coverage", return_value=Subscore("coverage", 0.7)) as c,
    ):
        scores = aggregate._run_validators(pkg, meta)
    p.assert_called_once_with(pkg)
    s.assert_called_once_with(pkg)
    m.assert_called_once_with(pkg)
    c.assert_called_once_with(pkg, None)
    d.assert_called_once_with(pkg, "ExpertPlus", pkg.info["beatsPerMinute"],
                              target_nps=meta.nps)
    assert scores == {
        "playability": Subscore("playability", 1.0),
        "sync": Subscore("sync", 0.9),
        "density": Subscore("density", 0.5, {"target_nps": 9.0}),
        "mod_usage": Subscore("mod_usage", 0.3),
        "coverage": Subscore("coverage", 0.7)}


def test_run_validators_passes_meta_nps_to_density():
    from beatbench.scoring import aggregate
    from beatbench.types import Subscore
    pkg = valid_pkg()
    meta = TrackMeta(title="t", difficulties=["ExpertPlus"], nps=2.5)
    with (
        patch.object(aggregate, "validate_playability", return_value=Subscore("playability", 1.0)),
        patch.object(aggregate, "validate_sync", return_value=Subscore("sync", 0.9)),
        patch.object(aggregate, "validate_density", return_value=Subscore("density", 0.5)) as d,
        patch.object(aggregate, "validate_mod_usage", return_value=Subscore("mod_usage", 0.3)),
        patch.object(aggregate, "validate_coverage", return_value=Subscore("coverage", 0.7)),
    ):
        aggregate._run_validators(pkg, meta)
    d.assert_called_once_with(pkg, "ExpertPlus", pkg.info["beatsPerMinute"],
                              target_nps=2.5)


def test_run_validators_passes_duration_to_coverage():
    from beatbench.scoring import aggregate
    from beatbench.types import Subscore
    pkg = valid_pkg()
    meta = TrackMeta(title="t", difficulties=["ExpertPlus"])
    with (
        patch.object(aggregate, "validate_playability", return_value=Subscore("playability", 1.0)),
        patch.object(aggregate, "validate_sync", return_value=Subscore("sync", 1.0)),
        patch.object(aggregate, "validate_mod_usage", return_value=Subscore("mod_usage", 1.0)),
        patch.object(aggregate, "validate_density", return_value=Subscore("density", 0.5)),
        patch.object(aggregate, "_audio_duration", return_value=123.0) as dur,
        patch.object(aggregate, "validate_coverage", return_value=Subscore("coverage", 0.5)) as cov,
    ):
        aggregate._run_validators(pkg, meta)
    dur.assert_called_once_with(pkg)
    cov.assert_called_once_with(pkg, 123.0)
