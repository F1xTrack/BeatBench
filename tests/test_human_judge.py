import json
from datetime import datetime
from pathlib import Path

import pytest

from beatbench.judges.human import collect_ratings


AXES = ("musicality", "phrase_response", "density_and_rest", "playability", "presentation")


def _answers(value=50):
    return {key: value for key in AXES}


def test_collect_ratings_writes_bound_timestamped_rating(tmp_path: Path):
    answers = {**dict(zip(AXES, (80, 60, 70, 90, 50))), "comment": "nice map"}
    out = collect_ratings(tmp_path, answers)

    assert out.parent.parent == tmp_path / "human-ratings"
    assert out.name == "rating.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scores"] == dict(zip(AXES, (80, 60, 70, 90, 50)))
    assert data["overall_score_100"] == pytest.approx(71.0)
    assert data["comment"] == "nice map"


def test_collect_ratings_comment_defaults_to_empty(tmp_path: Path):
    out = collect_ratings(tmp_path, _answers())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["comment"] == ""


def test_collect_ratings_out_of_range_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        collect_ratings(tmp_path, {**_answers(), "musicality": 101})


def test_collect_ratings_timestamp_is_iso8601_utc(tmp_path: Path):
    out = collect_ratings(tmp_path, _answers())
    data = json.loads(out.read_text(encoding="utf-8"))
    parsed = datetime.fromisoformat(data["timestamp"])
    assert parsed.tzinfo is not None
