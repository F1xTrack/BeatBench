import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import uuid

SCORE_KEYS = ("musicality", "phrase_response", "density_and_rest", "playability", "presentation")
SCORE_WEIGHTS = {"musicality": 0.25, "phrase_response": 0.25, "density_and_rest": 0.15,
                 "playability": 0.20, "presentation": 0.15}
MIN_SCORE = 0
MAX_SCORE = 100

CORE_AXES = ("musicality", "phrase_response", "density_and_rest", "playability")


def collect_ratings(result_dir: Path, answers: dict) -> Path:
    scores = {}
    for key in SCORE_KEYS:
        value = int(answers[key])
        if not MIN_SCORE <= value <= MAX_SCORE:
            raise ValueError(f"{key} must be an integer between {MIN_SCORE} and {MAX_SCORE}, got {value}")
        scores[key] = value

    weighted = sum(scores[key] * SCORE_WEIGHTS[key] for key in SCORE_KEYS)
    caps = []
    if scores["musicality"] <= 20 or scores["phrase_response"] <= 20:
        caps.append({"reason": "critical_music_or_phrase_failure", "maximum": 39})
    if sum(scores[key] < 40 for key in CORE_AXES) >= 2:
        caps.append({"reason": "multiple_weak_core_axes", "maximum": 49})
    overall = min([weighted] + [cap["maximum"] for cap in caps])
    primary_maps = [p for p in result_dir.glob("*.zip") if not p.stem.endswith("-v2")]
    source_map = primary_maps[0] if len(primary_maps) == 1 else None
    source_sha256 = None
    if source_map is not None:
        with source_map.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "rubric_version": "human-quality-v1.0",
        "score_kind": "single_rater_absolute_quality_0_100",
        "calibration_status": "single_rater_unvalidated",
        "scores": scores,
        "weights": SCORE_WEIGHTS,
        "weighted_score_before_caps": round(weighted, 2),
        "caps": caps,
        "overall": overall,
        "overall_score_100": round(overall, 2),
        "scale": {"minimum": MIN_SCORE, "maximum": MAX_SCORE},
        "source_map": str(source_map.resolve()) if source_map else None,
        "source_map_sha256": source_sha256,
        "source_binding_status": "bound_to_primary_zip" if source_sha256 else "unbound_no_unique_primary_zip",
        "comment": str(answers.get("comment", "")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    rating_root = result_dir / "human-ratings"
    rating_root.mkdir(parents=True, exist_ok=True)
    rating_dir = rating_root / (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    rating_dir.mkdir(exist_ok=False)
    out_path = rating_dir / "rating.json"
    with out_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return out_path
