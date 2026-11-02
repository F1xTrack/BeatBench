"""Result aggregation: schema gate, weighted subscore mean, results.json output."""
from __future__ import annotations
import json
import librosa
from pathlib import Path

from .coverage import validate_coverage
from .density import validate_density
from .mod_usage import validate_mod_usage
from .playability import validate_playability
from .schema_check import check_schema
from .sync import validate_sync
from .review import review_package
from .assessment import VERSION, assess_package
from ..types import GeneratedPackage, RunResult, Subscore, TrackMeta


def _audio_duration(pkg: GeneratedPackage) -> float | None:
    try:
        return float(librosa.get_duration(path=str(pkg.audio_path)))
    except Exception:
        return None


def _run_validators(
    pkg: GeneratedPackage,
    meta: TrackMeta,
    audio_duration_sec: float | None = None,
) -> dict[str, Subscore]:
    bpm = float(pkg.info.get("beatsPerMinute", 0) or 0)
    duration = audio_duration_sec
    if duration is None:
        duration = _audio_duration(pkg)
    scores: dict[str, Subscore] = {
        "playability": validate_playability(pkg),
        "sync": validate_sync(pkg),
        "mod_usage": validate_mod_usage(pkg),
        "coverage": validate_coverage(pkg, duration),
    }
    density_scores = [
        validate_density(pkg, name, bpm, target_nps=meta.nps)
        for name in meta.difficulties
    ]
    density_score = sum(s.score for s in density_scores) / len(density_scores)
    if len(density_scores) == 1:
        density_details = density_scores[0].details
    else:
        density_details = {
            name: s.details
            for name, s in zip(meta.difficulties, density_scores)
        }
    scores["density"] = Subscore(
        name="density",
        score=density_score,
        details=density_details,
    )
    return scores


def _weighted_final(scores: dict[str, float], weights: dict[str, float]) -> float:
    used = {n: weights[n] for n in scores if n in weights and weights[n] > 0}
    total_w = sum(used.values())
    if total_w == 0:
        return 0.0
    return sum(w * scores[n] for n, w in used.items()) / total_w


def run_scoring(
    pkg: GeneratedPackage,
    meta: TrackMeta,
    model_name: str,
    out_dir: Path,
    cfg: dict,
    song_stem: str | None = None,
    gate_errors: list[str] | None = None,
    generation_prompt: dict | None = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = song_stem if song_stem is not None else Path(pkg.audio_path).stem

    if gate_errors is None:
        gate_passed, errors = check_schema(pkg)
    else:
        gate_passed, errors = not gate_errors, list(gate_errors)

    if not gate_passed:
        result = RunResult(
            model=model_name,
            song_stem=stem,
            gate_passed=False,
            status="invalid",
            errors=errors,
            final_score=None,
        )
        _write_results(out_dir, result, generation_prompt)
        return result

    scores = _run_validators(pkg, meta)
    subscores = [
        scores[name]
        for name in ("playability", "sync", "density", "mod_usage", "coverage")
        if name in scores
    ]
    final = _weighted_final({s.name: s.score for s in subscores}, cfg.get("weights", {}))
    diagnostics = review_package(pkg, target_nps=meta.nps)
    result = RunResult(
        model=model_name,
        song_stem=stem,
        gate_passed=True,
        subscores=subscores,
        final_score=final,
        status="ok",
        diagnostics=diagnostics,
        assessment=assess_package(pkg, diagnostics, target_nps=meta.nps),
    )
    _write_results(out_dir, result, generation_prompt)
    return result


def _write_results(out_dir: Path, result: RunResult, generation_prompt: dict | None = None) -> None:
    payload = {
        "model": result.model,
        "song": result.song_stem,
        "status": result.status,
        "gate_passed": result.gate_passed,
        "errors": list(result.errors),
        "subscores": [
            {"name": s.name, "score": s.score, "details": s.details} for s in result.subscores
        ],
        "final": result.final_score,
        "scoring_version": "legacy-v1",
        "score_kind": "uncalibrated_heuristic",
        "quality_score": None,
        "calibration_status": "uncalibrated",
        "diagnostics": result.diagnostics,
        "primary_scoring_version": VERSION,
        "experimental_score": (result.assessment or {}).get("experimental_score"),
        "assessment": result.assessment,
        "generation_prompt": generation_prompt,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
