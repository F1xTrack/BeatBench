"""Read existing V2/V3 maps without changing their files or legacy results."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import uuid
import zipfile

from .review import review_package
from .assessment import assess_package
from ..types import GeneratedPackage


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _reference(root: Path, name: str) -> Path:
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError(f"Unsafe map reference: {name!r}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Map reference outside source: {name!r}")
    if not path.is_file():
        raise ValueError(f"Missing referenced file: {name!r}")
    return path


def _normalize(data: dict) -> dict:
    version = str(data.get("version", data.get("_version", "")))
    if version.startswith("3."):
        return data
    if not version.startswith("2."):
        raise ValueError(f"Unsupported beatmap version: {version!r}; expected V2/V3")
    notes = data.get("_notes", [])
    if any(n.get("_type") not in (0, 1, 3) for n in notes):
        raise ValueError("Unsupported V2 note type")
    return {
        "version": "3.0.0", "sourceVersion": version,
        "colorNotes": [{"b": n["_time"], "x": n["_lineIndex"], "y": n["_lineLayer"],
                        "c": n["_type"], "d": n["_cutDirection"], "a": 0,
                        "customData": n.get("_customData", {})} for n in notes if n["_type"] in (0, 1)],
        "bombNotes": [{"b": n["_time"], "x": n["_lineIndex"], "y": n["_lineLayer"]} for n in notes if n["_type"] == 3],
        "obstacles": [{"b": o["_time"], "x": o["_lineIndex"], "y": 0 if o["_type"] == 0 else 2,
                       "d": o["_duration"], "w": o["_width"], "h": 5 if o["_type"] == 0 else 3,
                       "customData": o.get("_customData", {})} for o in data.get("_obstacles", [])],
        "basicBeatmapEvents": [{"b": e["_time"], "et": e["_type"], "i": e["_value"],
                               "f": e.get("_floatValue", 1), "customData": e.get("_customData", {})}
                              for e in data.get("_events", [])],
        "customData": data.get("_customData", {}), "_BPMChanges": data.get("_BPMChanges", []),
    }


def _load(info_path: Path):
    raw = _json(info_path)
    version = str(raw.get("version", raw.get("_version", "")))
    if not version.startswith(("2.", "3.")):
        raise ValueError(f"Unsupported Info version {version!r}")
    info = {key[1:] if key.startswith("_") else key: value for key, value in raw.items()}
    root = info_path.parent
    audio = _reference(root, info["songFilename"])
    sources = [info_path, audio]
    difficulties = {}
    skipped = []
    for group in info.get("difficultyBeatmapSets", []):
        characteristic = group.get("_beatmapCharacteristicName", group.get("beatmapCharacteristicName"))
        if characteristic != "Standard":
            skipped.append(characteristic)
            continue
        for entry in group.get("_difficultyBeatmaps", group.get("difficultyBeatmaps", [])):
            filename = entry.get("_beatmapFilename", entry.get("beatmapFilename"))
            path = _reference(root, filename)
            difficulties[filename] = _normalize(_json(path))
            sources.append(path)
    if not difficulties:
        raise ValueError("No supported Standard difficulties")
    for name, data in difficulties.items():
        for index, n in enumerate(data.get("colorNotes", [])):
            if not all(isinstance(n.get(k), (int, float)) and not isinstance(n.get(k), bool)
                       and math.isfinite(n[k]) for k in ("b", "x", "y", "c", "d")):
                raise ValueError(f"{name}: malformed color note {index}")
    return GeneratedPackage(info, difficulties, audio, None), sources, skipped


def _player_rating(run_root: Path, source_sha256: str | None) -> dict | None:
    """Aggregate only player ratings explicitly bound to this exact primary ZIP."""
    if not source_sha256:
        return None
    root = run_root / "human-ratings"
    entries = []
    for path in root.glob("*/rating.json"):
        try:
            value = _json(path)
            if value.get("source_map_sha256") != source_sha256 or value.get("rubric_version") != "human-quality-v1.0":
                continue
            scores = value.get("scores", {})
            if any(not isinstance(scores.get(key), (int, float)) or not 0 <= scores[key] <= 100
                   for key in ("musicality", "phrase_response", "density_and_rest", "playability", "presentation")):
                continue
            if not isinstance(value.get("overall_score_100"), (int, float)) or not 0 <= value["overall_score_100"] <= 100:
                continue
            entries.append(value)
        except (OSError, ValueError, TypeError):
            continue
    if not entries:
        return None
    keys = ("musicality", "phrase_response", "density_and_rest", "playability", "presentation")
    mean = lambda key: round(sum(float(entry[key]) for entry in entries) / len(entries), 2)
    axis_means = {key: round(sum(float(entry["scores"][key]) for entry in entries) / len(entries), 2)
                  for key in keys}
    return {"rubric_version": "human-quality-v1.0", "score_kind": "player_rated_absolute_quality_0_100",
            "calibration_status": "single_rater_unvalidated" if len(entries) == 1 else "repeat_ratings_unvalidated",
            "rating_count": len(entries), "scores": axis_means,
            "overall_score_100": mean("overall_score_100"), "weights": entries[-1].get("weights"),
            "source_map_sha256": source_sha256,
            "rating_records": [entry.get("timestamp") for entry in entries]}


@contextmanager
def _map_source(source: Path):
    archive = source
    if source.is_dir():
        infos = [p for p in source.iterdir() if p.is_file() and p.name.lower() == "info.dat"]
        if infos:
            yield infos[0], {"kind": "directory", "path": str(source)}, source
            return
        candidates = [p for p in source.glob("*.zip") if not p.stem.endswith("-v2")]
        if len(candidates) != 1:
            raise ValueError("Provide a map directory or one explicit ZIP (ambiguous run directory)")
        archive = candidates[0]
    elif source.name.lower() == "info.dat":
        yield source, {"kind": "directory", "path": str(source.parent)}, source.parent
        return
    if archive.suffix.lower() != ".zip":
        raise ValueError("Provide a ZIP, Info.dat, map directory or run directory")
    before = _sha(archive)
    with tempfile.TemporaryDirectory(prefix="beatbench-review-") as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > 10000 or sum(m.file_size for m in members) > 512 * 1024**2:
                raise ValueError("Map archive exceeds review limits")
            seen = set()
            for member in members:
                name = member.filename.replace("\\", "/")
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name:
                    raise ValueError(f"Unsafe archive member: {member.filename!r}")
                target = root.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(root) or str(target).casefold() in seen:
                    raise ValueError("Duplicate or unsafe archive member")
                seen.add(str(target).casefold())
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as stream, target.open("xb") as output:
                        while chunk := stream.read(1024 * 1024):
                            output.write(chunk)
        infos = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == "info.dat"]
        if len(infos) != 1:
            raise ValueError("Archive must contain exactly one Info.dat")
        yield infos[0], {"kind": "zip", "path": str(archive), "sha256": before}, archive.parent
    if _sha(archive) != before:
        raise RuntimeError("Source ZIP changed during review")


def review_source(source: Path, output_root: Path, target_nps: float | None = None,
                  analyze_audio: bool = True) -> tuple[dict, Path]:
    source = source.resolve()
    with _map_source(source) as (info_path, origin, run_root):
        pkg, files, skipped = _load(info_path)
        hashes = {p.relative_to(info_path.parent).as_posix(): _sha(p) for p in files}
        review = review_package(pkg, target_nps, analyze_audio=analyze_audio)
        review["assessment"] = assess_package(pkg, review, target_nps)
        if any(_sha(p) != hashes[p.relative_to(info_path.parent).as_posix()] for p in files):
            raise RuntimeError("Source map files changed during review")
        review.update({"recorded_at": datetime.now(timezone.utc).isoformat(), "source": origin,
                       "diagnostic_code_sha256": {"review.py": _sha(Path(__file__).with_name("review.py")),
                                                  "review_io.py": _sha(Path(__file__))},
                       "source_files_sha256": hashes, "source_files_unchanged": True,
                       "skipped_characteristics": skipped, "song": pkg.info.get("songName"),
                       "mapper": pkg.info.get("levelAuthorName")})
        player_rating = _player_rating(run_root, origin.get("sha256"))
        review["player_rating"] = player_rating
        if player_rating:
            review["assessment"]["quality_score"] = player_rating["overall_score_100"]
            review["assessment"]["quality_score_source"] = "player_rubric"
            review["assessment"]["calibration_status"] = player_rating["calibration_status"]
        legacy = run_root / "results.json"
        if legacy.exists():
            data = _json(legacy)
            review["legacy_proxy"] = {"value": data.get("final"), "calibrated": False,
                                      "source": str(legacy), "sha256": _sha(legacy)}
    output_root = output_root.resolve()
    if ".docs" in output_root.parts and "history" in output_root.parts:
        raise ValueError("Review output cannot be placed in append-only goal history")
    directory = output_root / (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "review.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return review, path
