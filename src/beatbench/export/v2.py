"""V3 → V2 map conversion and Quest-ready V2 zip packaging."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from ..types import GeneratedPackage
from .colors import apply_palette
from .packager import _valid_saber_colors, sanitize_folder

CUT_DIRECTION_OK = {0, 1, 2, 3, 4, 5, 6, 7, 8}

_DIFFICULTY_RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}


def convert_info_v3_to_v2(info: dict) -> dict:
    sets_out = []
    for s in info.get("difficultyBeatmapSets", []):
        beats = []
        for b in s.get("_difficultyBeatmaps", []):
            beats.append({
                "_difficulty": b.get("_difficulty", "ExpertPlus"),
                "_difficultyRank": b.get(
                    "_difficultyRank",
                    _DIFFICULTY_RANK.get(b.get("_difficulty", "ExpertPlus"), 9)),
                "_beatmapFilename": b.get("_beatmapFilename", "ExpertPlus.dat"),
                "_noteJumpMovementSpeed": b.get("_noteJumpMovementSpeed", 18),
                "_noteJumpStartBeatOffset": b.get("_noteJumpStartBeatOffset", 0),
                "_beatmapColorSchemeIdx": b.get("_beatmapColorSchemeIdx", -1),
                "_customData": copy.deepcopy(b.get("_customData", {})),
            })
        sets_out.append({
            "_beatmapCharacteristicName": s.get("_beatmapCharacteristicName", "Standard"),
            "_difficultyBeatmaps": beats,
        })
    bpm = info.get("beatsPerMinute", 120)
    return {
        "_version": "2.1.0",
        "_songName": info.get("songName", ""),
        "_songSubName": info.get("songSubName", ""),
        "_songAuthorName": info.get("songAuthorName", ""),
        "_levelAuthorName": info.get("levelAuthorName", ""),
        "_beatsPerMinute": bpm,
        "_shuffle": info.get("shuffle", 0.0),
        "_shufflePeriod": info.get("shufflePeriod", 0.0),
        "_previewStartTime": max(0.0, float(info.get("previewStartTime", 15.0))),
        "_previewDuration": float(info.get("previewDuration", 10.0)),
        "_songFilename": info.get("songFilename", "song.ogg"),
        "_coverImageFilename": info.get("coverImageFilename", "cover.png"),
        "_environmentName": info.get("environmentName", "DefaultEnvironment"),
        "_songTimeOffset": info.get("songTimeOffset", 0.0),
        "_colorSchemes": copy.deepcopy(info.get("colorSchemes", [])),
        "_difficultyBeatmapSets": sets_out,
    }


def convert_difficulty_v3_to_v2(dat: dict) -> dict:
    if dat.get("sliders") or dat.get("burstSliders"):
        raise ValueError(
            "V3 arcs (sliders) and chains (burstSliders) cannot be represented exactly in V2; use the V3 map"
        )
    notes = []
    for n in dat.get("colorNotes", []):
        notes.append({
            "_time": float(n["b"]),
            "_lineIndex": int(n["x"]),
            "_lineLayer": int(n["y"]),
            "_type": 0 if int(n["c"]) == 0 else 1,
            "_cutDirection": int(n["d"]) if int(n["d"]) in CUT_DIRECTION_OK else 1,
        })
    for b in dat.get("bombNotes", []):
        notes.append({
            "_time": float(b["b"]),
            "_lineIndex": int(b["x"]),
            "_lineLayer": int(b["y"]),
            "_type": 3,
            "_cutDirection": 0,
        })
    notes.sort(key=lambda n: (n["_time"], n["_lineIndex"], n["_lineLayer"]))
    events = [
        {"_time": float(e["b"]), "_type": int(e["et"]), "_value": int(e["i"]),
         "_floatValue": float(e.get("f", 1.0))}
        for e in dat.get("basicBeatmapEvents", [])
    ]
    obstacles = []
    for o in dat.get("obstacles", []):
        shape = (o.get("y", 0), o.get("h", 5))
        if shape not in ((0, 5), (2, 3)):
            raise ValueError("This V3 wall geometry cannot be represented exactly in V2; use the V3 map")
        obstacles.append({
            "_time": float(o["b"]),
            "_lineIndex": int(o["x"]),
            "_type": 0 if shape == (0, 5) else 1,
            "_duration": float(o["d"]),
            "_width": int(o.get("w", 1)),
        })
    return {
        "_version": "2.1.0",
        "_notes": notes,
        "_obstacles": obstacles,
        "_events": events,
    }


def load_saber_colors(path: Path) -> tuple[list[float], list[float]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def norm(raw) -> list[float] | None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        vals = []
        for v in raw:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return None
            if v < 0 or v > 255:
                return None
            vals.append(float(v))
        if any(v > 1 for v in vals):
            vals = [v / 255 for v in vals]
        return [*vals, 1.0]

    left = norm(data.get("left"))
    right = norm(data.get("right"))
    if left is None or right is None:
        return None
    return left, right


def convert_folder(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "Info.dat").read_text(encoding="utf-8-sig"))
    colors_path = src / "saber_colors.json"
    scene_path = src / "scene_colors.json"
    saber = json.loads(colors_path.read_text("utf-8-sig")) if colors_path.exists() else info.get("saberColors")
    scene = json.loads(scene_path.read_text("utf-8-sig")) if scene_path.exists() else info.get("sceneColors")
    apply_palette(info, saber, scene)
    out_info = convert_info_v3_to_v2(info)
    (dst / "Info.dat").write_text(json.dumps(out_info), encoding="utf-8")
    for f in ("song.ogg", "cover.png", "cover.jpg"):
        p = src / f
        if p.exists():
            (dst / f).write_bytes(p.read_bytes())
    for ref, _fallback in _beatmap_references(info):
        dat = json.loads((src / ref).read_text(encoding="utf-8-sig"))
        (dst / ref).write_text(
            json.dumps(convert_difficulty_v3_to_v2(dat)), encoding="utf-8")


def _beatmap_references(info: dict) -> list[tuple[str, str]]:
    refs = []
    for s in info.get("difficultyBeatmapSets", []):
        for b in s.get("_difficultyBeatmaps", []):
            difficulty = b.get("_difficulty", "ExpertPlus")
            ref = b.get("_beatmapFilename") or f"{difficulty}.dat"
            refs.append((ref, f"{difficulty}.dat"))
    return refs


def build_v2_zip(pkg: GeneratedPackage, out_dir: Path, stem: str) -> Path:
    folder = sanitize_folder(
        f"{pkg.info.get('songAuthorName') or 'Unknown Artist'}"
        f" - {pkg.info['songName']}")
    audio_name = pkg.info.get("songFilename") or "song.ogg"
    cover_name = pkg.info.get("coverImageFilename")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "src"
        src.mkdir()
        info = copy.deepcopy(pkg.info)
        saber_colors = info.pop("saberColors", None)
        scene_colors = info.pop("sceneColors", None)
        (src / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
        if _valid_saber_colors(saber_colors):
            (src / "saber_colors.json").write_text(
                json.dumps({"left": saber_colors["left"],
                            "right": saber_colors["right"]}), encoding="utf-8")
        if _valid_saber_colors(scene_colors):
            (src / "scene_colors.json").write_text(json.dumps(scene_colors), encoding="utf-8")
        for name, dat in pkg.difficulties.items():
            (src / name).write_text(json.dumps(dat), encoding="utf-8")
        for ref, fallback_name in _beatmap_references(info):
            target = src / ref
            if target.exists():
                continue
            payload = pkg.difficulties.get(ref)
            if payload is None:
                payload = pkg.difficulties.get(fallback_name)
            if payload is not None:
                target.write_text(json.dumps(payload), encoding="utf-8")
        shutil.copyfile(pkg.audio_path, src / audio_name)
        if pkg.cover_path and cover_name:
            shutil.copyfile(pkg.cover_path, src / cover_name)

        dst = tmp_dir / "dst"
        convert_folder(src, dst)
        for extra_name in (audio_name, cover_name):
            if extra_name and extra_name != "Info.dat":
                extra_src = src / extra_name
                extra_dst = dst / extra_name
                if extra_src.exists() and not extra_dst.exists():
                    shutil.copyfile(extra_src, extra_dst)

        zip_path = out_dir / f"{stem}-v2.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dst.rglob("*")):
                if p.is_file():
                    z.write(p, f"{folder}/{p.relative_to(dst).as_posix()}")
    return zip_path
