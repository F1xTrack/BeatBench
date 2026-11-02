"""Zip packaging of a GeneratedPackage into an uploadable BeatSaber map."""
from __future__ import annotations

import copy
import json
import re
import zipfile
from pathlib import Path

from ..types import GeneratedPackage

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_folder(name: str) -> str:
    cleaned = _ILLEGAL_CHARS.sub("_", name).rstrip(". ")
    return cleaned or "_"


def _valid_saber_colors(value) -> bool:
    return (isinstance(value, dict)
            and isinstance(value.get("left"), list)
            and isinstance(value.get("right"), list))


def build_zip(pkg: GeneratedPackage, out_dir: Path, stem: str) -> Path:
    folder = sanitize_folder(
        f"{pkg.info.get('songAuthorName') or 'Unknown Artist'}"
        f" - {pkg.info['songName']}")
    info = copy.deepcopy(pkg.info)
    saber_colors = info.pop("saberColors", None)
    scene_colors = info.pop("sceneColors", None)
    zip_path = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{folder}/Info.dat", json.dumps(info))
        if _valid_saber_colors(saber_colors):
            z.writestr(
                f"{folder}/saber_colors.json",
                json.dumps({"left": saber_colors["left"],
                            "right": saber_colors["right"]}))
        if _valid_saber_colors(scene_colors):
            z.writestr(f"{folder}/scene_colors.json", json.dumps(scene_colors))
        for name, dat in pkg.difficulties.items():
            z.writestr(f"{folder}/{name}", json.dumps(dat))
        z.write(pkg.audio_path,
                f"{folder}/{pkg.info.get('songFilename') or 'song.ogg'}")
        if pkg.cover_path:
            z.write(pkg.cover_path, f"{folder}/{pkg.info['coverImageFilename']}")
    return zip_path
