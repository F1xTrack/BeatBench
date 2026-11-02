"""Track metadata from sidecar song.yaml and/or CLI flag overrides."""
from __future__ import annotations
from pathlib import Path
import yaml
from .types import TrackMeta

DIFFICULTY_ALIASES = {
    "easy": "Easy", "normal": "Normal", "hard": "Hard",
    "expert": "Expert", "expertplus": "ExpertPlus", "expert+": "ExpertPlus",
}


def load_meta(audio_path: Path, overrides: dict) -> TrackMeta:
    sidecar = audio_path.with_suffix(".yaml")
    data: dict = {}
    if sidecar.exists():
        data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    merged = {
        "title": data.get("title"),
        "author": data.get("author"),
        "bpm": data.get("bpm"),
        "nps": data.get("nps"),
        "notes": data.get("notes", ""),
        "difficulties": data.get("difficulties"),
        **{k: v for k, v in overrides.items() if v is not None},
    }
    diffs = merged["difficulties"] or ["ExpertPlus"]
    diffs = [DIFFICULTY_ALIASES.get(str(d).lower(), d) for d in diffs]
    return TrackMeta(
        title=merged["title"] or audio_path.stem,
        author=merged["author"] or "Unknown",
        bpm=merged["bpm"],
        nps=merged["nps"],
        difficulties=diffs,
        notes=merged["notes"],
    )
