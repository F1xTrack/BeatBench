"""Benchmark configuration loaded from beatbench.toml next to CWD."""
from __future__ import annotations
import tomllib
from pathlib import Path

DEFAULT_WEIGHTS = {
    "playability": 0.30,
    "sync": 0.20,
    "density": 0.17,
    "mod_usage": 0.15,
    "coverage": 0.18,
}
MAX_AUDIO_SECONDS = 240.0
SYNC_TOLERANCE_MS = 50.0
DEAD_ZONE_SECONDS = 4.0

TARGET_NPS = {
    "Easy": 1.0, "Normal": 2.0, "Hard": 3.5,
    "Expert": 5.5, "ExpertPlus": 8.0,
}


def load_config(cwd: Path | None = None) -> dict:
    path = (cwd or Path.cwd()) / "beatbench.toml"
    cfg: dict = {"weights": {**DEFAULT_WEIGHTS}}
    if path.exists():
        user = tomllib.loads(path.read_text(encoding="utf-8"))
        if "scoring" in user and "weights" in user["scoring"]:
            cfg["weights"] = {**DEFAULT_WEIGHTS, **user["scoring"]["weights"]}
    return cfg
