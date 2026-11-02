"""Encode model palettes as native schemes and per-difficulty SongCore overrides."""
from __future__ import annotations

import copy
import math


def rgb255(raw) -> dict | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           or v < 0 or v > 255 for v in raw):
        return None
    return {**dict(zip(("r", "g", "b"), (float(v) / 255 for v in raw))), "a": 1.0}


def apply_palette(info: dict, saber: dict | None, scene: dict | None) -> bool:
    """Mutate only metadata. Values come from the model, never a track-name rule."""
    saber = saber if isinstance(saber, dict) else {}
    scene = scene if isinstance(scene, dict) else {}
    left, right = rgb255(saber.get("left")), rgb255(saber.get("right"))
    scene_left = rgb255(scene.get("left")) or left
    scene_right = rgb255(scene.get("right")) or right
    if not all((left, right, scene_left, scene_right)):
        return False
    boost_left = rgb255(scene.get("leftBoost")) or scene_left
    boost_right = rgb255(scene.get("rightBoost")) or scene_right
    walls = rgb255(scene.get("obstacles")) or scene_left
    scheme = {"colorSchemeId": "BeatBench", "saberAColor": left, "saberBColor": right,
              "environmentColor0": scene_left, "environmentColor1": scene_right,
              "environmentColor0Boost": boost_left, "environmentColor1Boost": boost_right,
              "obstaclesColor": walls}
    info["colorSchemes"] = [{"useOverride": True, "colorScheme": scheme}]
    overrides = {"_colorLeft": left, "_colorRight": right, "_envColorLeft": scene_left,
                 "_envColorRight": scene_right, "_envColorLeftBoost": boost_left,
                 "_envColorRightBoost": boost_right, "_obstacleColor": walls}
    for group in info.get("difficultyBeatmapSets", []):
        for entry in group.get("_difficultyBeatmaps", []):
            entry["_beatmapColorSchemeIdx"] = 0
            entry["_customData"] = {**entry.get("_customData", {}), **copy.deepcopy(overrides)}
    return True
