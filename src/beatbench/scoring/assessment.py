"""Versioned, model-blind experimental scores; validated gameplay ratings stay separate."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from .review import (_ANGLES, _BeatClock, _audio_features, _distances, _geometry_is_unknown,
                     _playable_hits, _same_swing)
from ..types import GeneratedPackage

VERSION = "experimental-v6.0"
WEIGHTS = {"musicality": 0.25, "movement": 0.15, "pattern_development": 0.25,
           "density_response": 0.15, "readability": 0.05, "presentation": 0.15}
# Initial engineering hypotheses, deliberately frozen and disclosed, not fitted to model identity.
PARAMETERS = {"fast_return_seconds": 0.6, "return_penalty": 1.8, "ambiguous_penalty": 0.35,
              "section_seconds": 4.0, "density_tolerance": 0.25,
              "density_null_shuffles": 64, "minimum_geometry_support": 0.80,
              "presentation_floor": 0.35}
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _clip(value):
    return float(np.clip(value, 0, 1))


def _axis(score, evidence, limits=(), confidence="limited"):
    return {"score": round(_clip(score) * 100, 2) if score is not None else None,
            "raw_proxy_0_1": round(_clip(score), 6) if score is not None else None,
            "confidence": confidence if score is not None else "unavailable",
            "evidence": evidence, "limitations": list(limits)}


def geometric_score(axes: dict, weights: dict = WEIGHTS) -> float | None:
    # Missing evidence never causes weight renormalization or a free pass.
    values = [axes.get(name, {}).get("score") for name in weights]
    if any(value is None for value in values):
        return None
    if any(value <= 0 for value in values):
        return 0.0
    return float(math.exp(sum(weights[name] * math.log(axes[name]["score"] / 100) for name in weights)) * 100)


def _groups(notes, clock):
    hands = []
    for color in (0, 1):
        groups = []
        for note in sorted((n for n in notes if n["c"] == color), key=lambda n: (n["b"], n["x"], n["y"])):
            if groups and _same_swing(groups[-1][-1], note, clock):
                groups[-1].append(note)
            else:
                groups.append([note])
        hands.append(groups)
    return hands


def _movement(notes, data, clock, trusted):
    if not trusted:
        return _axis(None, {}, ["Unsupported timebase, animation, rotation or arcs/chains."])
    groups = _groups(notes, clock)
    total = assessed = returns = 0
    burden = ambiguous = recovery = 0.0
    examples = []
    bombs = [clock.seconds(b["b"]) for b in data.get("bombNotes", [])]
    for hand, sequence in enumerate(groups):
        for previous, current in zip(sequence, sequence[1:]):
            total += 1
            a, b = previous[-1], current[0]
            if a["d"] not in _ANGLES or b["d"] not in _ANGLES:
                continue
            assessed += 1
            ta, tb = clock.seconds(a["b"]), clock.seconds(b["b"])
            gap = tb - ta
            delta = abs(_ANGLES[a["d"]] - _ANGLES[b["d"]])
            delta = min(delta, 360 - delta)
            # Bomb resets need contextual analysis; don't count them as a confirmed failure.
            reset_context = any(ta < t < tb for t in bombs)
            severity = 0.0
            if delta <= 45 and 0 < gap < PARAMETERS["fast_return_seconds"] and not reset_context:
                returns += 1
                severity = 1 - gap / PARAMETERS["fast_return_seconds"]
                burden += severity
            if delta == 90 and 0 < gap < PARAMETERS["fast_return_seconds"]:
                ambiguous += 1 - gap / PARAMETERS["fast_return_seconds"]
            elif delta == 90 and gap > 0:
                ambiguous += math.exp(-gap / 0.75)
            if gap > 0:
                va = np.array([math.cos(math.radians(_ANGLES[a["d"]])), math.sin(math.radians(_ANGLES[a["d"]]))])
                vb = np.array([math.cos(math.radians(_ANGLES[b["d"]])), math.sin(math.radians(_ANGLES[b["d"]]))])
                end = np.array([a["x"], a["y"]]) + 0.5 * va
                start = np.array([b["x"], b["y"]]) - 0.5 * vb
                # Plane-space recovery distance per second, using disclosed unit stroke lengths.
                recovery += 1 - math.exp(-float(np.linalg.norm(start - end)) / gap / 6)
            if (severity or delta == 90) and len(examples) < 20:
                examples.append({"hand": hand, "seconds": tb, "gap_seconds": gap,
                                 "direction_change_degrees": delta, "return_severity": severity})
    support = assessed / total if total else 0
    evidence = {"transitions": total, "assessed": assessed, "assessment_fraction": support,
                "fast_return_candidates": returns, "return_burden": burden / assessed if assessed else None,
                "ambiguous_burden": ambiguous / assessed if assessed else None, "examples": examples}
    evidence["planar_recovery_effort"] = recovery / assessed if assessed else None
    if assessed < 8 or support < PARAMETERS["minimum_geometry_support"]:
        return _axis(None, evidence, ["Too few supported directional transitions."])
    score = (1 - PARAMETERS["return_penalty"] * burden / assessed
             - PARAMETERS["ambiguous_penalty"] * ambiguous / assessed - 0.3 * recovery / assessed)
    return _axis(score, evidence, ["Planar stroke/recovery proxy only; assumes half-cell strokes and no physical wrist simulation.",
                                   "Bomb reset intervals are excluded; 90-degree turns are ambiguous, not automatically bad."])


def _musicality(times, features, diagnostic):
    evidence = diagnostic.get("timing", {})
    full = evidence.get("full", {})
    if len(times) < 8 or not full.get("available"):
        return _axis(None, evidence, ["Insufficient audible attack evidence."])
    # Keep the full-spectrum layer fixed. Never choose a different winning band per note.
    precision = full["within_50ms"]
    chance = full["shifted_baseline_mean_50ms"]
    excess = _clip((precision - chance) / max(1e-9, 1 - chance))
    attacks = features["attacks"]["full"]
    indexes = np.searchsorted(attacks, times)
    left = np.clip(indexes - 1, 0, len(attacks) - 1)
    right = np.clip(indexes, 0, len(attacks) - 1)
    chosen = np.where(abs(times - attacks[left]) <= abs(times - attacks[right]), attacks[left], attacks[right])
    interval_errors = abs(np.diff(times) - np.diff(chosen))
    phrase_support = float(np.mean(interval_errors <= 0.05))
    reused_attack_fraction = float(np.mean(np.diff(chosen) == 0))
    # Reusing one detected sound for a train of hits cannot be rewarded as rhythm matching.
    rhythm = phrase_support * (1 - reused_attack_fraction)
    score = 0.45 * precision + 0.30 * excess + 0.25 * rhythm
    return _axis(score, {"fixed_layer": "full", "bands": evidence,
                         "chance_corrected_support": excess, "interval_support_50ms": phrase_support,
                         "reused_attack_fraction": reused_attack_fraction},
                 ["Spectral attacks are not perceived instruments; soft vocals/sustained layers may be missed.",
                  "No semantic phrase recognition; interval support is a detector proxy."])


def _normalized_pairs(matrix):
    matrix = np.asarray(matrix, dtype=float)
    scale = np.std(matrix, axis=0)
    useful = scale > 1e-8
    if not useful.any():
        return np.zeros(len(matrix) * (len(matrix) - 1) // 2)
    z = (matrix[:, useful] - np.mean(matrix[:, useful], axis=0)) / scale[useful]
    pairs = pdist(z) / math.sqrt(useful.sum())
    return np.clip(pairs / 3, 0, 1)


def _patterns(notes, times, features, diagnostic):
    edges = features["edges"]
    windows = int(features["duration"] // PARAMETERS["section_seconds"])
    if windows < 4 or len(times) < 16:
        return _axis(None, {}, ["Not enough sections for development assessment."])
    attacks = np.asarray([np.histogram(features["attacks"][band], edges)[0][:windows]
                          for band in ("low", "mid", "high")]).T
    audio = np.column_stack((features["levels"][:windows], attacks))
    # Describe rhythms/positions/directions; entropy itself earns no points.
    descriptors = np.zeros((windows, 1 + 8 + 6 + 6))
    sorted_notes = sorted(notes, key=lambda n: (n["b"], n["c"]))
    clock = _BeatClock(float(diagnostic["_bpm"]), diagnostic["_bpm_events"])
    previous = None
    for note in sorted_notes:
        t = clock.seconds(note["b"])
        win = int(t // PARAMETERS["section_seconds"])
        if not 0 <= win < windows:
            continue
        descriptors[win, 0] += 1
        if 0 <= note["d"] < 8:
            descriptors[win, 1 + int(note["d"])] += 1
        descriptors[win, 9 + int(note["c"]) * 3 + int(note["y"])] += 1
        if previous is not None:
            gap = t - previous
            bucket = int(np.searchsorted([0.08, 0.2, 0.4, 0.7, 1.2], gap))
            descriptors[win, 15 + bucket] += 1
        previous = t
    # Retain counts as the first feature, normalize the remainder to separate pattern from density.
    descriptors[:, 1:] /= np.maximum(descriptors[:, :1], 1)
    audio_pairs = _normalized_pairs(audio)
    map_pairs = _normalized_pairs(descriptors)
    mismatch = float(np.mean(abs(audio_pairs - map_pairs)))
    if np.std(audio_pairs) < 1e-8:
        correlation = None
        support = 1.0 if np.std(map_pairs) < 1e-8 else 0.5
        chance_mean = None
    elif np.std(map_pairs) < 1e-8:
        correlation, support, chance_mean = 0.0, 0.0, None
    else:
        correlation = float(np.corrcoef(audio_pairs, map_pairs)[0, 1])
        rng = np.random.default_rng(1138)
        chance = [float(np.corrcoef(audio_pairs, _normalized_pairs(descriptors[rng.permutation(windows)]))[0, 1])
                  for _ in range(32)]
        chance_mean = float(np.mean(chance))
        support = _clip((correlation - chance_mean) / max(1e-9, 1 - chance_mean))
    change = np.linalg.norm(np.diff(audio, axis=0) / np.maximum(np.std(audio, axis=0), 1e-8), axis=1)
    map_change = np.linalg.norm(np.diff(descriptors, axis=0), axis=1)
    music_changes = change >= 1.0
    unchanged = map_change <= 0.05
    ignored = float(np.mean(unchanged[music_changes])) if music_changes.any() else 0.0
    repeats = [v.get("periodic_match_fraction") for v in diagnostic["flow"].values()]
    repeat = min((v for v in repeats if v is not None), default=0)
    rigid = repeat * ignored
    # A uniform motif on a uniform track is acceptable. Unexplained random change is a mismatch too.
    score = (0.4 * (1 - mismatch) + 0.6 * support) * (1 - 0.85 * rigid)
    return _axis(score, {"sections": windows, "section_seconds": PARAMETERS["section_seconds"],
                         "music_change_boundaries": int(music_changes.sum()),
                         "ignored_change_fraction": ignored, "minimum_hand_periodicity": repeat,
                         "rigidity_mismatch": rigid, "pairwise_structure_error": mismatch,
                         "pairwise_structure_correlation": correlation,
                         "permuted_baseline_mean": chance_mean, "chance_corrected_structure_support": support},
                 ["Coarse spectral/section proxy; thematic musical motives need listening.",
                  "Neither direction variety nor random note entropy is rewarded directly."])


def _density_response(notes, clock, features):
    """Reward phrase-level density that follows the audio, without rewarding target NPS."""
    windows = int(features["duration"] // PARAMETERS["section_seconds"])
    if windows < 4:
        return _axis(None, {}, ["Not enough audio sections for density-response assessment."])
    edges = features["edges"]
    note_times = np.asarray([clock.seconds(note["b"]) for note in notes], dtype=float)
    map_counts = np.histogram(note_times[(note_times >= 0) & (note_times < features["duration"])], edges)[0][:windows]
    attack_counts = np.histogram(features["attacks"]["full"], edges)[0][:windows]
    levels = np.asarray(features["levels"][:windows], dtype=float)
    audio_axes = {"level": levels, "attacks": attack_counts.astype(float)}
    rng = np.random.default_rng(5105)
    evidence = {"sections": windows, "section_seconds": PARAMETERS["section_seconds"],
                "notes_per_section": map_counts.tolist(), "audio_level_dbfs": np.round(levels, 3).tolist(),
                "attack_counts": attack_counts.tolist(), "signals": {}}
    supports = []
    for label, signal in audio_axes.items():
        if np.std(signal) < 1e-8:
            corr = None
            chance_mean = None
            support = 1.0 if np.std(map_counts) < 1e-8 else 0.5
        elif np.std(map_counts) < 1e-8:
            corr, chance_mean, support = 0.0, 0.0, 0.0
        else:
            corr = float(np.corrcoef(signal, map_counts)[0, 1])
            chance = [float(np.corrcoef(signal, rng.permutation(map_counts))[0, 1])
                      for _ in range(PARAMETERS["density_null_shuffles"])]
            chance_mean = float(np.mean(chance))
            support = _clip((corr - chance_mean) / max(1e-9, 1 - chance_mean))
        evidence["signals"][label] = {"correlation": corr, "permuted_baseline_mean": chance_mean,
                                       "chance_corrected_support": support}
        supports.append(support)
    mean_support = float(np.mean(supports))
    # Chance-level correspondence is neutral (.4); only positive, reproducible response earns credit.
    score = 0.4 + 0.6 * mean_support
    evidence["mean_chance_corrected_support"] = mean_support
    return _axis(score, evidence,
                 ["Four-second level/attack correspondence is a coarse proxy, not semantic phrase understanding.",
                  "No target-NPS reward; quiet musical passages are not assumed to be silence."])


def _readability(notes, data, clock, profile, duration, trusted):
    njs = profile.get("_noteJumpMovementSpeed", profile.get("noteJumpMovementSpeed"))
    offset = profile.get("_noteJumpStartBeatOffset", profile.get("noteJumpStartBeatOffset", 0))
    if not trusted or not isinstance(njs, (int, float)) or not math.isfinite(njs) or njs <= 0:
        return _axis(None, {"njs": njs}, ["Missing spawn metadata or unsupported geometry/timebase."])
    offset = float(offset or 0)
    if not math.isfinite(offset):
        return _axis(None, {}, ["Nonfinite spawn offset."])
    visible = 0
    simultaneous_conflicts = 0
    seen = Counter((n["b"], n["x"], n["y"]) for n in notes)
    simultaneous_conflicts = sum(count - 1 for count in seen.values() if count > 1)
    times = sorted(clock.seconds(n["b"]) for n in notes)
    times_array = np.asarray(times)
    jumps = []
    hidden = 0
    for note in notes:
        index = max(0, np.searchsorted(clock.beats, note["b"], side="right") - 1)
        beat_seconds = 60 / clock.rates[index]
        half_beats = 4.0
        while njs * beat_seconds * half_beats > 18:
            half_beats /= 2
        half_seconds = max(1, half_beats + offset) * beat_seconds
        jumps.append(half_seconds)
        # Conservative face-note candidate count, not a geometric occlusion simulation.
        if note["x"] in (1, 2) and note["y"] == 1:
            t = clock.seconds(note["b"])
            if np.searchsorted(times_array, t + half_seconds) - np.searchsorted(times_array, t, side="right") > 0:
                hidden += 1
        visible += 1
    counts, _ = np.histogram(times_array, np.arange(0, duration + 2, 2))
    # Actual local jump in demand, not deviation from a constant NPS target.
    spikes = sum(b > max(8, 3 * a) for a, b in zip(counts, counts[1:]))
    penalties = 0.7 * hidden / max(1, visible) + simultaneous_conflicts / max(1, visible)
    penalties += 0.2 * spikes / max(1, len(counts) - 1)
    return _axis(1 - penalties, {"njs": njs, "offset_beats": offset,
                                 "half_jump_seconds_min": min(jumps) if jumps else None,
                                 "half_jump_seconds_max": max(jumps) if jumps else None,
                                 "vision_block_candidates": hidden, "duplicate_cell_candidates": simultaneous_conflicts,
                                 "density_spikes": int(spikes)},
                 ["Spawn time is reported without universal good NJS bounds.",
                  "No measured human reaction time, collision trajectories or accumulated fatigue."])


def _presentation(data, clock, features, diagnostic):
    lighting = diagnostic["lighting"]
    if any(lighting["event_box_groups"].values()) or lighting["custom_events"]:
        return _axis(None, {"lighting": lighting, "palette": diagnostic.get("palette")},
                     ["Custom/event-box lighting requires renderer support; no automatic penalty."])
    events = sorted((e for e in data.get("basicBeatmapEvents", []) if e.get("et") in range(5)), key=lambda e: e["b"])
    if not events:
        return _axis(PARAMETERS["presentation_floor"], {"standard_events": 0, "raw_authored_show_proxy": 0,
                                                          "palette": diagnostic.get("palette"),
                                                          "palette_contrast_diagnostic": _palette_metrics(diagnostic.get("palette"))},
                     ["No authored standard show; this does not prove the actual environment is black."])
    # Estimate only standard event activity. Fade/flash durations and environment rendering are unknown.
    duration = features["duration"]
    grid = np.arange(0, duration, 0.1)
    state = np.zeros(5)
    active = []
    i = 0
    for t in grid:
        while i < len(events) and clock.seconds(events[i]["b"]) <= t:
            e = events[i]
            state[int(e["et"])] = int(e.get("i", 0)) % 4 != 0
            i += 1
        active.append(bool(state.any()))
    coverage = float(np.mean(active)) if active else 0
    event_times = np.unique([clock.seconds(e["b"]) for e in events if 0 <= clock.seconds(e["b"]) < duration])
    distances = _distances(event_times, features["attacks"]["full"])
    align = float(np.mean(distances <= 0.1)) if distances is not None and len(distances) else 0
    score = coverage * (0.4 + 0.6 * align)
    if len(events) <= 1:
        score = min(score, 0.2)
    palette = diagnostic.get("palette")
    palette_metrics = _palette_metrics(palette)
    return _axis(PARAMETERS["presentation_floor"] + (1 - PARAMETERS["presentation_floor"]) * score,
                 {"standard_events": len(events), "raw_authored_show_proxy": score,
                         "estimated_standard_on_fraction": coverage,
                         "event_attack_alignment_100ms": align, "palette": palette,
                         "palette_contrast_diagnostic": palette_metrics},
                 ["Authored standard-light activity proxy; visibility, brightness, fades and theme fit remain unverified.",
                  "Palette contrast is reported but not scored: thematic fit needs blind human/rendered review.",
                  "A floor prevents absent authored lighting from collapsing the entire geometric score."])


def _palette_metrics(palette):
    if not isinstance(palette, dict):
        return {"available": False}
    left, right = palette.get("scene_left"), palette.get("scene_right")
    def rgb(value):
        if not isinstance(value, dict):
            return None
        channels = [value.get(key) for key in ("r", "g", "b")]
        if not all(isinstance(channel, (int, float)) and math.isfinite(channel) for channel in channels):
            return None
        return np.asarray(channels, dtype=float)
    a, b = rgb(left), rgb(right)
    return {"available": a is not None and b is not None,
            "scene_pair_euclidean_rgb_distance": float(np.linalg.norm(a - b)) if a is not None and b is not None else None,
            "scene_pair_distinct": bool(np.linalg.norm(a - b) >= 0.2) if a is not None and b is not None else None}


def assess_package(pkg: GeneratedPackage, diagnostics: dict, target_nps: float | None = None) -> dict:
    try:
        path = Path(pkg.audio_path).resolve()
        stat = path.stat()
        features = _audio_features(str(path), stat.st_size, stat.st_mtime_ns) if diagnostics.get("audio_assessed") else None
    except (OSError, ValueError):
        features = None
    reviews = {}
    for name, data in pkg.difficulties.items():
        diagnostic = diagnostics.get("difficulty_reviews", {}).get(name, {})
        notes = data.get("colorNotes", [])
        hits = _playable_hits(data)
        profiles = [entry for group in pkg.info.get("difficultyBeatmapSets", [])
                    for entry in group.get("_difficultyBeatmaps", group.get("difficultyBeatmaps", []))
                    if entry.get("_beatmapFilename", entry.get("beatmapFilename")) == name]
        profile = profiles[0] if profiles else {}
        trusted = (diagnostic.get("assessed", False) and all(v.get("timebase_assessed", True) for v in diagnostic.get("flow", {}).values())
                   and not any(_geometry_is_unknown(n) or n.get("a", 0) for n in notes)
                   and all(n.get("x") in range(4) and n.get("y") in range(3) and n.get("c") in (0, 1) for n in notes)
                   and diagnostic.get("elements", {}).get("valid", True))
        try:
            clock = _BeatClock(float(pkg.info.get("beatsPerMinute", 0)), data.get("bpmEvents", []))
        except (TypeError, ValueError, KeyError):
            reviews[name] = {"experimental_score": None, "reason": "unsupported BPM timebase"}
            continue
        all_times = np.asarray([clock.seconds(n["b"]) for n in hits])
        times = np.unique(all_times[(all_times >= 0) & (all_times < features["duration"])]) if features is not None else []
        d = {**diagnostic, "_bpm": pkg.info.get("beatsPerMinute"), "_bpm_events": data.get("bpmEvents", [])}
        # Use the V3 playable projection so arc heads and chain slices are not
        # silently excluded from movement evidence.
        axes = {"movement": _movement(hits, data, clock, trusted)}
        if features is not None and trusted and features["duration"] >= features["full_duration"] - 0.01:
            axes.update({"musicality": _musicality(times, features, diagnostic),
                         "pattern_development": _patterns(hits, times, features, d),
                         "density_response": _density_response(hits, clock, features),
                         "readability": _readability(hits, data, clock, profile, features["full_duration"], trusted)})
            presentation = _presentation(data, clock, features, diagnostic)
        else:
            axes.update({k: _axis(None, {}, ["Audio/timebase/geometry unavailable; no weight renormalization."])
                         for k in ("musicality", "pattern_development", "density_response", "readability")})
            presentation = _axis(None, {}, ["Audio or supported geometry unavailable."])
        axes["presentation"] = presentation
        structure = diagnostic.get("structure", {})
        score = geometric_score(axes)
        experience = score
        nps = structure.get("actual_nps")
        compliant = None if target_nps is None or nps is None else abs(nps - target_nps) <= target_nps * PARAMETERS["density_tolerance"]
        reviews[name] = {"axes": axes, "presentation": presentation,
                         "base_geometric_score": round(score, 2) if score is not None else None,
                         "experimental_score": round(score, 2) if score is not None else None,
                         "experimental_score_100": round(score, 2) if score is not None else None,
                         "experience_proxy": round(experience, 2) if experience is not None else None,
                         "caps": [], "requirements": {"target_nps": target_nps, "actual_nps": nps,
                             "within_density_tolerance": compliant},
                         "unrated_axes": [k for k, v in axes.items() if v["score"] is None]}
    values = [r.get("experimental_score_100") for r in reviews.values()]
    overall = min(values) if values and all(v is not None for v in values) else None
    return {"version": VERSION, "score_kind": "uncalibrated_machine_proxy_0_100", "calibration_status": "awaiting_blind_player_ratings",
            "quality_score": None, "experimental_score": overall, "difficulty_policy": "minimum; each difficulty is also reported",
            "difficulty_assessments": reviews, "weights": WEIGHTS, "parameters": PARAMETERS,
            "code_sha256": CODE_SHA256,
            "scale": {"minimum": 0, "maximum": 100, "meaning": "linear display of current offline proxy evidence; not an absolute human quality rating"},
            "interpretation": "Axis and aggregate values are displayed on 0-100, but remain uncalibrated offline proxies. They must not be read as absolute map quality until blind player ratings calibrate them. Includes phrase-level density response and authored standard-light presentation. Target NPS remains a separate requirement; no model, format, endpoint, mod or token bonuses. Unknown essential evidence leaves the aggregate null."}
