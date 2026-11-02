"""Swing simulation: detect unplayable note patterns."""
from __future__ import annotations
from collections import defaultdict
from .elements import discrete_hits
from ..types import GeneratedPackage, Subscore

WALL_GRACE_BEATS = 0.25
RESET_GAP_BEATS = 0.75


def _count_resets(notes: list[dict]) -> int:
    ordered = sorted(notes, key=lambda n: n["b"])
    return sum(
        1
        for prev, cur in zip(ordered, ordered[1:])
        if prev.get("c") == cur.get("c")
        and cur["b"] - prev["b"] < RESET_GAP_BEATS
        and (cur["y"] >= prev["y"] or cur["x"] == prev["x"])
    )


def validate_playability(pkg: GeneratedPackage) -> Subscore:
    violations: dict[str, int] = defaultdict(int)
    total = dirty = resets = 0
    for fname, dat in pkg.difficulties.items():
        notes = discrete_hits(dat)
        total += len(notes)
        resets += _count_resets(notes)

        bad_walls = [
            o for o in dat.get("obstacles", [])
            if o.get("d") == 0
            and o.get("x", 99) <= 1 and o.get("x", -1) + o.get("w", 0) >= 3
            and o.get("y", 0) <= 1 and o.get("y", 9) + o.get("h", 0) >= 1
        ]

        seen: set[tuple] = set()
        dirty_cells: set[tuple] = set()
        dup_only_cells: set[tuple] = set()
        for n in notes:
            animated = bool((n.get("customData") or {}).get("animation"))
            key = (n["b"], n["x"], n["y"])
            if not animated and key in seen:
                violations["duplicate_cell"] += 1
                if key not in dirty_cells:
                    dup_only_cells.add(key)
                continue
            seen.add(key)
            bomb_hit = any(
                (bomb.get("b"), bomb.get("x"), bomb.get("y")) == key
                for bomb in dat.get("bombNotes", []))
            wall_hit = any(abs(n["b"] - w["b"]) <= WALL_GRACE_BEATS
                           for w in bad_walls)
            if bomb_hit:
                violations[f"{fname}:bomb_overlap"] += 1
            if wall_hit:
                violations[f"{fname}:wall_on_note"] += 1
            if bomb_hit or wall_hit:
                dirty_cells.add(key)
        dirty += len(dirty_cells) + len(dup_only_cells)

    reset_rate = resets / total if total else 0.0
    score = 0.0 if total == 0 else max(0.0, round(1.0 - dirty / total, 4))
    score = max(0.0, round(score * (1.0 - min(reset_rate, 0.5)), 4))
    return Subscore("playability", score,
                    {"violations": dict(violations), "total_notes": total,
                     "reset_rate": round(reset_rate, 4)})
