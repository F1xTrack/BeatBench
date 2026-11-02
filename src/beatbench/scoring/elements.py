"""Shared V3 playable-element helpers.

V3 arcs live in ``sliders`` and chains (multiblocks) in ``burstSliders``.
They are not duplicated in ``colorNotes`` by the format, so validators that
only inspect color notes silently undercount maps.  This module provides a
small, conservative projection for timing/density/playability diagnostics;
the source beatmap is never mutated.
"""
from __future__ import annotations


def map_end_beat(data: dict) -> float:
    """Return the furthest beat occupied by any playable V3 element."""
    ends: list[float] = []
    for note in data.get("colorNotes", []) or []:
        try:
            ends.append(float(note["b"]))
        except (KeyError, TypeError, ValueError):
            continue
    for key in ("sliders", "burstSliders"):
        for element in data.get(key, []) or []:
            if not isinstance(element, dict):
                continue
            try:
                value = element.get("tb")
                if value is None:
                    value = element.get("b")
                ends.append(float(value))
            except (KeyError, TypeError, ValueError):
                continue
    return max(ends, default=0.0)


def _cell_key(item: dict) -> tuple[float, int | None, int | None, int | None]:
    return (float(item.get("b", 0)), item.get("c"), item.get("x"), item.get("y"))


def discrete_hits(data: dict) -> list[dict]:
    """Project V3 objects to conservative discrete timing observations.

    Arcs contribute their head, while chains contribute their authored slices.
    Existing color notes at the same cell/time are de-duplicated.  Arc tails
    extend coverage but are not treated as an additional cut hit.
    """
    hits = [dict(note) for note in data.get("colorNotes", []) or []]
    existing = {_cell_key(note) for note in hits}

    for index, arc in enumerate(data.get("sliders", []) or []):
        try:
            head = {"b": float(arc["b"]), "x": int(arc["x"]), "y": int(arc["y"]),
                    "c": int(arc["c"]), "d": int(arc.get("d", 8)), "_arc_head": True,
                    "_arc_index": index}
        except (KeyError, TypeError, ValueError):
            continue
        if _cell_key(head) not in existing:
            existing.add(_cell_key(head))
            hits.append(head)

    for chain_index, chain in enumerate(data.get("burstSliders", []) or []):
        try:
            count = max(1, int(chain["sc"]))
            start, end = float(chain["b"]), float(chain["tb"])
            sx, sy = float(chain["x"]), float(chain["y"])
            tx, ty = float(chain["tx"]), float(chain["ty"])
            color, direction = int(chain["c"]), int(chain.get("d", 8))
        except (KeyError, TypeError, ValueError):
            continue
        for slice_index in range(count):
            ratio = slice_index / max(1, count - 1)
            hit = {"b": start + (end - start) * ratio,
                   "x": round(sx + (tx - sx) * ratio),
                   "y": round(sy + (ty - sy) * ratio),
                   "c": color, "d": direction if slice_index == 0 else 8,
                   "_chain_slice": True, "_chain_index": chain_index,
                   "_chain_slice_index": slice_index}
            if _cell_key(hit) in existing:
                continue
            existing.add(_cell_key(hit))
            hits.append(hit)
    return hits

