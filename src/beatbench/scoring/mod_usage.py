"""Presence and structural correctness of Chroma / Noodle Extensions usage."""
from __future__ import annotations
from ..types import GeneratedPackage, Subscore


def validate_mod_usage(pkg: GeneratedPackage) -> Subscore:
    chroma_present = chroma_ok = noodle_present = False

    for dat in pkg.difficulties.values():
        events = dat.get("basicBeatmapEvents", [])
        custom_events = [e for e in events if e.get("customData")]
        if custom_events or dat.get("lightColorEventBoxGroups") \
                or dat.get("useNormalEventsAsCompatibleEvents"):
            chroma_present = True
        groups = dat.get("lightColorEventBoxGroups") or []
        chroma_ok = chroma_ok or bool(custom_events) or all(
            isinstance(g, dict) and "ebt" in g and "l" in g for g in groups)

        for coll in ("colorNotes", "bombNotes", "obstacles", "sliders", "burstSliders"):
            for obj in dat.get(coll, []):
                anim = (obj.get("customData") or {}).get("animation")
                if isinstance(anim, dict):
                    noodle_present = True

    chroma_score = 1.0 if chroma_present and chroma_ok else (0.5 if chroma_present else 0.0)
    noodle_score = 1.0 if noodle_present else 0.0
    return Subscore("mod_usage", round((chroma_score + noodle_score) / 2, 4),
                    {"chroma": chroma_score, "noodle": noodle_score})
