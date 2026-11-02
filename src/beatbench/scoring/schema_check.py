"""Structural gate: every file must pass its V3 JSON Schema."""
from __future__ import annotations
import json
from importlib import resources
import jsonschema
from ..types import GeneratedPackage


def _load(name: str) -> dict:
    return json.loads(resources.files("beatbench").joinpath(f"schema/{name}").read_text("utf-8"))


INFO_SCHEMA = _load("info_v3.schema.json")
DIFF_SCHEMA = _load("difficulty_v3.schema.json")


def _finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == value and value not in (float("inf"), float("-inf")))


def _validate_v3_elements(dat: dict) -> list[str]:
    """Validate semantic V3 arc/chain geometry after JSON-schema validation.

    The schema can establish field types and ranges, but it cannot express the
    relation between a head and a tail.  Keep this gate conservative: malformed
    arcs/chains must not be scored as ordinary note maps.
    """
    errors: list[str] = []
    for index, arc in enumerate(dat.get("sliders", []) or []):
        if not isinstance(arc, dict):
            continue
        if not (_finite_number(arc.get("b")) and _finite_number(arc.get("tb"))):
            continue
        if arc["tb"] <= arc["b"]:
            errors.append(f"sliders/{index}/tb: arc tail beat must be after head beat")

    for index, chain in enumerate(dat.get("burstSliders", []) or []):
        if not isinstance(chain, dict):
            continue
        if not (_finite_number(chain.get("b")) and _finite_number(chain.get("tb"))):
            continue
        if chain["tb"] < chain["b"]:
            errors.append(f"burstSliders/{index}/tb: tail beat cannot precede head beat")
        if isinstance(chain.get("sc"), int) and chain["sc"] < 1:
            errors.append(f"burstSliders/{index}/sc: slice count must be positive")
        if _finite_number(chain.get("s")) and chain["s"] <= 0:
            errors.append(f"burstSliders/{index}/s: squish factor must be positive")
    return errors


def check_schema(pkg: GeneratedPackage) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for label, obj, schema in [("Info.dat", pkg.info, INFO_SCHEMA)] + [
        (name, dat, DIFF_SCHEMA) for name, dat in pkg.difficulties.items()
    ]:
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            loc = "/".join(str(p) for p in e.absolute_path)
            errors.append(f"{label}: {loc}: {e.message}")
        if label != "Info.dat" and isinstance(obj, dict):
            errors.extend(f"{label}: {error}" for error in _validate_v3_elements(obj))
    return len(errors) == 0, errors
