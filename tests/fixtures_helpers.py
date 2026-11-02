import json
from pathlib import Path

from beatbench.types import GeneratedPackage

FIXTURES = Path(__file__).parent / "fixtures"


def _load(which: str) -> dict:
    return json.loads((FIXTURES / "maps" / which / "info.json").read_text(encoding="utf-8"))


def __getattr__(name: str):
    if name == "INFO":
        return _load("valid")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def valid_pkg(broken: bool = False) -> GeneratedPackage:
    which = "broken" if broken else "valid"
    info = _load(which)
    dat = json.loads((FIXTURES / "maps" / which / "ExpertPlus.dat").read_text(encoding="utf-8"))
    return GeneratedPackage(
        info=info,
        difficulties={"ExpertPlus.dat": dat},
        audio_path=Path("song.ogg"),
        cover_path=None,
    )
