"""CLI wrapper: convert a generated V3 map package to V2 for older Quest SongCore builds."""
from __future__ import annotations

import sys
from pathlib import Path

from beatbench.export.v2 import convert_folder

if __name__ == "__main__":
    convert_folder(Path(sys.argv[1]), Path(sys.argv[2]))
