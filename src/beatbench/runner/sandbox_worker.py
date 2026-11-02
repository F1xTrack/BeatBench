"""Container/namespace worker. No benchmark code or credentials are mounted here."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import traceback


class BoundedText(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.length = 0
        self.truncated = False

    def write(self, text):
        remaining = max(0, self.limit - self.length)
        if remaining:
            self.parts.append(text[:remaining])
        self.length += min(len(text), remaining)
        self.truncated |= len(text) > remaining
        return len(text)

    def value(self):
        return "".join(self.parts)


def main():
    request = json.loads(sys.stdin.read(100001))
    resource.setrlimit(resource.RLIMIT_CPU, (request["cpu_seconds"], request["cpu_seconds"]))
    memory = request.get("address_space_bytes", 2 * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    processes = request.get("process_limit", 32)
    resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
    signal.alarm(request["wall_seconds"])
    output = BoundedText(24000)
    error = BoundedText(4000)
    success = True
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
        try:
            exec(compile(request["code"], "<model-python>", "exec"), {"__name__": "__main__"})
        except BaseException:
            success = False
            traceback.print_exc(limit=6)
    images = []
    from PIL import Image
    for path in sorted(Path("/work").glob("*.png"))[:2]:
        if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
            continue
        try:
            with Image.open(path) as im:
                im.thumbnail((1400, 800))
                buffer = io.BytesIO()
                im.convert("RGB").save(buffer, format="PNG")
            images.append({"name": path.name, "data": base64.b64encode(buffer.getvalue()).decode("ascii")})
        except Exception:
            pass
    payload = {"ok": success, "stdout": output.value(), "stderr": error.value(),
               "truncated": output.truncated or error.truncated, "images": images}
    sys.__stdout__.write(json.dumps(payload))


if __name__ == "__main__":
    main()
