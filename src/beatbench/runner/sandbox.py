"""Python analysis isolated by Linux namespaces inside WSL (or native Linux)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

DEFAULT_LIMITS = {"max_calls": 12, "total_wall_seconds": 180,
                  "call_wall_seconds": 30, "call_cpu_seconds": 20,
                  "address_space_bytes": 2 * 1024**3, "process_limit": 32,
                  "workspace_bytes": 64 * 1024**2, "output_bytes": 2 * 1024**2}
VENV = "/opt/beatbench-sandbox"


class PythonSandbox:
    def __init__(self, audio: Path, cover: Path | None, distro="Ubuntu-24.04", limits=None):
        self.distro = distro
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.calls = 0
        self.elapsed_s = 0.0
        self._temp = tempfile.TemporaryDirectory(prefix="beatbench_signal_")
        self.input_dir = Path(self._temp.name) / "input"
        self.input_dir.mkdir()
        self.worker_path = Path(self._temp.name) / "worker.py"
        shutil.copyfile(Path(__file__).with_name("sandbox_worker.py"), self.worker_path)
        shutil.copyfile(audio, self.input_dir / "audio.wav")
        self.cover_name = None
        if cover is not None:
            self.cover_name = "cover" + cover.suffix.lower()
            shutil.copyfile(cover, self.input_dir / self.cover_name)

    def close(self):
        self._temp.cleanup()

    def _prefix(self):
        return ["wsl", "-d", self.distro, "--"] if os.name == "nt" else []

    def _linux_path(self, path: Path) -> str:
        if os.name != "nt":
            return str(path.resolve())
        result = subprocess.run(self._prefix() + ["wslpath", "-a", path.resolve().as_posix()],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        if result.returncode:
            raise RuntimeError("WSL path conversion failed: " + result.stderr[-1000:])
        return result.stdout.strip()

    def preflight(self) -> dict:
        cmd = self._prefix() + ["sh", "-c", "command -v bwrap && "
                               f"{VENV}/bin/python -c 'import importlib.metadata as m, json; "
                               "print(json.dumps({p:m.version(p) for p in "
                               "[\"numpy\",\"scipy\",\"librosa\",\"soundfile\",\"matplotlib\",\"pillow\"]}))'"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=20)
        if result.returncode:
            raise RuntimeError("Python sandbox unavailable; run tools/setup_python_sandbox.ps1. " + result.stderr[-1000:])
        versions = json.loads(result.stdout.splitlines()[-1])
        probe = self.execute("import os, socket; print(os.getuid()); print(os.listdir('/input')); "
                             "print(os.path.exists('/mnt')); print(socket.if_nameindex())", charge=False)
        if not probe.get("ok"):
            raise RuntimeError("Python sandbox isolation probe failed: " + str(probe))
        return {"backend": "bubblewrap", "distro": self.distro if os.name == "nt" else None,
                "libraries": versions, "limits": self.limits, "probe": probe,
                "worker_sha256": __import__("hashlib").sha256(self.worker_path.read_bytes()).hexdigest()}

    def command(self, wall_seconds=None) -> list[str]:
        worker = self._linux_path(self.worker_path)
        cmd = self._prefix() + ["timeout", "--signal=KILL", str(wall_seconds or self.limits["call_wall_seconds"]),
                                "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                                "--cap-drop", "ALL", "--clearenv"]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            cmd.extend(["--ro-bind", path, path])
        cmd.extend(["--ro-bind", VENV, VENV, "--ro-bind", worker, "/worker.py",
                    "--ro-bind", self._linux_path(self.input_dir), "/input",
                    "--proc", "/proc", "--dev", "/dev",
                    "--size", str(self.limits["workspace_bytes"]), "--tmpfs", "/tmp",
                    "--size", str(self.limits["workspace_bytes"]), "--tmpfs", "/work",
                    "--chdir", "/work"])
        for key, value in {"PATH": f"{VENV}/bin:/usr/bin:/bin", "HOME": "/tmp",
                           "MPLCONFIGDIR": "/tmp/matplotlib", "NUMBA_CACHE_DIR": "/tmp/numba",
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}.items():
            cmd.extend(["--setenv", key, value])
        cmd.extend(["--", f"{VENV}/bin/python", "-I", "/worker.py"])
        return cmd

    def execute(self, code: str, charge=True) -> dict:
        if not isinstance(code, str) or len(code.encode("utf-8")) > 60000:
            return {"ok": False, "error": "code must be a string of at most 60000 UTF-8 bytes"}
        if charge and (self.calls >= self.limits["max_calls"] or
                       self.elapsed_s >= self.limits["total_wall_seconds"]):
            return {"ok": False, "error": "run-wide Python budget exhausted"}
        wall = self.limits["call_wall_seconds"]
        if charge:
            wall = min(wall, max(1, int(self.limits["total_wall_seconds"] - self.elapsed_s)))
            self.calls += 1
        request = json.dumps({"code": code, "wall_seconds": wall,
                              "cpu_seconds": self.limits["call_cpu_seconds"],
                              "address_space_bytes": self.limits["address_space_bytes"],
                              "process_limit": self.limits["process_limit"]}).encode()
        start = time.monotonic()
        proc = subprocess.Popen(self.command(wall), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = threading.Event()

        def read(stream, key):
            while chunk := stream.read(8192):
                if len(buffers[key]) + len(chunk) > self.limits["output_bytes"]:
                    overflow.set()
                    proc.kill()
                    break
                buffers[key].extend(chunk)

        readers = [threading.Thread(target=read, args=(stream, key), daemon=True)
                   for stream, key in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
        for thread in readers:
            thread.start()
        try:
            proc.stdin.write(request)
            proc.stdin.close()
            proc.wait(timeout=wall + 5)
        except (subprocess.TimeoutExpired, BrokenPipeError):
            proc.kill()
            proc.wait()
        finally:
            for thread in readers:
                thread.join(timeout=3)
            for stream in (proc.stdout, proc.stderr):
                stream.close()
        elapsed = time.monotonic() - start
        if charge:
            self.elapsed_s += elapsed
        if overflow.is_set():
            result = {"ok": False, "error": "sandbox output exceeded byte limit"}
        elif proc.returncode:
            result = {"ok": False, "error": f"sandbox terminated (exit {proc.returncode})",
                      "stderr": buffers["stderr"].decode("utf-8", errors="replace")[-4000:]}
        else:
            try:
                result = json.loads(buffers["stdout"].decode("utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeError):
                result = {"ok": False, "error": "sandbox returned malformed output"}
        result["elapsed_s"] = elapsed
        return result
