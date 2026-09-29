"""Compile a harness source through Task 5a's harness-compile-only image."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE = "dllm-openexr-harness-compile-only"
_DOCKERFILE_DIR = Path(__file__).resolve().parents[1] / "openexr_runtime/harness-compile-only"
_ERROR_LINE = re.compile(r"harness\.cc:\d+:\d+: (fatal )?error:")
_image_ready = False


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Whether a harness source compiled, and its error text if not."""

    compiled: bool
    error: str | None


def _ensure_image() -> None:
    """(Re)build the compile-only image once per process if it isn't already present.

    A prior cleanup (or a fresh checkout) can leave ``IMAGE`` untagged; ``docker run``
    then fails with a *pull* error that looks nothing like a compile error, silently
    scoring every harness as non-compiling. Checking (and building) once here, instead
    of leaving that failure mode for the caller to hit, is worth the first-call cost.
    """
    global _image_ready  # noqa: PLW0603 - one-shot process-local build cache.
    if _image_ready:
        return
    inspect = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        ["docker", "image", "inspect", IMAGE],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if inspect.returncode != 0:
        subprocess.run(  # noqa: S603 - fixed argv, shell=False.
            [  # noqa: S607
                "docker", "build", "-q", "-f", str(_DOCKERFILE_DIR / "Dockerfile"),
                "-t", IMAGE, str(_DOCKERFILE_DIR),
            ],
            check=True,
            capture_output=True,
        )
    _image_ready = True


def compile_harness(source: str, *, mode: str = "coverage", timeout: float = 90.0) -> CompileResult:
    """Compile ``source`` inside the pinned compile-only image; nothing persists after."""
    _ensure_image()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        work.chmod(0o777)
        (work / "harness.cc").write_text(source)
        uid = f"{os.getuid()}:{os.getgid()}"
        result = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
            ["docker", "run", "--rm", "-u", uid, "-v", f"{work}:/work", IMAGE, mode],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        binary_name = "coverage_target" if mode == "coverage" else "fuzz_target"
        if result.returncode == 0 and (work / binary_name).exists():
            return CompileResult(compiled=True, error=None)
        lines = [
            line
            for line in (*result.stdout.splitlines(), *result.stderr.splitlines())
            if _ERROR_LINE.search(line)
        ]
        error = "\n".join(line.replace("/work/", "") for line in lines[:3]) or result.stderr[-500:]
        return CompileResult(compiled=False, error=error)
