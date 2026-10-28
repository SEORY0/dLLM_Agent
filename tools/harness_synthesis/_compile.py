"""Compile a harness source through Task 5a's harness-compile-only image."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

IMAGE = "dllm-openexr-harness-compile-only"
_DOCKERFILE_DIR = Path(__file__).resolve().parents[1] / "openexr_runtime/harness-compile-only"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_CORPUS_HEX = _REPO_ROOT / "fixtures/openexr/c0-validity-preserving-zero.hex"
_ERROR_LINE = re.compile(r"harness\.cc:\d+:\d+: (fatal )?error:")
_COV_LINE = re.compile(r"cov: (\d+)")
_image_ready = False


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Whether a harness source compiled, and its error text if not."""

    compiled: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class FuzzResult:
    """Compile-and-fuzz outcome for RQ3's fixed-budget coverage comparison.

    ``edges`` is ``None`` when the harness did not compile -- callers doing an
    intent-to-treat analysis (plan Task 10 protocol item 3: primary outcome defined
    over ALL attempted trials, not just compiling ones) should treat that as 0 edges,
    not drop the trial.
    """

    compiled: bool
    compile_error: str | None
    edges: int | None


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


def _build_coverage_binary(work: Path, *, timeout: float) -> str | None:
    """Build ``work/coverage_target`` from ``work/harness.cc``; return an error, or None."""
    _ensure_image()
    uid = f"{os.getuid()}:{os.getgid()}"
    build = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        ["docker", "run", "--rm", "-u", uid, "-v", f"{work}:/work", IMAGE, "coverage"],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if build.returncode == 0 and (work / "coverage_target").exists():
        return None
    lines = [
        line
        for line in (*build.stdout.splitlines(), *build.stderr.splitlines())
        if _ERROR_LINE.search(line)
    ]
    return "\n".join(line.replace("/work/", "") for line in lines[:3]) or build.stderr[-500:]


def fuzz_once(binary_dir: Path, *, fuzz_seconds: float, seed: int) -> int | None:
    """Fuzz an already-built ``binary_dir/coverage_target`` once; return edges or ``None``.

    Lets a caller reuse one compiled binary across several fuzz replicates (a fixed
    seed bundle, plan Task 10's "replicate to estimate variance" item) without
    recompiling for each one.
    """
    uid = f"{os.getuid()}:{os.getgid()}"
    fuzz = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        [  # noqa: S607
            "docker", "run", "--rm", "--entrypoint", "/work/coverage_target",
            "-u", uid, "-v", f"{binary_dir}:/work", "-w", "/work", IMAGE,
            f"-max_total_time={int(fuzz_seconds)}", f"-seed={seed}",
            "-print_final_stats=1", "-close_fd_mask=3", "/work/corpus",
        ],
        capture_output=True,
        text=True,
        timeout=fuzz_seconds + 60,
        check=False,
    )
    covered = _COV_LINE.findall(fuzz.stdout + fuzz.stderr)
    return int(covered[-1]) if covered else None


@contextlib.contextmanager
def compile_for_repeated_fuzzing(
    source: str, *, timeout: float = 180.0
) -> Iterator[tuple[Path | None, str | None]]:
    """Compile once; yield ``(binary_dir, None)`` on success or ``(None, error)``.

    ``binary_dir`` is valid for repeated ``fuzz_once`` calls until the context exits,
    when it is removed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        work.chmod(0o777)
        (work / "harness.cc").write_text(source)
        corpus = work / "corpus"
        corpus.mkdir()
        (corpus / "c0").write_bytes(bytes.fromhex(_SEED_CORPUS_HEX.read_text().strip()))
        error = _build_coverage_binary(work, timeout=timeout)
        yield (None, error) if error is not None else (work, None)


def compile_and_fuzz(
    source: str, *, fuzz_seconds: float, seed: int, timeout: float = 180.0
) -> FuzzResult:
    """Compile, then fuzz once for exactly ``fuzz_seconds``; return the edge count.

    Fuzzes from the pinned C0 seed corpus. A fixed-duration run (not a fixed-executions
    run) matches plan Task 10 item 2's "match arms on fuzzing time": every arm gets the
    same wall-clock fuzzing budget regardless of how fast or slow its own binary
    executes. For repeated fuzz replicates of the same binary, use
    ``compile_for_repeated_fuzzing`` + ``fuzz_once`` instead of calling this in a loop
    (which would recompile every time).
    """
    with compile_for_repeated_fuzzing(source, timeout=timeout) as (binary_dir, error):
        if binary_dir is None:
            return FuzzResult(compiled=False, compile_error=error, edges=None)
        edges = fuzz_once(binary_dir, fuzz_seconds=fuzz_seconds, seed=seed)
        return FuzzResult(compiled=True, compile_error=None, edges=edges)


_INCLUDE_FLAGS = (
    "-I/opt/openexr-coverage/include/OpenEXR",
    "-I/opt/openexr-coverage/include/Imath",
    "-I/opt/openexr-coverage/include",
)


def syntax_diagnostics_many(sources: list[str], *, timeout: float = 600.0) -> list[list[str]]:
    """Every ``harness.cc`` ``error:`` line clang reports for each source, uncapped.

    ``compile_harness``/``_build_coverage_binary`` keep only the first three error lines,
    which is enough to pick a location but not to count diagnostics. This runs
    ``clang++ -fsyntax-only -ferror-limit=0`` against the same pinned OpenEXR headers,
    with one container for the whole batch (about 0.3 s per source instead of a 3 s
    container start each). It does not link, so a link-only failure would read as
    clean. The retrospective audit checks that its verdict matches the stored build
    result for every source.
    """
    _ensure_image()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        work.chmod(0o777)
        for index, source in enumerate(sources):
            case = work / str(index)
            case.mkdir()
            case.chmod(0o777)
            (case / "harness.cc").write_text(source)
        flags = " ".join(("-std=c++17", "-fsyntax-only", "-ferror-limit=0", *_INCLUDE_FLAGS))
        script = (
            f"for i in $(seq 0 {len(sources) - 1}); do "
            f"clang++ {flags} /work/$i/harness.cc > /work/$i/diag.txt 2>&1; done; exit 0"
        )
        uid = f"{os.getuid()}:{os.getgid()}"
        subprocess.run(  # noqa: S603 - fixed argv, shell=False on the host side.
            [  # noqa: S607
                "docker", "run", "--rm", "-u", uid, "-v", f"{work}:/work",
                "--entrypoint", "sh", IMAGE, "-c", script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return [
            [
                line.replace("/work/", "").removeprefix(f"{index}/")
                for line in (work / str(index) / "diag.txt").read_text().splitlines()
                if _ERROR_LINE.search(line)
            ]
            for index in range(len(sources))
        ]
