"""Typed subprocess boundary shared by WebUI jobs."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

LOG_BYTE_LIMIT = 16_384


class ManagedProcess(Protocol):
    """Small subprocess protocol used by the one-worker manager."""

    def poll(self) -> int | None:
        """Return the child exit code when complete."""
        ...

    def communicate(self) -> tuple[bytes, bytes]:
        """Return bounded stdout and stderr after process completion."""
        ...


@dataclass(frozen=True, slots=True)
class ProcessLaunch:
    """Complete subprocess launch request with fixed safety flags."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    stdout: int
    stderr: int
    shell: Literal[False]
    close_fds: Literal[True]


class ProcessFactory(Protocol):
    """Factory for fixed-argv subprocesses."""

    def __call__(self, request: ProcessLaunch) -> ManagedProcess:
        """Launch a process."""
        ...


def default_process_factory(request: ProcessLaunch) -> ManagedProcess:
    """Launch one fixed-argv process without a shell."""
    return subprocess.Popen(  # noqa: S603 - argv is typed and assembled from trusted config.
        request.argv,
        cwd=request.cwd,
        env=request.env,
        stdout=request.stdout,
        stderr=request.stderr,
        shell=request.shell,
        close_fds=request.close_fds,
    )


def bounded_log(raw: bytes) -> str:
    """Decode only the bounded tail exposed to the browser."""
    return raw[-LOG_BYTE_LIMIT:].decode("utf-8", errors="replace")
