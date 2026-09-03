# ruff: noqa: D100,D101,D102,TC003

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CommandResult(Protocol):
    returncode: int


class CommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> CommandResult: ...


class CommandTimeoutError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _Completed:
    returncode: int


class SubprocessCommandRunner:
    def run(
        self,
        argv: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> _Completed:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_file,
                stderr_path.open("w", encoding="utf-8") as stderr_file,
            ):
                completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
                    argv,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as error:
            message = f"timeout after {timeout_seconds}s: {' '.join(argv)}"
            raise CommandTimeoutError(message) from error
        return _Completed(returncode=completed.returncode)
