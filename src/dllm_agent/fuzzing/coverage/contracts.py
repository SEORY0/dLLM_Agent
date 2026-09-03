"""LLVM coverage runner request and result contracts."""

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dllm_agent.targets.profiles import TargetProfile, require_target_profile


@dataclass(frozen=True, slots=True)
class CoverageRequest:
    """Inputs for one pinned local Docker LLVM coverage replay."""

    image_id: str
    source_root: Path
    harness_bytes: bytes
    corpus: tuple[bytes, ...]
    timeout_seconds: float
    target_profile: TargetProfile = "header-only"

    def __post_init__(self) -> None:
        """Reject target profiles outside the reviewed runtime set."""
        _ = require_target_profile(self.target_profile)


@dataclass(frozen=True, slots=True)
class CoverageTotals:
    """LLVM coverage percentages from report TOTAL rows."""

    regions: float
    functions: float
    lines: float
    branches: float


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Observed coverage build, replay, merge, and report results."""

    build_status: str
    run_status: str
    merge_status: str
    report_status: str
    build_log: bytes
    run_log: bytes
    merge_log: bytes
    report_log: bytes
    build_argv: tuple[str, ...]
    run_argv: tuple[str, ...]
    merge_argv: tuple[str, ...]
    report_argv: tuple[str, ...]
    coverage: CoverageTotals | None


class CommandRunner(Protocol):
    """Narrow subprocess seam for deterministic tests without Docker."""

    def __call__(
        self, command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one argv tuple with the supplied environment and timeout."""
        ...
