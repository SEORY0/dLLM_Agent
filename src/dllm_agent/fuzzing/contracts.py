"""libFuzzer runner request and result contracts."""

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from dllm_agent.targets.profiles import TargetProfile, require_target_profile

type BuildStatus = Literal["succeeded", "failed", "timed_out"]
type FuzzStatus = Literal["completed", "crashed", "timed_out", "skipped"]
type SanitizerProfile = Literal["address,undefined", "address"]


@dataclass(frozen=True, slots=True)
class FuzzRunRequest:
    """Inputs for one pinned local Docker libFuzzer run."""

    image_id: str
    source_root: Path
    harness_bytes: bytes
    corpus: tuple[bytes, ...]
    duration_seconds: int
    max_len: int
    rss_limit_mb: int
    timeout_seconds: float
    sanitizers: SanitizerProfile = "address,undefined"
    fuzzer_seed: int = 0
    target_profile: TargetProfile = "header-only"

    def __post_init__(self) -> None:
        """Reject target profiles outside the reviewed runtime set."""
        _ = require_target_profile(self.target_profile)


@dataclass(frozen=True, slots=True)
class FuzzArtifact:
    """Immutable artifact bytes copied out of a fuzz run."""

    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class FuzzRunResult:
    """Observed compiler, build, fuzzing, and artifact results."""

    compiler_version_summary: bytes
    fuzzer_version_summary: bytes
    build_status: BuildStatus
    build_log: bytes
    fuzz_status: FuzzStatus
    fuzz_log: bytes
    corpus_artifacts: tuple[FuzzArtifact, ...]
    crash_artifacts: tuple[FuzzArtifact, ...]
    compiler_version_argv: tuple[str, ...]
    fuzzer_version_argv: tuple[str, ...]
    build_argv: tuple[str, ...]
    fuzz_argv: tuple[str, ...]
    sanitizers: SanitizerProfile = "address,undefined"


class CommandRunner(Protocol):
    """Narrow subprocess seam for deterministic tests without Docker."""

    def __call__(
        self, command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one argv tuple with the supplied environment and timeout."""
        ...
