"""Hardened Docker command construction for Magma multitarget runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, override

if TYPE_CHECKING:
    from pathlib import Path

type TargetId = Literal["libpng", "libtiff", "libxml2"]

_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_LOG_SIZE_BYTES: Final = 20_971_520


@dataclass(slots=True)
class DockerRuntimeError(Exception):
    """Raised when a Magma Docker runtime request is unsafe or invalid."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Pinned Magma target, program, and upstream source revision."""

    target_id: TargetId
    program: str
    upstream_commit: str


@dataclass(frozen=True, slots=True)
class ContainerRequest:
    """Boundary object for one Magma Docker container launch."""

    name: str
    profile: RuntimeProfile
    image_id: str
    shared_dir: Path
    corpus_dir: Path
    duration_seconds: int
    fuzzer_seed: int
    cpu_index: int
    poll_seconds: int


def runtime_profile(target_id: TargetId) -> RuntimeProfile:
    """Return the exact pinned Magma profile for a supported target."""
    match target_id:
        case "libpng":
            return RuntimeProfile(
                target_id=target_id,
                program="libpng_read_fuzzer",
                upstream_commit="a37d4836519517bdce6cb9d956092321eca3e73b",
            )
        case "libtiff":
            return RuntimeProfile(
                target_id=target_id,
                program="tiff_read_rgba_fuzzer",
                upstream_commit="c145a6c14978f73bb484c955eb9f84203efcb12e",
            )
        case "libxml2":
            return RuntimeProfile(
                target_id=target_id,
                program="libxml2_xml_read_memory_fuzzer",
                upstream_commit="ec6e3efb06d7b15cf5a2328fabd3845acea4c815",
            )


def container_argv(request: ContainerRequest) -> tuple[str, ...]:
    """Build a shell-free, bounded `docker run` argv for one Magma worker."""
    _validate_request(request)
    return (
        "docker",
        "run",
        "--detach",
        "--name",
        request.name,
        "--network",
        "none",
        "--cpuset-cpus",
        str(request.cpu_index),
        "--pids-limit",
        "256",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--env",
        f"ASAN_OPTIONS={_ASAN_OPTIONS}",
        "--env",
        f"PROGRAM={request.profile.program}",
        "--env",
        "ARGS=",
        "--env",
        f"FUZZARGS=-seed={request.fuzzer_seed}",
        "--env",
        f"POLL={request.poll_seconds}",
        "--env",
        f"TIMEOUT={request.duration_seconds}s",
        "--env",
        f"LOGSIZE={_LOG_SIZE_BYTES}",
        "--volume",
        f"{request.shared_dir}:/magma_shared:rw",
        "--volume",
        _corpus_mount(request),
        "--pull",
        "never",
        request.image_id,
        _container_corpus_path(request),
    )


def _corpus_mount(request: ContainerRequest) -> str:
    return f"{request.corpus_dir}:{_container_corpus_path(request)}:rw"


def _container_corpus_path(request: ContainerRequest) -> str:
    return f"/magma/targets/{request.profile.target_id}/corpus/{request.profile.program}"


def _validate_request(request: ContainerRequest) -> None:
    if _IMAGE_DIGEST.fullmatch(request.image_id) is None:
        raise DockerRuntimeError(reason="image_id must be a sha256 digest")
    if _CONTAINER_NAME.fullmatch(request.name) is None:
        raise DockerRuntimeError(reason="container name is not docker-safe")
    if request.duration_seconds <= 0:
        raise DockerRuntimeError(reason="duration_seconds must be positive")
    if request.fuzzer_seed < 0:
        raise DockerRuntimeError(reason="fuzzer_seed must be nonnegative")
    if request.cpu_index < 0:
        raise DockerRuntimeError(reason="cpu_index must be nonnegative")
    if request.poll_seconds <= 0:
        raise DockerRuntimeError(reason="poll_seconds must be positive")
    if not request.shared_dir.is_absolute():
        raise DockerRuntimeError(reason="shared_dir must be absolute")
    if not request.corpus_dir.is_absolute():
        raise DockerRuntimeError(reason="corpus_dir must be absolute")
