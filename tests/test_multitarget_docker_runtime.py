"""Hardened Magma container command contracts for campaign cells."""

from pathlib import Path

import pytest

from dllm_agent.experiments.multitarget.docker_runtime import (
    ContainerRequest,
    DockerRuntimeError,
    TargetId,
    container_argv,
    runtime_profile,
)

IMAGE_ID = "sha256:" + ("a" * 64)


@pytest.mark.parametrize(
    ("target_id", "program"),
    [
        ("libpng", "libpng_read_fuzzer"),
        ("libtiff", "tiff_read_rgba_fuzzer"),
        ("libxml2", "libxml2_xml_read_memory_fuzzer"),
    ],
)
def test_runtime_profile_binds_exact_magma_program(target_id: TargetId, program: str) -> None:
    # Given / When
    profile = runtime_profile(target_id)

    # Then
    assert profile.program == program
    assert profile.target_id == target_id
    assert profile.upstream_commit.isascii()
    assert len(profile.upstream_commit) == 40


def test_campaign_command_is_networkless_cpu_pinned_and_uses_exact_worker_budget(
    tmp_path: Path,
) -> None:
    # Given
    request = ContainerRequest(
        name="dllm-libpng-selective-revision-r0",
        profile=runtime_profile("libpng"),
        image_id=IMAGE_ID,
        shared_dir=tmp_path / "shared",
        corpus_dir=tmp_path / "corpus",
        duration_seconds=85_440,
        fuzzer_seed=202_608_210,
        cpu_index=7,
        poll_seconds=60,
    )

    # When
    argv = container_argv(request)

    # Then
    assert argv[:3] == ("docker", "run", "--detach")
    assert argv[argv.index("--network") : argv.index("--network") + 2] == (
        "--network",
        "none",
    )
    assert argv[argv.index("--cpuset-cpus") : argv.index("--cpuset-cpus") + 2] == (
        "--cpuset-cpus",
        "7",
    )
    assert "PROGRAM=libpng_read_fuzzer" in argv
    assert "TIMEOUT=85440s" in argv
    assert "POLL=60" in argv
    assert "FUZZARGS=-seed=202608210" in argv
    assert argv[-2] == IMAGE_ID
    assert argv[-1] == "/magma/targets/libpng/corpus/libpng_read_fuzzer"
    assert f"{request.shared_dir}:/magma_shared:rw" in argv
    assert argv.count("-seed=202608210") == 0
    assert any(
        item.endswith(":/magma/targets/libpng/corpus/libpng_read_fuzzer:rw") for item in argv
    )


def test_probe_command_uses_same_runtime_surface_with_thirty_second_budget(
    tmp_path: Path,
) -> None:
    # Given
    request = ContainerRequest(
        name="dllm-libxml2-probe-p0001",
        profile=runtime_profile("libxml2"),
        image_id=IMAGE_ID,
        shared_dir=tmp_path / "shared",
        corpus_dir=tmp_path / "corpus",
        duration_seconds=30,
        fuzzer_seed=19,
        cpu_index=3,
        poll_seconds=5,
    )

    # When
    argv = container_argv(request)

    # Then
    assert "TIMEOUT=30s" in argv
    assert "POLL=5" in argv
    assert "PROGRAM=libxml2_xml_read_memory_fuzzer" in argv
    assert argv[-1] == "/magma/targets/libxml2/corpus/libxml2_xml_read_memory_fuzzer"
    assert "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:symbolize=1" in argv


def test_zero_cpu_and_zero_fuzzer_seed_are_valid_deterministic_values(tmp_path: Path) -> None:
    # Given
    request = ContainerRequest(
        name="dllm-libtiff-zero-seed",
        profile=runtime_profile("libtiff"),
        image_id=IMAGE_ID,
        shared_dir=tmp_path / "shared",
        corpus_dir=tmp_path / "corpus",
        duration_seconds=30,
        fuzzer_seed=0,
        cpu_index=0,
        poll_seconds=5,
    )

    # When
    argv = container_argv(request)

    # Then
    assert "FUZZARGS=-seed=0" in argv
    assert argv[argv.index("--cpuset-cpus") + 1] == "0"


def test_invalid_image_digest_raises_the_typed_boundary_error(tmp_path: Path) -> None:
    # Given
    request = ContainerRequest(
        name="dllm-libpng-invalid-image",
        profile=runtime_profile("libpng"),
        image_id="libpng:latest",
        shared_dir=tmp_path / "shared",
        corpus_dir=tmp_path / "corpus",
        duration_seconds=30,
        fuzzer_seed=0,
        cpu_index=0,
        poll_seconds=5,
    )

    # When / Then
    with pytest.raises(DockerRuntimeError, match="sha256 digest"):
        _ = container_argv(request)
