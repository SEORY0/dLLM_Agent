"""OpenEXR structured particle materializer tests."""

import hashlib
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from dllm_agent.fuzzing.execution import run_command
from dllm_agent.targets.openexr.particle import (
    FIELD_ORDER,
    FIELD_SELECTOR_CARDINALITIES,
    MATERIALIZED_MUTABLE_FIELDS,
    FieldName,
    MaterializedSnapshot,
    ParticleMode,
    decode_particle,
    materialize_particle,
)

type FixtureSelectors = tuple[int, int, int, int, int, int, int, int, int, int]

OPENEXR_IMAGE_ID = "sha256:de7bef7b8c28a04e7c1c73e909a6e52958eada1a4538b03c289396556d139531"
ACCEPTING_HARNESS = b"""#include <cstddef>
#include <cstdint>
#include <ImfNamespace.h>
#include <ImfCheckFile.h>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
bool exr = OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(
  reinterpret_cast<const char*>(data), size, true, true, false);
bool core = OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(
  reinterpret_cast<const char*>(data), size, true, true, true);
if (!exr || !core) {
  __builtin_trap();
}
return 0;
}
"""


class FixtureSpan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)

    field: FieldName
    start: int
    end: int


class OwnedFixture(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["dllm-openexr-particle-fixture/v1"]
    field_order: tuple[FieldName, ...]
    selectors: FixtureSelectors
    mode: ParticleMode
    sha256: str
    anchor_sha256: str
    content_hex: str
    field_spans: tuple[FixtureSpan, ...]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_materialize_particle_keeps_openexr_anchors_and_derives_offsets() -> None:
    # Given
    particle = decode_particle((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "validity-preserving")

    # When
    snapshot = materialize_particle(particle)

    # Then
    assert snapshot.content[:8] == bytes.fromhex("762f310102000000")
    assert snapshot.sha256 == sha256_bytes(snapshot.content)
    assert {span.field for span in snapshot.field_spans} == set(FIELD_ORDER)
    assert all(span.end <= len(snapshot.content) for span in snapshot.field_spans)
    assert snapshot.fault_field is None


def test_field_selector_contract_identifies_validity_preserving_mutable_fields() -> None:
    # Given / When / Then
    assert FIELD_SELECTOR_CARDINALITIES == {
        "origin_x": 4,
        "origin_y": 4,
        "width": 4,
        "height": 4,
        "compression": 1,
        "pixel_type": 1,
        "line_order": 1,
        "x_sampling": 1,
        "payload_pattern": 4,
        "chunk_offset": 4,
    }
    assert MATERIALIZED_MUTABLE_FIELDS == (
        "origin_x",
        "origin_y",
        "width",
        "height",
        "payload_pattern",
    )


def test_boundary_breaking_changes_only_declared_derived_condition() -> None:
    # Given
    valid = materialize_particle(decode_particle((0,) * 10, "validity-preserving"))

    # When
    broken = materialize_particle(decode_particle((0,) * 10, "boundary-breaking"))

    # Then
    assert valid.anchor_sha256 == broken.anchor_sha256
    assert valid.content != broken.content
    assert broken.fault_field == "chunk_offset"


def test_representative_particle_hashes_are_stable_after_relocation() -> None:
    # Given
    selectors = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # When
    valid = materialize_particle(decode_particle(selectors, "validity-preserving"))
    broken = materialize_particle(decode_particle(selectors, "boundary-breaking"))

    # Then
    assert valid.sha256 == "3dacb4a1977a7e584b50b0359297253b475ff1d18984020abf980bca605cd1e2"
    assert broken.sha256 == "ad7b86791b142402792439a0e0f5d229c7599f5900608736a5ef989a2c157ce7"
    assert valid.anchor_sha256 == broken.anchor_sha256
    assert valid.anchor_sha256 == "ae2bfa268cf31e5724adf2b06fa686e0db285a66704d320189cdc2de9f45332c"
    assert len(valid.content) == len(broken.content) == 661


def test_owned_fixture_matches_materialized_particle_contract() -> None:
    # Given
    fixture_path = Path("fixtures/poc_revision/owned_exr_particle_v1.json")
    fixture = OwnedFixture.model_validate_json(fixture_path.read_text(encoding="utf-8"))
    snapshot = materialize_particle(decode_particle(fixture.selectors, fixture.mode))

    # When / Then
    assert fixture.field_order == FIELD_ORDER
    assert fixture.sha256 == snapshot.sha256
    assert fixture.anchor_sha256 == snapshot.anchor_sha256
    assert fixture.content_hex == snapshot.content.hex()


def test_validity_preserving_fixture_is_accepted_by_pinned_openexr_parser() -> None:
    # Given
    snapshot = materialize_particle(
        decode_particle((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "validity-preserving")
    )
    with tempfile.TemporaryDirectory(prefix="dllm-openexr-parser-") as directory:
        work_root = Path(directory)
        work_root.chmod(0o777)
        harness = work_root / "harness.cc"
        seed = work_root / "seed-0000"
        _ = harness.write_bytes(ACCEPTING_HARNESS)
        _ = seed.write_bytes(snapshot.content)
        harness.chmod(0o444)
        seed.chmod(0o444)

        # When
        build = run_command((*_docker_prefix(work_root), "/opt/dllm/bin/build-fuzz"), _env(), 60.0)
        run = run_command(
            (*_docker_prefix(work_root), "/work/fuzz_target", "/work/seed-0000", "-runs=1"),
            _env(),
            30.0,
        )

    # Then
    assert build.returncode == 0, (build.stdout + build.stderr).decode("utf-8", errors="replace")
    assert run.returncode == 0, (run.stdout + run.stderr).decode("utf-8", errors="replace")
    assert b"ERROR: AddressSanitizer" not in run.stdout + run.stderr


def test_box_selector_changes_are_covered_by_declared_spans_and_stable_anchors() -> None:
    # Given
    baseline = materialize_particle(
        decode_particle((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), "validity-preserving")
    )

    for selectors in (
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ):
        changed = materialize_particle(decode_particle(selectors, "validity-preserving"))

        # When
        changed_offsets = _changed_offsets(
            baseline.content[: _chunk_offset_start(baseline)],
            changed.content[: _chunk_offset_start(changed)],
        )
        declared_offsets = _declared_offsets(baseline) | _declared_offsets(changed)

        # Then
        assert changed_offsets <= declared_offsets
        assert changed.anchor_sha256 == baseline.anchor_sha256


def test_chunk_offset_selector_controls_boundary_breaking_fault_variant() -> None:
    # Given
    base_selectors = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    fault_selectors = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

    # When
    valid_base = materialize_particle(decode_particle(base_selectors, "validity-preserving"))
    valid_variant = materialize_particle(decode_particle(fault_selectors, "validity-preserving"))
    broken_base = materialize_particle(decode_particle(base_selectors, "boundary-breaking"))
    broken_variant = materialize_particle(decode_particle(fault_selectors, "boundary-breaking"))

    # Then
    assert valid_base.content == valid_variant.content
    assert broken_base.content != broken_variant.content
    assert broken_base.fault_field == "chunk_offset"
    assert broken_variant.fault_field == "chunk_offset"


def _declared_offsets(snapshot: MaterializedSnapshot) -> set[int]:
    offsets: set[int] = set()
    for span in snapshot.field_spans:
        offsets.update(range(span.start, span.end))
    return offsets


def _changed_offsets(left: bytes, right: bytes) -> set[int]:
    offsets = {
        index for index, value in enumerate(left) if index < len(right) and value != right[index]
    }
    offsets.update(range(min(len(left), len(right)), max(len(left), len(right))))
    return offsets


def _chunk_offset_start(snapshot: MaterializedSnapshot) -> int:
    for span in snapshot.field_spans:
        if span.field == "chunk_offset":
            return span.start
    raise AssertionError


def _env() -> dict[str, str]:
    return {
        "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1:symbolize=1",
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        "EXR_CHECK_ENABLE_PRINTS": "1",
    }


def _docker_prefix(work_root: Path) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        "128",
        "--memory",
        "2048m",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",  # noqa: S108 - Docker tmpfs mount.
        "--env",
        "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:symbolize=1",
        "--env",
        "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1",
        "--env",
        "EXR_CHECK_ENABLE_PRINTS=1",
        "-v",
        f"{Path.cwd().resolve()}:/target:ro",
        "-v",
        f"{work_root}:/work:rw",
        "-w",
        "/work",
        "--pull",
        "never",
        OPENEXR_IMAGE_ID,
    )
