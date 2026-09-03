"""Typed structured-particle contracts for the three Magma targets."""

import hashlib
import struct
import xml.etree.ElementTree as ET

import pytest

from dllm_agent.targets.structured import TARGET_IDS, load_structured_target


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_target_materializer_produces_at_least_sixteen_unique_snapshots(
    target_id: str,
) -> None:
    # Given
    target = load_structured_target(target_id)
    raw_selectors = tuple(
        tuple((particle_index // (4**slot_index)) % 4 for slot_index in range(10))
        for particle_index in range(32)
    )

    # When
    snapshots = tuple(target.materialize(selectors) for selectors in raw_selectors)

    # Then
    assert len(target.field_order) == 10
    assert len({snapshot.sha256 for snapshot in snapshots}) >= 16
    assert all(
        snapshot.sha256 == hashlib.sha256(snapshot.content).hexdigest() for snapshot in snapshots
    )
    assert all(
        {span.field for span in snapshot.field_spans} == set(target.field_order)
        for snapshot in snapshots
    )
    assert all(
        0 <= span.start < span.end <= len(snapshot.content)
        for snapshot in snapshots
        for span in snapshot.field_spans
    )


def test_libpng_particle_has_valid_signature_chunks_and_crc() -> None:
    # Given
    target = load_structured_target("libpng")

    # When
    snapshot = target.materialize((0, 1, 2, 3, 0, 1, 2, 3, 0, 1))

    # Then
    assert snapshot.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert snapshot.content.endswith(b"IEND\xaeB`\x82")
    assert b"IHDR" in snapshot.content
    assert b"IDAT" in snapshot.content


def test_libtiff_particle_has_little_endian_ifd_with_in_bounds_pixel_data() -> None:
    # Given
    target = load_structured_target("libtiff")

    # When
    snapshot = target.materialize((1, 2, 3, 0, 1, 2, 3, 0, 1, 2))
    ifd_offset = struct.unpack_from("<I", snapshot.content, 4)[0]
    entry_count = struct.unpack_from("<H", snapshot.content, ifd_offset)[0]

    # Then
    assert snapshot.content[:4] == b"II*\x00"
    assert ifd_offset == 8
    assert entry_count == 10
    assert len(snapshot.content) > ifd_offset + 2 + (entry_count * 12) + 4


def test_libtiff_payload_selector_is_bound_to_the_payload_field_span() -> None:
    # Given
    target = load_structured_target("libtiff")
    baseline = target.materialize((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # When
    changed = target.materialize((0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
    changed_offsets = {
        index
        for index, (left, right) in enumerate(zip(baseline.content, changed.content, strict=True))
        if left != right
    }
    payload_spans = tuple(span for span in changed.field_spans if span.field == "image_payload")

    # Then
    assert "image_payload" in target.field_order
    assert len(payload_spans) == 1
    assert changed_offsets
    assert changed_offsets <= set(range(payload_spans[0].start, payload_spans[0].end))


def test_libxml2_particle_is_well_formed_xml() -> None:
    # Given
    target = load_structured_target("libxml2")

    # When
    snapshot = target.materialize((2, 1, 3, 0, 2, 1, 3, 0, 2, 1))
    root = ET.fromstring(snapshot.content)  # noqa: S314 - owned deterministic fixture.

    # Then
    assert root.tag.endswith("root")
    assert snapshot.content.startswith(b"<?xml")


def test_libxml2_child_count_span_points_into_generated_child_content() -> None:
    # Given
    target = load_structured_target("libxml2")

    # When
    snapshot = target.materialize((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    child_count_span = next(span for span in snapshot.field_spans if span.field == "child_count")

    # Then
    assert child_count_span.start > snapshot.content.index(b"<root")
    assert snapshot.content[child_count_span.start : child_count_span.end] == b"0"
