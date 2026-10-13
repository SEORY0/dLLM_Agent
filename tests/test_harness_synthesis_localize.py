"""Compiler-diagnostic parsing, body-line localization, and covering-window construction."""

from dllm_agent.harness_synthesis.localize import (
    CompilerDiagnostic,
    body_line_regions,
    locate_in_body,
    parse_compiler_diagnostics,
    window_covering,
)
from dllm_agent.harness_synthesis.regions import SourceRegion
from dllm_agent.harness_synthesis.remask import TokenWindow

# A real RQ3-v3 pilot failure (seed 10000004): the scaffold prefix is 34 lines, so the
# body's first line is harness line 35.
_PREFIX = "x\n" * 34
_BODY = (
    "    OPENEXR_IMF_NAMESPACE::InputFile file(&stream);\n"
    "    // Optionally inspect the file's header\n"
    "    const ImfHeader& header = file.header();\n"
    "  "
)
_ERROR = (
    "harness.cc:35:38: error: no matching constructor for initialization of 'Imf::InputFile'\n"
    "harness.cc:37:11: error: unknown type name 'ImfHeader'"
)


def test_parse_compiler_diagnostics_reads_every_harness_error_in_order() -> None:
    assert parse_compiler_diagnostics(_ERROR) == (
        CompilerDiagnostic(
            35, 38, "no matching constructor for initialization of 'Imf::InputFile'"
        ),
        CompilerDiagnostic(37, 11, "unknown type name 'ImfHeader'"),
    )


def test_parse_compiler_diagnostics_accepts_fatal_errors_and_path_prefixes() -> None:
    text = "/work/harness.cc:3:10: fatal error: 'ImfStream.h' file not found"
    assert parse_compiler_diagnostics(text) == (
        CompilerDiagnostic(3, 10, "'ImfStream.h' file not found"),
    )


def test_parse_compiler_diagnostics_skips_other_files_notes_and_warnings() -> None:
    text = (
        "/opt/include/ImfInputFile.h:80:5: error: in header\n"
        "harness.cc:4:1: note: candidate constructor\n"
        "harness.cc:5:2: warning: unused variable\n"
        "myharness.cc:6:1: error: different file\n"
    )
    assert parse_compiler_diagnostics(text) == ()


def test_body_line_regions_skips_blank_lines_but_keeps_comments_and_partial_lines() -> None:
    regions = body_line_regions(_BODY)
    assert [region.line_number for region in regions] == [1, 2, 3]
    assert regions[1].text == "// Optionally inspect the file's header"
    for region in regions:
        assert _BODY[region.char_start : region.char_end].strip() == region.text


def test_locate_in_body_maps_the_first_error_onto_the_body_line() -> None:
    first = parse_compiler_diagnostics(_ERROR)[0]
    region = locate_in_body(first.line, prefix=_PREFIX, body=_BODY)
    assert region is not None
    assert region.line_number == 1
    assert region.text == "OPENEXR_IMF_NAMESPACE::InputFile file(&stream);"


def test_locate_in_body_returns_none_for_prefix_suffix_and_blank_lines() -> None:
    assert locate_in_body(34, prefix=_PREFIX, body=_BODY) is None  # last scaffold line
    assert locate_in_body(38, prefix=_PREFIX, body=_BODY) is None  # whitespace-only line
    assert locate_in_body(40, prefix=_PREFIX, body=_BODY) is None  # past the body


def test_window_covering_spans_every_overlapping_token_contiguously() -> None:
    region = SourceRegion(line_number=1, text="ab;", char_start=2, char_end=5)
    offsets = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8)]
    assert window_covering(region, offsets) == TokenWindow(start=1, length=2)
    assert window_covering(region, offsets, position_offset=100) == TokenWindow(101, 2)


def test_window_covering_includes_a_token_straddling_the_region_edge() -> None:
    region = SourceRegion(line_number=1, text="b", char_start=3, char_end=4)
    assert window_covering(region, [(0, 2), (2, 4), (4, 6)]) == TokenWindow(start=1, length=1)


def test_window_covering_returns_none_when_no_token_overlaps() -> None:
    region = SourceRegion(line_number=9, text="x;", char_start=50, char_end=52)
    assert window_covering(region, [(0, 4), (4, 8)]) is None
