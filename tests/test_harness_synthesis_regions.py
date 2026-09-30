"""Statement-boundary detection: regex/paren-matching, not an AST (spec §5)."""

from dllm_agent.harness_synthesis.regions import region_at_line, statement_regions

_HARNESS = """\
#include <cstddef>
#include <ImfCheckFile.h>

// checks the file twice, tolerant and strict
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const char* bytes = reinterpret_cast<const char*>(data);
  (void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);
  return 0;
}
"""


def test_statement_regions_includes_every_include_line() -> None:
    regions = statement_regions(_HARNESS)

    includes = [r for r in regions if r.text.startswith("#include")]
    assert [r.text for r in includes] == ["#include <cstddef>", "#include <ImfCheckFile.h>"]


def test_statement_regions_excludes_the_entry_point_signature() -> None:
    regions = statement_regions(_HARNESS)

    assert not any("LLVMFuzzerTestOneInput" in r.text for r in regions)


def test_statement_regions_excludes_a_comment_only_line() -> None:
    regions = statement_regions(_HARNESS)

    assert not any("checks the file" in r.text for r in regions)


def test_statement_regions_excludes_brace_and_blank_lines() -> None:
    regions = statement_regions(_HARNESS)

    assert not any(r.text in ("{", "}", "") for r in regions)


def test_statement_regions_includes_semicolon_terminated_body_statements() -> None:
    regions = statement_regions(_HARNESS)

    body = [r.text for r in regions if not r.text.startswith("#include")]
    assert body == [
        "const char* bytes = reinterpret_cast<const char*>(data);",
        "(void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);",
        "return 0;",
    ]


def test_statement_regions_char_span_slices_back_to_the_original_line() -> None:
    regions = statement_regions(_HARNESS)

    first_include = next(r for r in regions if r.text.startswith("#include"))
    assert _HARNESS[first_include.char_start : first_include.char_end] == "#include <cstddef>"


def test_statement_regions_line_numbers_are_one_based() -> None:
    regions = statement_regions(_HARNESS)

    assert regions[0].line_number == 1


def test_region_at_line_finds_the_region_at_a_compiler_error_line() -> None:
    regions = statement_regions(_HARNESS)

    found = region_at_line(regions, line_number=6)
    assert found is not None
    assert found.text.startswith("const char* bytes")


def test_region_at_line_returns_none_for_a_non_region_line() -> None:
    regions = statement_regions(_HARNESS)

    # Line 5 is the excluded entry-point signature.
    assert region_at_line(regions, line_number=5) is None
