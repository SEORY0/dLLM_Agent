"""OpenEXR API knowledge extraction and compiler-error hint lookup (Task 1b)."""

from dataclasses import dataclass
from subprocess import CompletedProcess

import pytest

from dllm_agent.harness_synthesis.api_knowledge import (
    ApiKnowledgeBase,
    ApiKnowledgeError,
    UsageExample,
    extract_header_index,
    extract_text,
    find_declaration,
    hint_for_missing_headers,
    render_prompt_text,
    suggest_existing_headers,
)


@dataclass(slots=True)
class FakeRuntime:
    """Returns a scripted process regardless of the argv it was called with."""

    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int = 0

    def __call__(self, argv: tuple[str, ...], _timeout: float) -> CompletedProcess[bytes]:
        return CompletedProcess(argv, self.returncode, stdout=self.stdout, stderr=self.stderr)


def test_extract_header_index_filters_to_imf_headers_only() -> None:
    # Given an image's include dir listing mixing Imf headers, non-Imf dirs, and Imath
    runtime = FakeRuntime(stdout=b"ImfIO.h\nOpenEXR\nImfHeader.h\nImath\nImfCheckFile.h\n")

    # When extracting the header index
    headers = extract_header_index(runtime, image="dllm-openexr-runtime", include_dir="/inc")

    # Then only Imf*.h names remain, sorted
    assert headers == ("ImfCheckFile.h", "ImfHeader.h", "ImfIO.h")


def test_extract_header_index_raises_on_nonzero_exit() -> None:
    runtime = FakeRuntime(returncode=1, stderr=b"no such directory")

    with pytest.raises(ApiKnowledgeError, match="no such directory"):
        extract_header_index(runtime, image="img", include_dir="/missing")


def test_extract_text_returns_decoded_stdout() -> None:
    runtime = FakeRuntime(stdout=b"IMF_EXPORT bool checkOpenEXRFile(...);")

    text = extract_text(runtime, image="img", path="/inc/ImfCheckFile.h")

    assert text == "IMF_EXPORT bool checkOpenEXRFile(...);"


def test_extract_text_raises_on_nonzero_exit() -> None:
    runtime = FakeRuntime(returncode=1, stderr=b"no such file")

    with pytest.raises(ApiKnowledgeError, match="no such file"):
        extract_text(runtime, image="img", path="/missing.h")


def test_find_declaration_normalizes_whitespace_in_a_multiline_match() -> None:
    header_text = "junk\nIMF_EXPORT bool\ncheckOpenEXRFile (\n  const char*, int);\nmore junk"

    declaration = find_declaration(header_text, r"IMF_EXPORT bool\s+checkOpenEXRFile[^;]*;")

    assert declaration == "IMF_EXPORT bool checkOpenEXRFile ( const char*, int);"


def test_find_declaration_returns_none_when_pattern_absent() -> None:
    assert find_declaration("nothing relevant here", r"checkOpenEXRFile[^;]*;") is None


def test_render_prompt_text_includes_header_index_and_declarations() -> None:
    kb = ApiKnowledgeBase(
        header_index=("ImfCheckFile.h", "ImfIO.h"),
        namespace_note="The namespace is OPENEXR_IMF_NAMESPACE; there is no 'OpenEXR' namespace.",
        declarations=("IMF_EXPORT bool checkOpenEXRFile(const char*, int);",),
    )

    text = render_prompt_text(kb)

    assert "ImfCheckFile.h, ImfIO.h" in text
    assert "OPENEXR_IMF_NAMESPACE" in text
    assert "checkOpenEXRFile(const char*, int);" in text


def test_render_prompt_text_appends_verified_usage_examples() -> None:
    kb = ApiKnowledgeBase(header_index=(), namespace_note="", declarations=())
    example = UsageExample(
        name="checkOpenEXRFile", headers=("ImfCheckFile.h",), snippet="checkOpenEXRFile(a, b);\n"
    )

    text = render_prompt_text(kb, examples=(example,))

    assert "checkOpenEXRFile" in text
    assert "checkOpenEXRFile(a, b);" in text


def test_render_prompt_text_omits_example_block_when_none_given() -> None:
    kb = ApiKnowledgeBase(header_index=(), namespace_note="", declarations=())

    text = render_prompt_text(kb)

    assert "Verified usage example" not in text


def test_suggest_existing_headers_finds_a_close_match() -> None:
    index = ("ImfInputFile.h", "ImfIO.h", "ImfHeader.h")

    suggestions = suggest_existing_headers("ImfInputFilee.h", index)

    assert "ImfInputFile.h" in suggestions


def test_suggest_existing_headers_returns_empty_for_an_unrelated_name() -> None:
    index = ("ImfInputFile.h", "ImfIO.h")

    assert suggest_existing_headers("CompletelyUnrelatedName.h", index) == ()


def test_hint_for_missing_headers_names_the_missing_header_and_a_near_match() -> None:
    index = ("ImfInputFile.h", "ImfIO.h")
    error = "harness.cc:5:10: fatal error: 'ImfInputFilee.h' file not found"

    hint = hint_for_missing_headers(error, index)

    assert "'ImfInputFilee.h' does not exist" in hint
    assert "ImfInputFile.h" in hint


def test_hint_for_missing_headers_handles_multiple_missing_headers() -> None:
    index = ("ImfIO.h",)
    error = (
        "harness.cc:1:10: fatal error: 'ImfStream.h' file not found\n"
        "harness.cc:2:10: fatal error: 'ImfMemory.h' file not found"
    )

    hint = hint_for_missing_headers(error, index)

    assert "ImfStream.h" in hint
    assert "ImfMemory.h" in hint


def test_hint_for_missing_headers_returns_empty_string_when_nothing_matches() -> None:
    error = "harness.cc:8:11: error: using declaration requires a qualified name"

    assert hint_for_missing_headers(error, header_index=()) == ""
