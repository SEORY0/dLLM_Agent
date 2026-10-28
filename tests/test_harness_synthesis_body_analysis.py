"""Comment-aware lexical analysis: masking, right-context classes, calls, preservation."""

from dllm_agent.harness_synthesis.body_analysis import (
    MemberCall,
    active_member_calls,
    comment_text,
    commented_member_calls,
    constructed_object,
    executable_statements,
    lexical_edit_distance,
    line_kinds,
    mask_comments,
    right_context,
    unaffected_statements_preserved,
)

# Shapes taken from real RQ4 pilot bodies (InputFile seeds 12000000 and 12000004).
_COMMENT_ONLY_AFTER = (
    "    OPENEXR_IMF_NAMESPACE::InputFile inputFile(&stream);\n"
    "    // Optional inspection: Access the header\n"
    "    // OPENEXR_IMF_NAMESPACE::Header inputFile.header();\n"
    "  "
)
_CODE_AFTER = (
    "    OPENEXR_IMF_NAMESPACE::InputFile file(&stream);\n"
    "    // Optionally inspect the file's header\n"
    "    const ImfHeader& header = file.header();\n"
    "  "
)


def test_mask_comments_keeps_offsets_and_blanks_comments_and_literals() -> None:
    code = 'a(); // x.f()\nb("y.g()"); /* z.h(); */ c();'
    masked = mask_comments(code)
    assert len(masked) == len(code)
    assert masked.count("\n") == code.count("\n")
    assert "x.f" not in masked
    assert "y.g" not in masked
    assert "z.h" not in masked
    assert "c();" in masked


def test_comment_text_is_the_comment_complement_without_literals() -> None:
    code = 'f(); // keep.me()\ns = "not.me()";'
    text = comment_text(code)
    assert "keep.me()" in text
    assert "not.me" not in text
    assert len(text) == len(code)


def test_unterminated_block_comment_masks_to_end() -> None:
    assert mask_comments("a(); /* b.c();").strip() == "a();"


def test_line_kinds_distinguishes_blank_comment_and_code() -> None:
    assert line_kinds(_COMMENT_ONLY_AFTER) == ("code", "comment", "comment", "blank")


def test_right_context_separates_comment_only_from_code() -> None:
    # The pilot's has_right_context was True for both bodies.
    assert right_context(_COMMENT_ONLY_AFTER, 1) == "comment-only"
    assert right_context(_CODE_AFTER, 1) == "code"
    assert right_context(_CODE_AFTER, 3) == "none"


def test_active_calls_ignore_comments_and_commented_calls_find_them() -> None:
    assert active_member_calls(_COMMENT_ONLY_AFTER) == ()
    assert commented_member_calls(_COMMENT_ONLY_AFTER) == (MemberCall("inputFile", "header", 3),)
    assert active_member_calls(_CODE_AFTER) == (MemberCall("file", "header", 3),)


def test_constructed_object_needs_code_not_comment() -> None:
    assert constructed_object(_CODE_AFTER, "InputFile") == "file"
    assert (
        constructed_object("    // OPENEXR_IMF_NAMESPACE::InputFile f(stream);\n", "InputFile")
        is None
    )
    assert constructed_object("Imf::MultiPartInputFile mp{stream};", "MultiPartInputFile") == "mp"
    # Copy-initialised form, seen in the RQ4 pilot (InputFile seed 12000011, infill arm).
    copy_init = "    OPENEXR_IMF_NAMESPACE::InputFile f = OPENEXR_IMF_NAMESPACE::InputFile(stream);"
    assert constructed_object(copy_init, "InputFile") == "f"


def test_executable_statements_normalise_whitespace_and_drop_comments() -> None:
    assert executable_statements("  a  =  b; // c\n\n  // d\n  e();") == ("a = b;", "e();")


def test_unaffected_statements_preserved_excludes_the_defect_line() -> None:
    reference = "    A a(s);\n    int n = a.parts();\n    use(n);\n  "
    assert unaffected_statements_preserved(reference, reference, defect_line=2) == (2, 2)
    commented = "    A a(s);\n    // int n = a.parts();\n    // use(n);\n  "
    assert unaffected_statements_preserved(reference, commented, defect_line=2) == (1, 2)


def test_lexical_edit_distance_counts_tokens_not_characters() -> None:
    assert lexical_edit_distance("f(&stream);", "f(stream);") == 1
    assert lexical_edit_distance("a.readHeader();", "a.header();") == 1
    assert lexical_edit_distance("x(); // long comment", "x();") == 0
