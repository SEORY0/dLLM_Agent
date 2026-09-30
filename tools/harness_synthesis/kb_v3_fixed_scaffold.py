"""V3 -- fixed scaffold + model-generated body only (scope change, user-approved).

Redefines the task: not "write a full harness" (v1/v2, both of which failed
InputFile/MultiPartInputFile's gate twice), but "write the 1-4 statements that go
inside an already-fixed harness's try block". The scaffold -- includes, the
MemoryIStream class implementing IStream's exact contract, and the libFuzzer entry
point -- is fixed Python string data, never part of what the model generates, and
therefore cannot be corrupted by the model the way v1/v2's "copy this verbatim"
instruction still let happen. Only InputFile and MultiPartInputFile get a scaffold;
checkOpenEXRFile already clears every gate without one and has no IStream subclass
to fix.

The class body is copied byte-for-byte from kb_diversity_experiment._MEMORY_ISTREAM
(already independently re-verified against the real installed ImfIO.h -- see
docs/superpowers/plans/2026-09-29-trajectory-signal-harness-synthesis.md's RQ3-B0v2
section) rather than retyped here, so this module cannot introduce a fresh
transcription error at the one place this experiment most needs to be trustworthy.
"""

from kb_diversity_experiment import _MEMORY_ISTREAM

_HEADERS: dict[str, tuple[str, ...]] = {
    "InputFile": (
        "cstdint",
        "cstring",
        "exception",
        "stdexcept",
        "ImfIO.h",
        "ImfInputFile.h",
        "ImfNamespace.h",
    ),
    "MultiPartInputFile": (
        "cstdint",
        "cstring",
        "exception",
        "stdexcept",
        "ImfIO.h",
        "ImfMultiPartInputFile.h",
        "ImfNamespace.h",
    ),
}

_SUFFIX = "\n  } catch (const std::exception&) {\n  }\n  return 0;\n}\n"


def scaffold_prefix(example_name: str) -> str:
    """Everything before the model-written body: includes, class, entry point, `try {`."""
    includes = "\n".join(f"#include <{header}>" for header in _HEADERS[example_name])
    return (
        f"{includes}\n\n{_MEMORY_ISTREAM}"
        'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n'
        "  MemoryIStream stream(reinterpret_cast<const char*>(data), "
        "static_cast<int64_t>(size));\n"
        "  try {\n"
    )


def scaffold_suffix() -> str:
    """Everything after the model-written body: the try block's close, the function's."""
    return _SUFFIX


def assemble(example_name: str, body: str) -> str:
    """Build the full harness source: fixed prefix + generated body + fixed suffix."""
    return scaffold_prefix(example_name) + body + scaffold_suffix()


def placeholder_harness(example_name: str) -> str:
    """A trivial complete harness for verifying the scaffold compiles on its own."""
    return assemble(example_name, "    (void)0;")


def body_prompt(example_name: str) -> str:
    """Instructions for the model's turn.

    Used with --prefill-suffix so the actual continuation starts right after the
    fixed prefix's own "try {" line.
    """
    return (
        "You are completing a fixed C++ libFuzzer harness for OpenEXR's "
        f"{example_name}. A MemoryIStream implementing IStream, and an already-"
        "constructed `stream` object (not a pointer -- pass it as `stream`, not "
        "`&stream`), exist above you, inside the try block. Write 1 to 4 C++ "
        f"statements that construct an OPENEXR_IMF_NAMESPACE::{example_name} from "
        "`stream` and optionally inspect it (e.g. its header). Do not repeat any "
        "surrounding code, includes, or the class definition, and do not write the "
        "closing brace -- output only the statements themselves, each ending in a "
        "semicolon."
    )


def extract_body(generated_text: str) -> tuple[str, bool]:
    """Return ``(body_text, is_complete)``.

    Stops at the try block's own closing brace -- the first ``}`` whose depth,
    counted only within the generated text's own braces, is zero (any ``{`` the
    generated text opens must be closed before that point; the try block's opening
    ``{`` is in the fixed prefix, never in this text). Also stops at a stray code
    fence or stop token, which the model sometimes emits if it tries to close the
    whole harness itself. ``is_complete=False`` means neither was found -- the body
    could not be reliably extracted.
    """
    depth = 0
    index = 0
    length = len(generated_text)
    while index < length:
        if generated_text[index : index + 3] == "```" or generated_text[index : index + 2] == "<|":
            return generated_text[:index], False
        char = generated_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return generated_text[:index], True
            depth -= 1
        index += 1
    return generated_text, False
