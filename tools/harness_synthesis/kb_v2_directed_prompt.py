"""KB/prompt improvement v2 (one round only) for the `directed` framing.

Separate from ``kb_diversity_experiment.py``'s v1 prompt (Task 8/8b, RQ3 smoke, and
RQ3-B0 all used v1 -- none of that is touched). v2 changes ONLY the instruction text
around the examples; the example code itself (``kb_diversity_experiment._EXAMPLES``,
including ``_MEMORY_ISTREAM``) is reused unchanged, because it was independently
re-verified against the real installed header (``ImfIO.h`` inside
``dllm-openexr-runtime``) to already be contractually correct -- see
``_ISTREAM_CONTRACT`` below for the exact re-verification. The problem RQ3-B0 found
was not a *wrong* example; it was the model deviating from a *correct* example when
generating (wrong return type on an override, a missed pure-virtual method, etc.).
v2's one change is to state that contract explicitly and instruct verbatim copying,
rather than replacing or hand-writing any part of IStream on the caller's behalf.
This stays within prompt engineering -- if a literal, code-injected fixed IStream
(bypassing model generation for that part) turns out to be necessary, that is a
scope change to report and get confirmation on, not something to apply here.
"""

from kb_diversity_experiment import _EXAMPLES, _kb_and_examples

from dllm_agent.harness_synthesis.api_knowledge import render_prompt_text

_BASE_PROMPT = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    'Define extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) '
    "that feeds the fuzz bytes into OpenEXR parsing APIs and returns 0. Output only "
    "one ```cpp code block."
)

# Re-verified directly against /opt/openexr-coverage/include/OpenEXR/ImfIO.h inside
# the pinned dllm-openexr-runtime image (grep "virtual|= 0", 2026-09-30): exactly
# three methods are pure virtual (`= 0`) on IStream. Every other method
# (isMemoryMapped, readMemoryMapped, clear, size, isStatelessRead) has a default
# implementation and must NOT be overridden by a minimal subclass.
_ISTREAM_CONTRACT = (
    "OpenEXR's IStream (ImfIO.h) declares exactly three pure virtual methods that a "
    "subclass MUST override, with exactly these signatures (verified against the "
    "installed header):\n"
    "  bool read(char c[], int n) override;\n"
    "  uint64_t tellg() override;\n"
    "  void seekg(uint64_t pos) override;\n"
    "No other IStream method needs to be overridden."
)

_ENTRY_POINT_CONTRACT = (
    'The only required entry point is exactly: extern "C" int '
    "LLVMFuzzerTestOneInput(const uint8_t* data, size_t size). Any other spelling "
    '(wrong linkage, wrong parameter types, a missing extern "C") makes the binary '
    "fail to link even if every OpenEXR call is otherwise correct."
)


def _v2_instruction(example_name: str, uses_istream: bool) -> str:
    if not uses_istream:
        return (
            f"\n\nFor this specific task, use the '{example_name}' example above "
            "exactly as written: copy it character-for-character, changing nothing."
        )
    return (
        f"\n\n{_ISTREAM_CONTRACT}\n\n{_ENTRY_POINT_CONTRACT}\n\n"
        f"For this specific task, use the '{example_name}' example above as your "
        "reference implementation. Copy its MemoryIStream class definition "
        "character-for-character -- every method signature, the 'override' keyword "
        "on read/tellg/seekg, and nothing added or removed -- and copy its "
        f"{example_name} call the same way. Do not modify any signature."
    )


def prompt_v2(example_name: str) -> str:
    """Build the v2 `directed` prompt for one of ``_EXAMPLES`` by name."""
    example = next(candidate for candidate in _EXAMPLES if candidate.name == example_name)
    kb, examples = _kb_and_examples(example, "directed")
    uses_istream = "ImfIO.h" in example.headers
    return (
        _BASE_PROMPT
        + "\n\n"
        + render_prompt_text(kb, examples=examples)
        + _v2_instruction(example.name, uses_istream)
    )
