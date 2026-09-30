# ruff: noqa: T201 -- deliberate console report, not a library module
"""GPU smoke: real Dream-Coder generation replayed through the production modules.

Runs a real generation via tools/dream_runtime/venv (a subprocess, since that venv
cannot import dllm_agent), then feeds its raw commit trajectory through the actual
harness_synthesis.trajectory_hook.TrajectoryRecorder, regions.statement_regions, and
remask.select_lowest_confidence_region — proving Task 4's bookkeeping accepts real
Dream data (not just the unit tests' fabricated fixtures) and that region-confidence
scoring reproduces the localization result from
docs/harness-synthesis-feasibility-20260920.md (Result 1: lowest-confidence region
often matches, but is not guaranteed to match, the first compile error line).

    uv run python tools/harness_synthesis/dream_backend_smoke.py [seed]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from random import Random

from _replay import decode_code_block, replay, run_dream_generate

from dllm_agent.harness_synthesis.api_knowledge import (
    ApiKnowledgeBase,
    UsageExample,
    render_prompt_text,
)
from dllm_agent.harness_synthesis.regions import statement_regions
from dllm_agent.harness_synthesis.remask import (
    region_confidence,
    select_lowest_confidence_region,
    select_random_region,
)

_USAGE_EXAMPLE = UsageExample(
    name="checkOpenEXRFile",
    headers=("cstdint", "ImfCheckFile.h", "ImfNamespace.h"),
    snippet=(
        "#include <cstdint>\n"
        "#include <ImfCheckFile.h>\n"
        "#include <ImfNamespace.h>\n\n"
        'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n'
        "  const char* bytes = reinterpret_cast<const char*>(data);\n"
        "  (void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);\n"
        "  return 0;\n"
        "}\n"
    ),
)
_KB = ApiKnowledgeBase(
    header_index=("cstdint", "ImfCheckFile.h", "ImfNamespace.h"),
    namespace_note=(
        "The namespace is OPENEXR_IMF_NAMESPACE (from ImfNamespace.h); there is no "
        "'OpenEXR' namespace. uint8_t/size_t need <cstdint>/<cstddef> -- OpenEXR's own "
        "headers do not reliably pull these in transitively."
    ),
    declarations=(),
)
_PROMPT = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    'Define extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) '
    "that feeds the fuzz bytes into OpenEXR parsing APIs and returns 0. Output only one "
    "```cpp code block.\n\n" + render_prompt_text(_KB, examples=(_USAGE_EXAMPLE,))
)


def main() -> None:
    """Generate once for real, replay it, and report what the production modules see."""
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4242
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_dream_generate(_PROMPT, seed, Path(tmp) / "dream.json")

    recorder = replay(payload)
    print(
        f"real Dream trajectory replayed with no bookkeeping error: "
        f"{len(recorder.traces)} commits recorded"
    )

    prompt_length = int(payload["prompt_length"])
    code, code_offsets = decode_code_block(payload)
    print("--- decoded harness ---")
    print(code)

    regions = statement_regions(code)
    print(f"\n{len(regions)} candidate regions detected")

    lowest = select_lowest_confidence_region(
        regions, code_offsets, recorder.traces, position_offset=prompt_length
    )
    random_region = select_random_region(regions, Random(seed))  # noqa: S311
    for region in regions:
        confidence = region_confidence(
            region, code_offsets, recorder.traces, position_offset=prompt_length
        )
        marker = " <- lowest-confidence" if region is lowest else ""
        marker += " <- random pick" if region is random_region else ""
        print(f"  line {region.line_number:3d} conf={confidence}: {region.text}{marker}")


if __name__ == "__main__":
    main()
