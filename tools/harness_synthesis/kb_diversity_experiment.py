# ruff: noqa: T201 -- deliberate console report, not a library module
r"""Task 8 ("a안"): does a shared KB diversify, or does every harness collapse onto one example?

Three hand-authored, independently compile-verified OpenEXR call patterns
(``_EXAMPLES`` below -- each confirmed against ``harness-compile-only`` before this
script existed, never derived from model output per ``api_knowledge.py``'s design
principle). Two KB conditions per example: ``solo`` (only that example in the prompt)
and ``shared`` (all three examples in the prompt). For each (example, condition), N
seeds generate a harness, which gets compiled (Task 5a) and diffed against every
example's own snippet (``difflib`` ratio -- 1.0 is a verbatim copy). Every iteration is
recorded through the real audit ledger (Task 6).

    uv run python tools/harness_synthesis/kb_diversity_experiment.py \\
      --gpu 0 --seed-start 0 --seed-count 8 --out results-gpu0.json \\
      --ledger /data/dllm-agent/harness-synthesis/kb-diversity-ledger.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import tempfile
import time
from pathlib import Path

from _compile import compile_harness
from _replay import decode_code_block, replay, run_dream_generate

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.api_knowledge import (
    ApiKnowledgeBase,
    UsageExample,
    render_prompt_text,
)
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.targets.openexr.harness_reference import load_reference_harness

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _REPO_ROOT / "tools/openexr_runtime/poc_probe.cc"

_BASE_PROMPT = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    'Define extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) '
    "that feeds the fuzz bytes into OpenEXR parsing APIs and returns 0. Output only "
    "one ```cpp code block."
)
_NAMESPACE_NOTE = (
    "The namespace is OPENEXR_IMF_NAMESPACE (from ImfNamespace.h); there is no "
    "'OpenEXR' namespace. uint8_t/size_t need <cstdint>/<cstddef> -- OpenEXR's own "
    "headers do not reliably pull these in transitively."
)

_MEMORY_ISTREAM = """\
class MemoryIStream : public OPENEXR_IMF_NAMESPACE::IStream {
 public:
  MemoryIStream(const char* data, int64_t size)
      : OPENEXR_IMF_NAMESPACE::IStream("<memory>"), _data(data), _size(size), _pos(0) {}

  bool read(char c[], int n) override {
    if (_pos + n > _size) {
      throw std::runtime_error("read past end");
    }
    std::memcpy(c, _data + _pos, static_cast<size_t>(n));
    _pos += n;
    return _pos < _size;
  }

  uint64_t tellg() override { return static_cast<uint64_t>(_pos); }
  void seekg(uint64_t pos) override { _pos = static_cast<int64_t>(pos); }

 private:
  const char* _data;
  int64_t _size;
  int64_t _pos;
};

"""

_EXAMPLES = (
    UsageExample(
        name="checkOpenEXRFile",
        headers=("cstdint", "ImfCheckFile.h", "ImfNamespace.h"),
        snippet=(
            "#include <cstdint>\n#include <ImfCheckFile.h>\n#include <ImfNamespace.h>\n\n"
            'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n'
            "  const char* bytes = reinterpret_cast<const char*>(data);\n"
            "  (void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);\n"
            "  return 0;\n}\n"
        ),
    ),
    UsageExample(
        name="InputFile",
        headers=("cstdint", "cstring", "exception", "stdexcept", "ImfIO.h", "ImfInputFile.h",
                  "ImfNamespace.h"),
        snippet=(
            "#include <cstdint>\n#include <cstring>\n#include <exception>\n"
            "#include <stdexcept>\n\n#include <ImfIO.h>\n#include <ImfInputFile.h>\n"
            "#include <ImfNamespace.h>\n\n" + _MEMORY_ISTREAM +
            'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n'
            "  MemoryIStream stream(reinterpret_cast<const char*>(data), "
            "static_cast<int64_t>(size));\n"
            "  try {\n    OPENEXR_IMF_NAMESPACE::InputFile file(stream);\n"
            "  } catch (const std::exception&) {\n  }\n  return 0;\n}\n"
        ),
    ),
    UsageExample(
        name="MultiPartInputFile",
        headers=("cstdint", "cstring", "exception", "stdexcept", "ImfIO.h",
                  "ImfMultiPartInputFile.h", "ImfNamespace.h"),
        snippet=(
            "#include <cstdint>\n#include <cstring>\n#include <exception>\n"
            "#include <stdexcept>\n\n#include <ImfIO.h>\n#include <ImfMultiPartInputFile.h>\n"
            "#include <ImfNamespace.h>\n\n" + _MEMORY_ISTREAM +
            'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n'
            "  MemoryIStream stream(reinterpret_cast<const char*>(data), "
            "static_cast<int64_t>(size));\n"
            "  try {\n    OPENEXR_IMF_NAMESPACE::MultiPartInputFile file(stream);\n"
            "  } catch (const std::exception&) {\n  }\n  return 0;\n}\n"
        ),
    ),
)
_CONDITIONS = ("solo", "shared", "directed")


def _kb_and_examples(
    example: UsageExample, condition: str
) -> tuple[ApiKnowledgeBase, tuple[UsageExample, ...]]:
    # "directed" also shows every example -- it differs from "shared" only in the
    # instruction _prompt appends below, isolating whether an explicit per-iteration
    # reference (never present in "shared") is what "shared" was actually missing.
    examples = (example,) if condition == "solo" else _EXAMPLES
    headers = tuple(sorted({header for ex in examples for header in ex.headers}))
    kb = ApiKnowledgeBase(header_index=headers, namespace_note=_NAMESPACE_NOTE, declarations=())
    return kb, examples


def _prompt(example: UsageExample, condition: str) -> str:
    kb, examples = _kb_and_examples(example, condition)
    prompt = _BASE_PROMPT + "\n\n" + render_prompt_text(kb, examples=examples)
    if condition == "directed":
        prompt += (
            f"\n\nFor this specific task, use the '{example.name}' example above as your "
            "reference implementation: follow its structure and its API calls, not the "
            "other examples'."
        )
    return prompt


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> None:
    """Run the diversity sweep and write both the audit ledger and a results file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(_CONDITIONS), choices=_CONDITIONS)
    args = parser.parse_args()

    reference = load_reference_harness(_REFERENCE_PATH)
    ledger_path = Path(args.ledger)
    run_id = f"kb-diversity-gpu{args.gpu}-{int(time.time())}"

    results: list[dict[str, object]] = []
    iteration = 0
    for example in _EXAMPLES:
        for condition in args.conditions:
            prompt = _prompt(example, condition)
            kb_sha256 = sha256_bytes(prompt.encode())
            _, visible_examples = _kb_and_examples(example, condition)
            for offset in range(args.seed_count):
                seed = 3_000_000 + args.seed_start + offset
                with tempfile.TemporaryDirectory() as tmp:
                    payload = run_dream_generate(
                        prompt,
                        seed,
                        Path(tmp) / "dream.json",
                        gpu=args.gpu,
                        # The IStream-based examples' body alone is ~250+ tokens; the
                        # 256/128 defaults (sized for the one-line checkOpenEXRFile
                        # call) truncated every InputFile/MultiPartInputFile
                        # generation mid-signature -- a real bug this run first
                        # surfaced, not a finding about model capability.
                        new_tokens=768,
                        steps=384,
                    )
                replay(payload)  # raises TrajectoryRecorderError if bookkeeping disagrees
                code, _ = decode_code_block(payload)
                compile_result = compile_harness(code)
                self_similarity = _similarity(code, example.snippet)
                other_similarity = {
                    other.name: _similarity(code, other.snippet)
                    for other in _EXAMPLES
                    if other.name != example.name
                }
                reference_similarity = _similarity(code, reference.source)
                nearest_example = max(
                    (example.name, *other_similarity),
                    key=lambda name: self_similarity if name == example.name
                    else other_similarity[name],
                )
                followed_instruction = (
                    nearest_example == example.name if condition == "directed" else None
                )

                append(
                    ledger_path,
                    HarnessSynthesisLedgerRecord(
                        run_id=run_id,
                        iteration=iteration,
                        arm=f"{example.name}-{condition}",
                        seed=seed,
                        harness_source_sha256=sha256_bytes(code.encode()),
                        knowledge_base_sha256=kb_sha256,
                        usage_example_names=tuple(ex.name for ex in visible_examples),
                        compiled=compile_result.compiled,
                        compile_error=compile_result.error,
                        coverage_edges=None,
                        recorded_at_unix_seconds=time.time(),
                    ),
                )
                results.append(
                    {
                        "example": example.name,
                        "condition": condition,
                        "seed": seed,
                        "compiled": compile_result.compiled,
                        "compile_error": compile_result.error,
                        "self_similarity": self_similarity,
                        "other_similarity": other_similarity,
                        "reference_similarity": reference_similarity,
                        "nearest_example": nearest_example,
                        "followed_instruction": followed_instruction,
                        "code": code,
                    }
                )
                iteration += 1
                followed_str = (
                    "" if followed_instruction is None else f" followed={followed_instruction!s}"
                )
                print(
                    f"{example.name:18} {condition:8} seed={seed} "
                    f"compiled={compile_result.compiled!s:5} self_sim={self_similarity:.2f} "
                    f"ref_sim={reference_similarity:.2f} nearest={nearest_example}{followed_str}"
                )
                Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
