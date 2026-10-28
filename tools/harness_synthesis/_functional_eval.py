"""Run RQ4b's functional evaluator on harness sources, in the pinned compile-only image.

For each source, one container (shared by the whole batch) does four things.

1. Builds the coverage binary with the same CMake project and flags as
   ``build-harness coverage``. It configures once and rebuilds per source, which is much
   faster than one configure per source.
2. Runs the binary once on the validation input, the pinned C0 seed (a valid single-part
   scanline EXR; ``rq4b_validate_benchmark.py`` checks that both reference bodies
   construct and inspect it without an exception).
3. Exports llvm-cov region segments for ``harness.cc``.
4. Dumps clang's JSON AST of ``LLVMFuzzerTestOneInput``.

The verdict itself is pure: ``dllm_agent.harness_synthesis.functional_eval``. CPU only:
nothing here touches a GPU or a model.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _compile import _ERROR_LINE, _INCLUDE_FLAGS, _SEED_CORPUS_HEX, IMAGE, _ensure_image

from dllm_agent.harness_synthesis.functional_eval import (
    FunctionalVerdict,
    ast_facts,
    evaluate,
)

if TYPE_CHECKING:
    from dllm_agent.harness_synthesis.repair_benchmark import FunctionalSpec

_SCRIPT = r"""
set -u
umask 000
unset CFLAGS CXXFLAGS
mkdir -p /work/cur
cp /work/0/harness.cc /work/cur/harness.cc
cmake -S /opt/dllm/runtime -B /work/build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ -DDLLM_MODE=coverage \
  -DOPENEXR_INSTALL_ROOT=/opt/openexr-coverage \
  -DDLLM_HARNESS_SOURCE=/work/cur/harness.cc -DDLLM_OUTPUT_DIRECTORY=/work/cur \
  > /work/configure.log 2>&1
for i in $(seq 0 __LAST__); do
  d=/work/$i
  cp $d/harness.cc /work/cur/harness.cc
  touch /work/cur/harness.cc
  rm -f /work/cur/coverage_target /work/cur/run.profraw
  cmake --build /work/build --target dllm_harness --parallel 1 > $d/build.log 2>&1
  if [ -x /work/cur/coverage_target ]; then
    (cd /work/cur && LLVM_PROFILE_FILE=/work/cur/run.profraw timeout 60 ./coverage_target \
      -runs=1 /work/valid.exr > $d/run.log 2>&1; echo $? > $d/exit_code)
    if [ -f /work/cur/run.profraw ]; then
      llvm-profdata merge -sparse /work/cur/run.profraw -o /work/cur/run.profdata && \
      llvm-cov export -format=text -skip-functions -instr-profile=/work/cur/run.profdata \
        /work/cur/coverage_target /work/cur/harness.cc > $d/cov.json 2>/dev/null
    fi
    clang++ -std=c++17 -fsyntax-only __INCLUDES__ -Xclang -ast-dump=json \
      -Xclang -ast-dump-filter=LLVMFuzzerTestOneInput $d/harness.cc > $d/ast.json 2>/dev/null
  fi
done
exit 0
"""


@dataclass(frozen=True, slots=True)
class FunctionalRun:
    """Raw artefacts for one source, plus the verdict computed from them."""

    compiled: bool
    compile_error: str | None
    exit_code: int | None
    verdict: FunctionalVerdict
    ast_nodes: list[dict[str, Any]] | None = None
    segments: list[list[Any]] | None = None


def _json_objects(raw: str) -> list[dict[str, Any]]:
    decoder, objects, index = json.JSONDecoder(), [], 0
    while index < len(raw):
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index >= len(raw):
            break
        obj, index = decoder.raw_decode(raw, index)
        objects.append(obj)
    return objects


def _segments(cov_path: Path) -> list[list[Any]]:
    if not cov_path.exists() or not cov_path.read_text().strip():
        return []
    files = json.loads(cov_path.read_text())["data"][0]["files"]
    return next((f["segments"] for f in files if f["filename"].endswith("harness.cc")), [])


def evaluate_many(
    cases: list[tuple[str, int, int, FunctionalSpec]],
    *,
    timeout: float = 3600.0,
    keep_raw: bool = False,
) -> list[FunctionalRun]:
    """Evaluate ``(source, body_start, body_end, spec)`` cases (body offsets into source).

    ``keep_raw`` also returns the parsed AST and coverage segments (for test fixtures).
    """
    _ensure_image()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        work.chmod(0o777)
        (work / "valid.exr").write_bytes(bytes.fromhex(_SEED_CORPUS_HEX.read_text().strip()))
        for index, (source, _, _, _) in enumerate(cases):
            case = work / str(index)
            case.mkdir()
            case.chmod(0o777)
            (case / "harness.cc").write_text(source)
        script = _SCRIPT.replace("__LAST__", str(len(cases) - 1)).replace(
            "__INCLUDES__", " ".join(_INCLUDE_FLAGS)
        )
        uid = f"{os.getuid()}:{os.getgid()}"
        subprocess.run(  # noqa: S603 - fixed argv, shell=False on the host side.
            [  # noqa: S607
                "docker",
                "run",
                "--rm",
                "-u",
                uid,
                "-v",
                f"{work}:/work",
                "--entrypoint",
                "sh",
                IMAGE,
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        runs = []
        for index, (source, body_start, body_end, spec) in enumerate(cases):
            case = work / str(index)
            compiled = (case / "exit_code").exists()
            build_log = (case / "build.log").read_text() if (case / "build.log").exists() else ""
            errors = [
                re.sub(r"^.*?harness\.cc", "harness.cc", line)
                for line in build_log.splitlines()
                if _ERROR_LINE.search(line)
            ]
            exit_code = int((case / "exit_code").read_text().strip()) if compiled else None
            facts = None
            nodes = None
            if compiled and (case / "ast.json").exists():
                nodes = _json_objects((case / "ast.json").read_text())
                facts = ast_facts(
                    nodes,
                    spec,
                    body_start=body_start,
                    body_end=body_end,
                )
            segments = _segments(case / "cov.json")
            verdict = evaluate(
                spec,
                source=source,
                compiled=compiled,
                facts=facts,
                segments=segments,
                exit_code=exit_code,
            )
            runs.append(
                FunctionalRun(
                    compiled=compiled,
                    compile_error=None if compiled else ("\n".join(errors[:3]) or build_log[-500:]),
                    exit_code=exit_code,
                    verdict=verdict,
                    ast_nodes=nodes if keep_raw else None,
                    segments=segments if keep_raw else None,
                )
            )
        return runs
