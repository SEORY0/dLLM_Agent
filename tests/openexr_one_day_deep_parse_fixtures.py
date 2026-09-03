"""Deterministic Docker boundary fixtures for OpenEXR deep-parse tests."""

# pyright: reportUnnecessaryComparison=false, reportUnreachable=false

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess
from typing import Literal, assert_never

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellResult,
    CellRunRequest,
    SearchAccounting,
    SearchBuildProvenance,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRecord,
    CorpusInput,
    CoverageSeries,
)
from dllm_agent.experiments.openexr_one_day.post_search import DeepParseRequest

IMAGE_DIGEST = "sha256:" + "a" * 64
TARGET_COMMIT = "777d231a179de1711d2a942810d9559216ab3f4c"
MANIFEST_PATH = "/opt/dllm/evidence/build-manifest.json"
CHECKER_PATH = "/opt/dllm/bin/openexr-deep-parse-check"
SEARCH_BINARY_PATH = "/opt/dllm/bin/poc_probe-asan"

BUILD_FILES: tuple[tuple[str, bytes], ...] = (
    (SEARCH_BINARY_PATH, b"search-binary"),
    ("/opt/dllm/bin/poc_probe-coverage", b"coverage-binary"),
    (CHECKER_PATH, b"deep-parse-binary"),
    ("/opt/dllm/runtime/deep_parse_check.cc", b"deep-parse-source"),
    ("/opt/dllm/evidence/openexr-asan-compile-commands.json", b"asan-commands"),
    ("/opt/dllm/evidence/openexr-coverage-compile-commands.json", b"coverage-commands"),
    ("/opt/dllm/evidence/runtime-asan-compile-commands.json", b"runtime-commands"),
    ("/opt/dllm/evidence/runtime-coverage-compile-commands.json", b"runtime-cov-commands"),
)
CHECKER_SHA256 = sha256_bytes(dict(BUILD_FILES)[CHECKER_PATH])


def build_manifest_bytes(
    files: tuple[tuple[str, bytes], ...] = BUILD_FILES,
    *,
    checker_sha256: str | None = None,
) -> bytes:
    """Return a hand-derived canonical runtime build manifest."""
    return canonical_json_bytes(
        {
            "coverage_instrumentation": (
                "fuzzer-no-link",
                "profile-instr-generate",
                "coverage-mapping",
            ),
            "deep_parse_checker_sha256": (
                sha256_bytes(dict(files)[CHECKER_PATH])
                if checker_sha256 is None
                else checker_sha256
            ),
            "files": tuple(
                {"path": path, "sha256": sha256_bytes(content)} for path, content in files
            ),
            "schema_version": "dllm-openexr-runtime-build/v1",
            "search_sanitizers": ("address", "undefined"),
            "target_commit": TARGET_COMMIT,
            "undefined_recovery": False,
        }
    )


def deep_parse_request(
    root: Path,
    *,
    inputs: tuple[bytes, ...] = (b"accepted", b"rejected"),
    manifest: bytes | None = None,
    checker_binary_sha256: str = CHECKER_SHA256,
) -> DeepParseRequest:
    """Create a request whose inputs are exact files below one cell root."""
    corpus = root / "corpus-inputs"
    corpus.mkdir(parents=True)
    paths: list[Path] = []
    hashes: list[str] = []
    for index, content in enumerate(inputs):
        path = corpus / f"input-{index:04d}.exr"
        _ = path.write_bytes(content)
        paths.append(path)
        hashes.append(sha256_bytes(content))
    manifest_bytes = build_manifest_bytes() if manifest is None else manifest
    return DeepParseRequest(
        cell_id="wave-00-rep-00-selective-revision",
        artifact_root=root,
        cell_request_sha256="1" * 64,
        cell_result_sha256="2" * 64,
        target_commit=TARGET_COMMIT,
        image_digest=IMAGE_DIGEST,
        checker_binary_sha256=checker_binary_sha256,
        build_manifest_sha256=sha256_bytes(manifest_bytes),
        final_corpus_set_sha256="3" * 64,
        input_paths=tuple(paths),
        input_sha256=tuple(hashes),
    )


def deep_parse_cell(root: Path) -> tuple[CellRunRequest, CellResult]:
    """Create the cell request/result binding used by post-search reuse tests."""
    dictionary = root / "openexr.dict"
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    _ = dictionary.write_bytes(b'"EXR"')
    request = CellRunRequest(
        cell_id="wave-00-rep-00-selective-revision",
        artifact_root=root,
        corpus_root=root / "corpus",
        crash_root=root / "crashes",
        log_root=root / "logs",
        target_commit=TARGET_COMMIT,
        harness_sha256="1" * 64,
        config_sha256="2" * 64,
        bank_sha256="3" * 64,
        start_monotonic_seconds=0.0,
        deadline_monotonic_seconds=10_800.0,
        fuzzer_seed=100,
        max_input_length=1_048_576,
        target_timeout_seconds=10,
        rss_limit_mb=2_048,
        cpu_index=8,
        dictionary_path=dictionary,
        dictionary_sha256=sha256_bytes(b'"EXR"'),
        build=SearchBuildProvenance(
            image_digest=IMAGE_DIGEST,
            binary_sha256=sha256_bytes(dict(BUILD_FILES)[SEARCH_BINARY_PATH]),
            sanitizers=("address", "undefined"),
        ),
    )
    result = CellResult(
        cell_id=request.cell_id,
        status="completed",
        started_monotonic_seconds=0.0,
        finished_monotonic_seconds=10_800.0,
        coverage_freeze_elapsed_seconds=None,
        attempt_count=1,
        manifest_sha256="4" * 64,
        event_stream_sha256="5" * 64,
        corpus_index_stream_sha256="6" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(range(0, 10_801, 900)),
        search_accounting=SearchAccounting(cpu_seconds=1.0, wall_seconds=10_800.0),
    )
    return request, result


def private_cell_root(namespace: Path) -> Path:
    """Map one pytest namespace to a private absolute campaign root."""
    root = Path("/data/dllm-agent") / f"task7-{sha256_bytes(str(namespace).encode())[:16]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def final_coverage(root: Path, content: bytes, *, name: str = "input.exr") -> CoverageSeries:
    """Create one complete final-corpus coverage row with exact backing bytes."""
    digest = sha256_bytes(content)
    stored_path = Path("corpus-inputs") / name
    path = root / stored_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    row = CorpusIndexRecord(
        scheduled_elapsed_seconds=10_800,
        observation_kind="observed",
        source_elapsed_seconds=10_800.0,
        observed_lateness_seconds=0.0,
        new_inputs=(
            CorpusInput(sha256=digest, size_bytes=len(content), stored_path=stored_path.as_posix()),
        ),
        cumulative_hashes=(digest,),
        cumulative_count=1,
        cumulative_bytes=len(content),
        cumulative_set_sha256=sha256_bytes(canonical_json_bytes((digest,))),
    )
    return CoverageSeries(corpus_index=(row,), points=())


@dataclass(frozen=True, slots=True)
class ScriptedDockerRunner:
    """In-memory fake for the slow external Docker process boundary."""

    manifest: bytes = field(default_factory=build_manifest_bytes)
    files: tuple[tuple[str, bytes], ...] = BUILD_FILES
    parse_codes: tuple[int, ...] = (0, 1)
    parse_stdout: tuple[bytes, ...] = (b"accepted\n", b"")
    parse_stderr: tuple[bytes, ...] = (b"", b"parser rejected\n")
    failure: Literal["none", "timeout", "docker"] = "none"
    calls: list[tuple[str, ...]] = field(default_factory=list)
    parse_index: int = 0

    def __call__(self, argv: tuple[str, ...], timeout: float) -> CompletedProcess[bytes]:
        """Return exact scripted process results for inspect, provenance, and replay."""
        del timeout
        self.calls.append(argv)
        if argv[:3] == ("docker", "image", "inspect"):
            return CompletedProcess(argv, 0, f"{IMAGE_DIGEST}\n".encode(), b"")
        if argv[-2:] == ("/bin/cat", MANIFEST_PATH):
            return CompletedProcess(argv, 0, self.manifest, b"")
        if "/usr/bin/sha256sum" in argv:
            content_by_path = dict(self.files)
            requested = argv[argv.index("/usr/bin/sha256sum") + 1 :]
            output = b"".join(
                f"{sha256_bytes(content_by_path[path])}  {path}\n".encode() for path in requested
            )
            return CompletedProcess(argv, 0, output, b"")
        if CHECKER_PATH not in argv:
            raise AssertionError(argv)
        match self.failure:
            case "timeout":
                raise subprocess.TimeoutExpired(argv, 5.0, output=b"partial", stderr=b"late")
            case "docker":
                reason = "docker unavailable"
                raise OSError(reason)
            case "none":
                index = self.parse_index
                object.__setattr__(self, "parse_index", index + 1)
                return CompletedProcess(
                    argv,
                    self.parse_codes[index],
                    self.parse_stdout[index],
                    self.parse_stderr[index],
                )
            case _ as unreachable:
                assert_never(unreachable)
