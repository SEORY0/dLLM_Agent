"""Integration tests for deterministic native probe orchestration."""

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.fuzzing.coverage.runner import CoverageRequest, CoverageResult, CoverageTotals
from dllm_agent.fuzzing.runner import (
    BuildStatus,
    FuzzArtifact,
    FuzzRunRequest,
    FuzzRunResult,
    FuzzStatus,
)
from dllm_agent.hunt.models import (
    CandidateBank,
    DenoisingStep,
    GenerationEvidence,
    HarnessCandidate,
    LiveModelProvenance,
)
from dllm_agent.native.models import (
    NativeHuntConfig,
    NativeHuntReport,
    ProbeEvent,
    ProbeStage,
    SelectionEvent,
    StageEvent,
)
from dllm_agent.native.pipeline import run_native_hunt
from dllm_agent.targets.models import HuntManifest

MODEL_ID = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
IMAGE_ID = "sha256:" + ("a" * 64)
REPOSITORY_URL = "https://github.com/example/parser.git"
SECURITY = b"report privately\n"


class _LedgerRecord(BaseModel):
    """Strict test-side projection of one persisted controller event."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["probe", "stage", "selection"]
    event: ProbeEvent | StageEvent | SelectionEvent


def test_native_hunt_when_candidates_compete_uses_equal_isolated_probes_and_proof(
    tmp_path: Path,
) -> None:
    # Given: three candidates, where candidate-01 has the best trajectory but weak coverage.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)
    fuzz_requests: list[FuzzRunRequest] = []
    coverage_requests: list[CoverageRequest] = []

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        fuzz_requests.append(request)
        candidate_id = _candidate_id(request.harness_bytes)
        grown = f"grown-{candidate_id}-{request.fuzzer_seed}".encode()
        return _fuzz_result(candidate_id, request.fuzzer_seed, grown=grown)

    def coverage(request: CoverageRequest) -> CoverageResult:
        coverage_requests.append(request)
        candidate_id = _candidate_id(request.harness_bytes)
        branches = {"candidate-01": 10.0, "candidate-02": 90.0, "candidate-03": 20.0}[candidate_id]
        return _coverage_result(candidate_id, branches)

    # When: the controller runs two tournament stages and a separate proof.
    report = run_native_hunt(
        manifest_path,
        bank_path,
        controller_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
        artifact_dir=tmp_path / "evidence",
    )

    # Then: trajectory orders the first schedule, every candidate gets both fixed seeds,
    # dynamic coverage promotes candidate-02, and dropped candidate-01 gets no later probe.
    observed = tuple(
        (_candidate_id(request.harness_bytes), request.fuzzer_seed, request.duration_seconds)
        for request in fuzz_requests
    )
    assert observed == (
        ("candidate-01", 5, 2),
        ("candidate-01", 7, 2),
        ("candidate-02", 5, 2),
        ("candidate-02", 7, 2),
        ("candidate-03", 5, 2),
        ("candidate-03", 7, 2),
        ("candidate-02", 11, 3),
        ("candidate-02", 13, 3),
        ("candidate-03", 11, 3),
        ("candidate-03", 13, 3),
        ("candidate-02", 0, 9),
    )
    assert report.controller.stage_events[0].promoted_candidate_ids == (
        "candidate-02",
        "candidate-03",
    )
    assert report.controller.selection.selected_candidate_id == "candidate-02"
    assert report.proof.selected_candidate_id == "candidate-02"
    assert all(request.corpus == (b"A", b"B") for request in fuzz_requests[:-1])
    assert tuple(request.corpus for request in coverage_requests[:-1]) == tuple(
        (f"grown-{candidate_id}-{seed}".encode(),)
        for candidate_id, seed, _duration in observed[:-1]
    )
    assert all(request.target_profile == "openexr-v3.4" for request in fuzz_requests)
    assert all(request.target_profile == "openexr-v3.4" for request in coverage_requests)

    first_probe = report.controller.probe_events[0]
    assert (
        first_probe.evidence.build_log_sha256 == hashlib.sha256(b"build-candidate-01-5").hexdigest()
    )
    assert (
        first_probe.evidence.fuzz_log_sha256
        == hashlib.sha256(b"#10 DONE cov: 12 ft: 34 corp: 2/2b exec/s: 56\n").hexdigest()
    )
    assert (
        first_probe.evidence.build_argv_sha256
        == hashlib.sha256(b"docker\0build\0candidate-01\0" + b"5").hexdigest()
    )
    assert (
        first_probe.evidence.fuzz_argv_sha256
        == hashlib.sha256(b"docker\0fuzz\0candidate-01\0" + b"5").hexdigest()
    )
    assert first_probe.evidence.coverage is not None
    assert (
        first_probe.evidence.coverage.report_log_sha256
        == hashlib.sha256(b"coverage-candidate-01").hexdigest()
    )
    assert (
        first_probe.evidence.coverage.commands.run_argv_sha256
        == hashlib.sha256(b"docker\0coverage\0candidate-01").hexdigest()
    )
    assert (tmp_path / "evidence/probes/stage-00/candidate-01/seed-0000000005/fuzz.log").is_file()
    assert (tmp_path / "evidence/proof/fuzz.log").is_file()
    assert report.proof.fuzz.oracle_status == "not_observed"


def test_native_hunt_when_logical_inputs_repeat_writes_byte_stable_ledger(
    tmp_path: Path,
) -> None:
    # Given: one pinned input set and deterministic adapters.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        candidate_id = _candidate_id(request.harness_bytes)
        return _fuzz_result(candidate_id, request.fuzzer_seed, grown=b"grown")

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 50.0)

    # When: identical logical runs use different evidence roots.
    first = run_native_hunt(
        manifest_path,
        bank_path,
        controller_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
        artifact_dir=tmp_path / "first",
    )
    second = run_native_hunt(
        manifest_path,
        bank_path,
        controller_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
        artifact_dir=tmp_path / "second",
    )

    # Then: exact JSONL bytes and embedded hashes match, with monotonic discriminated events.
    first_bytes = (tmp_path / "first/controller-events.jsonl").read_bytes()
    second_bytes = (tmp_path / "second/controller-events.jsonl").read_bytes()
    records = tuple(_LedgerRecord.model_validate_json(line) for line in first_bytes.splitlines())
    assert first_bytes == second_bytes
    assert first.controller.controller_events_sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first.controller.controller_events_sha256 == second.controller.controller_events_sha256
    assert tuple(record.kind for record in records) == (
        *("probe" for _index in range(6)),
        "stage",
        *("probe" for _index in range(4)),
        "stage",
        "selection",
    )
    assert tuple(record.event.sequence for record in records) == tuple(range(1, len(records) + 1))


def test_native_hunt_when_every_build_fails_stops_without_proof_or_ledger(
    tmp_path: Path,
) -> None:
    # Given: every candidate fails to build in the first stage.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        candidate_id = _candidate_id(request.harness_bytes)
        return _fuzz_result(
            candidate_id,
            request.fuzzer_seed,
            grown=b"",
            build_status="failed",
        )

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 0.0)

    # When / Then: the controller fails closed after equal probes and writes no final ledger.
    artifact_dir = tmp_path / "evidence"
    with pytest.raises(HuntSetupError, match="zero successful builds"):
        _ = run_native_hunt(
            manifest_path,
            bank_path,
            controller_path,
            source,
            fuzz_runner=fuzz,
            coverage_runner=coverage,
            artifact_dir=artifact_dir,
        )
    assert not (artifact_dir / "proof").exists()
    assert not (artifact_dir / "controller-events.jsonl").exists()


def test_native_hunt_when_final_proof_build_fails_does_not_publish_ledger(
    tmp_path: Path,
) -> None:
    # Given: every probe succeeds, but the separate proof build fails and fuzzing is skipped.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        candidate_id = _candidate_id(request.harness_bytes)
        if request.duration_seconds == 9:
            return _fuzz_result(
                candidate_id,
                request.fuzzer_seed,
                grown=b"",
                build_status="failed",
            )
        return _fuzz_result(candidate_id, request.fuzzer_seed, grown=b"probe-grown")

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 50.0)

    # When / Then: proof raw evidence may remain, but no successful controller ledger is published.
    artifact_dir = tmp_path / "evidence"
    with pytest.raises(HuntSetupError, match="proof execution"):
        _ = run_native_hunt(
            manifest_path,
            bank_path,
            controller_path,
            source,
            fuzz_runner=fuzz,
            coverage_runner=coverage,
            artifact_dir=artifact_dir,
        )
    assert (artifact_dir / "proof/build.log").is_file()
    assert not (artifact_dir / "controller-events.jsonl").exists()


def test_native_hunt_when_final_proof_crashes_accepts_report_and_uses_proof_oracle(
    tmp_path: Path,
) -> None:
    # Given: probes observe sanitizers, while the successful-build proof crashes without one.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        candidate_id = _candidate_id(request.harness_bytes)
        result = _fuzz_result(candidate_id, request.fuzzer_seed, grown=b"grown")
        if request.duration_seconds == 9:
            return replace(
                result,
                fuzz_status="crashed",
                fuzz_log=b"libFuzzer deadly signal\n",
                crash_artifacts=(FuzzArtifact("crash-proof", b"proof"),),
            )
        return replace(
            result,
            fuzz_status="crashed",
            fuzz_log=b"ERROR: AddressSanitizer\n",
            crash_artifacts=(FuzzArtifact("crash-probe", b"probe"),),
        )

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 50.0)

    # When: a proof build succeeds and its fuzzer terminates with a crash.
    artifact_dir = tmp_path / "evidence"
    report = run_native_hunt(
        manifest_path,
        bank_path,
        controller_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
        artifact_dir=artifact_dir,
    )

    # Then: the crash is accepted, ledger is published, and only proof evidence sets the oracle.
    assert isinstance(report, NativeHuntReport)
    assert all(event.sanitizer_observed for event in report.controller.probe_events)
    assert report.proof.fuzz.fuzz_status == "crashed"
    assert report.proof.fuzz.oracle_status == "crash_without_sanitizer"
    assert (artifact_dir / "controller-events.jsonl").is_file()


def test_native_hunt_when_evidence_root_exists_refuses_to_overwrite(tmp_path: Path) -> None:
    # Given: an existing evidence root containing user data.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)
    artifact_dir = tmp_path / "evidence"
    artifact_dir.mkdir()
    marker = artifact_dir / "keep.txt"
    _ = marker.write_bytes(b"keep")

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        return _fuzz_result(_candidate_id(request.harness_bytes), request.fuzzer_seed, grown=b"x")

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 0.0)

    # When / Then: setup fails before a run and the existing bytes remain untouched.
    with pytest.raises(HuntSetupError):
        _ = run_native_hunt(
            manifest_path,
            bank_path,
            controller_path,
            source,
            fuzz_runner=fuzz,
            coverage_runner=coverage,
            artifact_dir=artifact_dir,
        )
    assert marker.read_bytes() == b"keep"


def test_native_hunt_when_candidate_id_is_a_path_fails_before_artifacts(tmp_path: Path) -> None:
    # Given: an otherwise valid candidate bank with an ID that would create nested evidence paths.
    manifest_path, bank_path, controller_path, source = _inputs(tmp_path)
    bank = CandidateBank.model_validate_json(bank_path.read_bytes())
    unsafe = bank.candidates[0].model_copy(update={"candidate_id": "nested/candidate-01"})
    _ = bank_path.write_text(
        bank.model_copy(update={"candidates": (unsafe, *bank.candidates[1:])}).model_dump_json(),
        encoding="utf-8",
    )

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        return _fuzz_result(_candidate_id(request.harness_bytes), request.fuzzer_seed, grown=b"x")

    def coverage(request: CoverageRequest) -> CoverageResult:
        return _coverage_result(_candidate_id(request.harness_bytes), 0.0)

    # When / Then: strict parsing rejects the path-like identifier before evidence exists.
    artifact_dir = tmp_path / "evidence"
    with pytest.raises(HuntSetupError, match="candidate id"):
        _ = run_native_hunt(
            manifest_path,
            bank_path,
            controller_path,
            source,
            fuzz_runner=fuzz,
            coverage_runner=coverage,
            artifact_dir=artifact_dir,
        )
    assert not artifact_dir.exists()


def _inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    source, commit = _git_target(root)
    prompt = root / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a parser harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest_path = root / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(_manifest(commit).model_dump_json(), encoding="utf-8")
    bank_path = root / "bank.json"
    _ = bank_path.write_text(_bank(prompt_bytes).model_dump_json(), encoding="utf-8")
    controller_path = root / "controller.json"
    config = NativeHuntConfig(
        schema_version="dllm-native-hunt-config/v1",
        stages=(
            ProbeStage(duration_seconds=2, fuzzer_seeds=(5, 7), survivors=2),
            ProbeStage(duration_seconds=3, fuzzer_seeds=(11, 13), survivors=1),
        ),
        proof_duration_seconds=9,
    )
    _ = controller_path.write_text(config.model_dump_json(), encoding="utf-8")
    return manifest_path, bank_path, controller_path, source


def _git_target(root: Path) -> tuple[Path, str]:
    source = root / "target"
    source.mkdir()
    _ = (source / "SECURITY.md").write_bytes(SECURITY)
    for command in (
        ("/usr/bin/git", "init", "-q"),
        ("/usr/bin/git", "config", "user.email", "test@example.invalid"),
        ("/usr/bin/git", "config", "user.name", "Test"),
        ("/usr/bin/git", "remote", "add", "origin", REPOSITORY_URL),
        ("/usr/bin/git", "add", "."),
        ("/usr/bin/git", "commit", "-qm", "target"),
    ):
        _ = subprocess.run(command, cwd=source, check=True, capture_output=True)  # noqa: S603
    result = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return source, result.stdout.strip()


def _manifest(commit: str) -> HuntManifest:
    return HuntManifest(
        schema_version="dllm-hunt/v1",
        hunt_id="native-test",
        target_id="parser",
        repository_url=REPOSITORY_URL,
        commit_sha=commit,
        security_policy_url="https://github.com/example/parser/security/advisories/new",
        security_policy_sha256=hashlib.sha256(SECURITY).hexdigest(),
        prompt="prompts/prompt.md",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        image_id=IMAGE_ID,
        target_profile="openexr-v3.4",
        sanitizers="address",
        allowed_includes=("cstddef",),
        required_symbols=("LLVMFuzzerTestOneInput", "candidate_marker"),
        seed_hex=("41", "42"),
        fuzz_seconds=1,
        max_len=64,
        rss_limit_mb=256,
        timeout_seconds=5,
    )


def _bank(prompt: bytes) -> CandidateBank:
    def candidate(candidate_id: str, confidence: float) -> HarnessCandidate:
        source = (
            "#include <cstddef>\n"
            f'constexpr const char* candidate_marker = "{candidate_id}";\n'
            'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, std::size_t) '
            "{ return candidate_marker[0] == 'x'; }\n"
        )
        return HarnessCandidate(
            candidate_id=candidate_id,
            source=source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            seed=1,
            nfe=2,
            steps=(
                DenoisingStep(
                    step_index=1,
                    masked_tokens_before=4,
                    committed_tokens=4,
                    mean_confidence=confidence,
                    state_sha256=hashlib.sha256(candidate_id.encode()).hexdigest(),
                ),
            ),
        )

    return CandidateBank(
        schema_version="dllm-candidate-bank/v1",
        prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        model=LiveModelProvenance(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            remote_code_sha256="f" * 64,
            python_version="3.12.13",
            transformers_version="5.3.0",
            torch_version="2.5.1+cu121",
            device="cuda:0",
            generation=GenerationEvidence(
                steps=64,
                gen_length=64,
                block_length=4,
                temperature=0.2,
            ),
        ),
        candidates=(
            candidate("candidate-01", 0.9),
            candidate("candidate-02", 0.7),
            candidate("candidate-03", 0.5),
        ),
    )


def _candidate_id(harness: bytes) -> str:
    text = harness.decode()
    return next(
        candidate_id
        for candidate_id in ("candidate-01", "candidate-02", "candidate-03")
        if candidate_id in text
    )


def _fuzz_result(
    candidate_id: str,
    seed: int,
    *,
    grown: bytes,
    build_status: BuildStatus = "succeeded",
) -> FuzzRunResult:
    fuzz_status: FuzzStatus = "completed" if build_status == "succeeded" else "skipped"
    argv = ("docker", "fuzz", candidate_id, str(seed))
    return FuzzRunResult(
        compiler_version_summary=b"clang 18",
        fuzzer_version_summary=b"libFuzzer",
        build_status=build_status,
        build_log=f"build-{candidate_id}-{seed}".encode(),
        fuzz_status=fuzz_status,
        fuzz_log=b"#10 DONE cov: 12 ft: 34 corp: 2/2b exec/s: 56\n",
        corpus_artifacts=((FuzzArtifact("grown", grown),) if grown else ()),
        crash_artifacts=(),
        compiler_version_argv=("docker", "clang-version"),
        fuzzer_version_argv=("docker", "fuzzer-version"),
        build_argv=("docker", "build", candidate_id, str(seed)),
        fuzz_argv=argv,
    )


def _coverage_result(candidate_id: str, branches: float) -> CoverageResult:
    argv = ("docker", "coverage", candidate_id)
    return CoverageResult(
        build_status="succeeded",
        run_status="completed",
        merge_status="succeeded",
        report_status="succeeded",
        build_log=f"coverage-build-{candidate_id}".encode(),
        run_log=f"coverage-run-{candidate_id}".encode(),
        merge_log=f"coverage-merge-{candidate_id}".encode(),
        report_log=f"coverage-{candidate_id}".encode(),
        build_argv=argv,
        run_argv=argv,
        merge_argv=argv,
        report_argv=argv,
        coverage=CoverageTotals(regions=50.0, functions=50.0, lines=50.0, branches=branches),
    )
