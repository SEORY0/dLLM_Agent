"""Live dLLM to sandboxed libFuzzer pipeline integration tests."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.fuzzing.coverage.runner import (
    CoverageRequest,
    CoverageResult,
    CoverageTotals,
)
from dllm_agent.fuzzing.runner import FuzzArtifact, FuzzRunRequest, FuzzRunResult
from dllm_agent.hunt.models import (
    CandidateBank,
    DenoisingStep,
    GenerationEvidence,
    HarnessCandidate,
    LiveModelProvenance,
)
from dllm_agent.hunt.pipeline import run_hunt
from dllm_agent.targets.models import HuntManifest

MODEL_ID = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
IMAGE_ID = "sha256:" + ("a" * 64)
REPOSITORY_URL = "https://github.com/example/parser.git"
SECURITY = b"report privately\n"
HARNESS = (
    "#include <cstddef>\n"
    '#include <jsoncons_ext/ubjson/ubjson.hpp>\nextern "C" int '
    "LLVMFuzzerTestOneInput(const unsigned char*, std::size_t) { "
    "try_decode_ubjson; ubjson_bytes_cursor; std::error_code* p = nullptr; "
    "return p == nullptr ? 0 : 1; }\n"
)


def test_live_hunt_selects_ranked_candidate_and_hashes_dynamic_evidence(tmp_path: Path) -> None:
    # Given
    source, commit = _git_target(tmp_path)
    prompt = tmp_path / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a UBJSON harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest = _manifest(commit)
    manifest_path = tmp_path / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    bank = _candidate_bank(prompt_bytes)
    bank_path = tmp_path / "bank.json"
    bank_bytes = bank.model_dump_json().encode()
    _ = bank_path.write_bytes(bank_bytes)
    requests: list[FuzzRunRequest] = []

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        requests.append(request)
        staged_header = request.source_root / "include/header.hpp"
        assert request.source_root != source.resolve()
        assert staged_header.read_bytes() == b"// pinned target\n"
        assert staged_header.stat().st_mode & 0o004 == 0o004
        assert request.source_root.stat().st_mode & 0o001 == 0o001
        assert request.sanitizers == "address,undefined"
        argv = ("docker", "run", request.image_id)
        return FuzzRunResult(
            compiler_version_summary=b"clang version 18.1.8\n",
            fuzzer_version_summary=b"libFuzzer\n",
            build_status="succeeded",
            build_log=b"",
            fuzz_status="completed",
            fuzz_log=(
                b"#20 DONE cov: 42 ft: 84 corp: 8/40b exec/s: 100\n"
                b"stat::number_of_executed_units: 25\n"
                b"stat::average_exec_per_sec: 101\n"
            ),
            corpus_artifacts=(FuzzArtifact("seed-0000", b"Z"),),
            crash_artifacts=(),
            compiler_version_argv=argv,
            fuzzer_version_argv=argv,
            build_argv=argv,
            fuzz_argv=argv,
        )

    def coverage(request: CoverageRequest) -> CoverageResult:
        assert request.corpus == (b"Z",)
        assert request.target_profile == "header-only"
        assert (request.source_root / "include/header.hpp").read_bytes() == b"// pinned target\n"
        argv = ("docker", "run", request.image_id)
        return CoverageResult(
            build_status="succeeded",
            run_status="completed",
            merge_status="succeeded",
            report_status="succeeded",
            build_log=b"",
            run_log=b"replayed\n",
            merge_log=b"",
            report_log=b"TOTAL 10 2 80.00% 4 1 75.00% 20 5 75.00% 8 6 25.00%\n",
            build_argv=argv,
            run_argv=argv,
            merge_argv=argv,
            report_argv=argv,
            coverage=CoverageTotals(regions=80.0, functions=75.0, lines=75.0, branches=25.0),
        )

    # When
    report = run_hunt(
        manifest_path,
        bank_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
    )

    # Then
    assert report.schema_version == "dllm-hunt-report/v1"
    assert report.selected_candidate_id == "candidate-02"
    assert report.target.commit_sha == commit
    assert report.candidate_bank_sha256 == hashlib.sha256(bank_bytes).hexdigest()
    assert report.fuzz.oracle_status == "not_observed"
    assert report.fuzz.executed_units == 25
    assert report.fuzz.executions_per_second == 101
    assert report.fuzz.corpus_artifacts[0].sha256 == hashlib.sha256(b"Z").hexdigest()
    assert report.coverage.totals is not None
    assert report.coverage.totals.lines == 75.0
    assert requests[0].image_id == IMAGE_ID
    assert requests[0].harness_bytes == HARNESS.encode()
    assert requests[0].fuzzer_seed == 0
    assert requests[0].duration_seconds == manifest.fuzz_seconds
    assert requests[0].target_profile == "header-only"


def test_live_hunt_propagates_openexr_profile_to_fuzz_and_coverage(tmp_path: Path) -> None:
    # Given
    source, commit = _git_target(tmp_path)
    prompt = tmp_path / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a UBJSON harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest = _manifest(commit).model_copy(
        update={"target_profile": "openexr-v3.4", "sanitizers": "address"}
    )
    manifest_path = tmp_path / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    bank_path = tmp_path / "bank.json"
    _ = bank_path.write_text(_candidate_bank(prompt_bytes).model_dump_json(), encoding="utf-8")
    fuzz_requests: list[FuzzRunRequest] = []
    coverage_requests: list[CoverageRequest] = []

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        fuzz_requests.append(request)
        argv = ("docker", "run", request.image_id)
        return FuzzRunResult(
            compiler_version_summary=b"clang",
            fuzzer_version_summary=b"libFuzzer",
            build_status="succeeded",
            build_log=b"",
            fuzz_status="completed",
            fuzz_log=b"",
            corpus_artifacts=(),
            crash_artifacts=(),
            compiler_version_argv=argv,
            fuzzer_version_argv=argv,
            build_argv=argv,
            fuzz_argv=argv,
            sanitizers=request.sanitizers,
        )

    def coverage(request: CoverageRequest) -> CoverageResult:
        coverage_requests.append(request)
        argv = ("docker", "run", request.image_id)
        return CoverageResult(
            build_status="succeeded",
            run_status="completed",
            merge_status="succeeded",
            report_status="succeeded",
            build_log=b"",
            run_log=b"",
            merge_log=b"",
            report_log=b"",
            build_argv=argv,
            run_argv=argv,
            merge_argv=argv,
            report_argv=argv,
            coverage=None,
        )

    # When
    _ = run_hunt(
        manifest_path,
        bank_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
    )

    # Then
    assert tuple(request.target_profile for request in fuzz_requests) == ("openexr-v3.4",)
    assert tuple(request.target_profile for request in coverage_requests) == ("openexr-v3.4",)
    assert fuzz_requests[0].sanitizers == "address"


def test_hunt_manifest_accepts_only_closed_target_profiles() -> None:
    # Given
    valid_json = (
        _manifest("a" * 40)
        .model_dump_json()
        .replace(
            '"target_profile":"header-only"',
            '"target_profile":"openexr-v3.4"',
        )
        .replace('"sanitizers":"address,undefined"', '"sanitizers":"address"')
    )
    invalid_json = valid_json.replace("openexr-v3.4", "manifest-shell-command")

    # When
    manifest = HuntManifest.model_validate_json(valid_json)

    # Then
    assert manifest.target_profile == "openexr-v3.4"
    with pytest.raises(ValueError, match="target_profile"):
        _ = HuntManifest.model_validate_json(invalid_json)


def test_hunt_manifest_rejects_openexr_with_default_sanitizer() -> None:
    # Given
    invalid_json = (
        _manifest("a" * 40)
        .model_dump_json()
        .replace(
            '"target_profile":"header-only"',
            '"target_profile":"openexr-v3.4"',
        )
    )

    # When / Then
    with pytest.raises(ValueError, match="address"):
        _ = HuntManifest.model_validate_json(invalid_json)


def test_live_hunt_when_valid_selection_and_duration_are_overridden_uses_both(
    tmp_path: Path,
) -> None:
    # Given: a valid lower-ranked candidate and a distinct positive proof duration.
    source, commit = _git_target(tmp_path)
    prompt = tmp_path / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a UBJSON harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest_path = tmp_path / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(_manifest(commit).model_dump_json(), encoding="utf-8")
    bank_path = tmp_path / "bank.json"
    _ = bank_path.write_text(_candidate_bank(prompt_bytes).model_dump_json(), encoding="utf-8")
    requests: list[FuzzRunRequest] = []

    def fuzz(request: FuzzRunRequest) -> FuzzRunResult:
        requests.append(request)
        argv = ("docker", "run", request.image_id)
        return FuzzRunResult(
            compiler_version_summary=b"clang",
            fuzzer_version_summary=b"libFuzzer",
            build_status="succeeded",
            build_log=b"",
            fuzz_status="completed",
            fuzz_log=b"",
            corpus_artifacts=(),
            crash_artifacts=(),
            compiler_version_argv=argv,
            fuzzer_version_argv=argv,
            build_argv=argv,
            fuzz_argv=argv,
        )

    def coverage(request: CoverageRequest) -> CoverageResult:
        argv = ("docker", "run", request.image_id)
        return CoverageResult(
            build_status="succeeded",
            run_status="completed",
            merge_status="succeeded",
            report_status="succeeded",
            build_log=b"",
            run_log=b"",
            merge_log=b"",
            report_log=b"",
            build_argv=argv,
            run_argv=argv,
            merge_argv=argv,
            report_argv=argv,
            coverage=None,
        )

    # When: proof selection and budget are explicitly overridden.
    report = run_hunt(
        manifest_path,
        bank_path,
        source,
        fuzz_runner=fuzz,
        coverage_runner=coverage,
        selected_candidate_id="candidate-01",
        fuzz_seconds=7,
    )

    # Then: the statically valid requested candidate alone receives the requested budget.
    assert report.selected_candidate_id == "candidate-01"
    assert requests[0].duration_seconds == 7


@pytest.mark.parametrize("selected_candidate_id", ["missing", "candidate-rejected"])
def test_live_hunt_when_selection_override_is_not_statically_valid_fails_closed(
    tmp_path: Path, selected_candidate_id: str
) -> None:
    # Given: a valid pinned hunt whose bank has no valid candidate with the requested ID.
    source, commit = _git_target(tmp_path)
    prompt = tmp_path / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a UBJSON harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest_path = tmp_path / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(_manifest(commit).model_dump_json(), encoding="utf-8")
    bank_path = tmp_path / "bank.json"
    bank = _candidate_bank(prompt_bytes)
    if selected_candidate_id == "candidate-rejected":
        rejected_source = "#include <vector>\n" + bank.candidates[0].source
        rejected = bank.candidates[0].model_copy(
            update={
                "candidate_id": selected_candidate_id,
                "source": rejected_source,
                "source_sha256": hashlib.sha256(rejected_source.encode()).hexdigest(),
            }
        )
        bank = bank.model_copy(update={"candidates": (rejected, bank.candidates[1])})
    _ = bank_path.write_text(bank.model_dump_json(), encoding="utf-8")

    # When / Then: selection fails before dynamic execution.
    with pytest.raises(HuntSetupError, match="selected candidate"):
        _ = run_hunt(
            manifest_path,
            bank_path,
            source,
            selected_candidate_id=selected_candidate_id,
        )


def test_live_hunt_when_duration_override_is_not_positive_fails_closed(tmp_path: Path) -> None:
    # Given: a valid pinned hunt and a zero proof duration.
    source, commit = _git_target(tmp_path)
    prompt = tmp_path / "prompts/prompt.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a UBJSON harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest_path = tmp_path / "configs/hunt.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(_manifest(commit).model_dump_json(), encoding="utf-8")
    bank_path = tmp_path / "bank.json"
    _ = bank_path.write_text(_candidate_bank(prompt_bytes).model_dump_json(), encoding="utf-8")

    # When / Then: invalid proof budget fails before dynamic execution.
    with pytest.raises(HuntSetupError, match="duration"):
        _ = run_hunt(manifest_path, bank_path, source, fuzz_seconds=0)


def _git_target(root: Path) -> tuple[Path, str]:
    source = root / "target"
    source.mkdir()
    include = source / "include"
    include.mkdir()
    _ = (include / "header.hpp").write_bytes(b"// pinned target\n")
    _ = (source / "SECURITY.md").write_bytes(SECURITY)
    commands = (
        ("/usr/bin/git", "init", "-q"),
        ("/usr/bin/git", "config", "user.email", "test@example.invalid"),
        ("/usr/bin/git", "config", "user.name", "Test"),
        ("/usr/bin/git", "remote", "add", "origin", REPOSITORY_URL),
        ("/usr/bin/git", "add", "."),
        ("/usr/bin/git", "commit", "-qm", "target"),
    )
    for command in commands:
        _ = subprocess.run(  # noqa: S603 - fixed test-only git argv
            command, cwd=source, check=True, capture_output=True
        )
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
        hunt_id="test-hunt",
        target_id="test-parser",
        repository_url=REPOSITORY_URL,
        commit_sha=commit,
        security_policy_url="https://github.com/example/parser/security/advisories/new",
        security_policy_sha256=hashlib.sha256(SECURITY).hexdigest(),
        prompt="prompts/prompt.md",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        image_id=IMAGE_ID,
        allowed_includes=("cstddef", "jsoncons_ext/ubjson/ubjson.hpp"),
        required_symbols=(
            "LLVMFuzzerTestOneInput",
            "try_decode_ubjson",
            "ubjson_bytes_cursor",
            "std::error_code",
        ),
        seed_hex=("5a",),
        fuzz_seconds=1,
        max_len=64,
        rss_limit_mb=256,
        timeout_seconds=5,
    )


def _candidate_bank(prompt: bytes) -> CandidateBank:
    source_hash = hashlib.sha256(HARNESS.encode()).hexdigest()

    def candidate(candidate_id: str, confidence: float) -> HarnessCandidate:
        return HarnessCandidate(
            candidate_id=candidate_id,
            source=HARNESS,
            source_sha256=source_hash,
            seed=1,
            nfe=2,
            steps=(
                DenoisingStep(
                    step_index=1,
                    masked_tokens_before=4,
                    committed_tokens=4,
                    mean_confidence=confidence,
                    state_sha256=("1" if candidate_id.endswith("01") else "2") * 64,
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
        candidates=(candidate("candidate-01", 0.7), candidate("candidate-02", 0.9)),
    )
