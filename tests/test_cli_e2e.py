"""Installed CLI end-to-end tests."""

import hashlib
import subprocess
import sys
from pathlib import Path

from dllm_agent.core.models import ExpandedPlan, ScanReport
from dllm_agent.hunt.models import (
    CandidateBank,
    DenoisingStep,
    GenerationEvidence,
    HarnessCandidate,
    LiveModelProvenance,
)
from dllm_agent.targets.models import HuntManifest

ROOT = Path(__file__).parents[1]
CLI = str(Path(sys.executable).parent / "dllm-agent")
MODEL_ID = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"


def test_installed_scan_writes_versioned_report(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "report.json"
    command = (
        CLI,
        "scan",
        "fixtures/cwe121/vuln.c",
        "--fixture",
        "fixtures/replay/stable-diffcoder-cwe121.json",
        "--output",
        str(output),
    )

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )

    # Then
    assert result.returncode == 0
    report = ScanReport.model_validate_json(output.read_bytes())
    assert report.validator.status == "proved"


def test_installed_plan_writes_expanded_plan(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "plan.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "plan",
            "configs/owned-fixture-mvp.json",
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 0
    plan = ExpandedPlan.model_validate_json(output.read_bytes())
    assert plan.schema_version == "dllm-plan/v1"


def test_installed_cli_returns_two_without_output_for_malformed_config(tmp_path: Path) -> None:
    # Given
    config = tmp_path / "bad.json"
    _ = config.write_text("{", encoding="utf-8")
    output = tmp_path / "plan.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (CLI, "plan", str(config), "--output", str(output)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert not output.exists()


def test_installed_cli_returns_two_without_output_for_missing_target(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "scan",
            "missing.c",
            "--fixture",
            "fixtures/replay/stable-diffcoder-cwe121.json",
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert not output.exists()


def test_installed_cli_rejects_untrusted_fixture_before_parsing(tmp_path: Path) -> None:
    # Given
    fixture = tmp_path / "bad-fixture.json"
    _ = fixture.write_text("{", encoding="utf-8")
    output = tmp_path / "report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "scan",
            "fixtures/cwe121/vuln.c",
            "--fixture",
            str(fixture),
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert "trusted repository fixture" in result.stderr
    assert not output.exists()


def test_installed_scan_rejects_external_matching_inputs_without_execution(
    tmp_path: Path,
) -> None:
    # Given
    marker = tmp_path / "executed.txt"
    target = tmp_path / "external.c"
    source = (
        "#include <stdio.h>\n"
        "int main(void) {\n"
        f'    FILE *file = fopen("{marker.as_posix()}", "w");\n'
        '    if (file != NULL) { fputs("executed", file); fclose(file); }\n'
        "    return 0;\n"
        "}\n"
    )
    _ = target.write_text(source, encoding="utf-8")
    target_hash = hashlib.sha256(source.encode()).hexdigest()
    trusted_fixture = ROOT / "fixtures/replay/stable-diffcoder-cwe121.json"
    fixture = tmp_path / "matching-fixture.json"
    raw = trusted_fixture.read_text(encoding="utf-8").replace(
        "42e55e7371d23312996c73b374e47d2459469e9fc4ef2f0dac592c3b15985144",
        target_hash,
        1,
    )
    _ = fixture.write_text(raw, encoding="utf-8")
    output = tmp_path / "report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "scan",
            str(target),
            "--fixture",
            str(fixture),
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert "trusted repository fixture" in result.stderr
    assert not marker.exists()
    assert not output.exists()


def test_installed_web_help_exposes_loopback_server_controls() -> None:
    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (CLI, "web", "--help"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "127.0.0.1" in result.stdout
    assert "--port" in result.stdout
    assert "--runtime-python" in result.stdout
    assert "--source-root" in result.stdout


def test_installed_web_rejects_non_loopback_bind_before_server_start() -> None:
    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (CLI, "web", "--host", "0.0.0.0"),  # noqa: S104 - intentional rejection case.
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert "must bind to a loopback address" in result.stderr


def test_installed_native_hunt_help_exposes_controller_inputs() -> None:
    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (CLI, "native-hunt", "--help"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 0
    assert "--candidate-bank" in result.stdout
    assert "--controller" in result.stdout
    assert "--source" in result.stdout
    assert "--artifact-dir" in result.stdout
    assert "--output" in result.stdout


def test_installed_native_hunt_rejects_malformed_controller_before_artifacts(
    tmp_path: Path,
) -> None:
    # Given: all earlier boundaries are valid, so only controller parsing can reject the run.
    manifest, bank = _native_hunt_boundaries(tmp_path)
    controller = tmp_path / "controller.json"
    _ = controller.write_text("{", encoding="utf-8")
    artifact_dir = tmp_path / "native-run"
    output = tmp_path / "native-report.json"

    # When
    result = subprocess.run(  # noqa: S603 - installed project CLI
        (
            CLI,
            "native-hunt",
            str(manifest),
            "--candidate-bank",
            str(bank),
            "--controller",
            str(controller),
            "--source",
            str(tmp_path / "target"),
            "--artifact-dir",
            str(artifact_dir),
            "--output",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert "native hunt config" in result.stderr
    assert not output.exists()
    assert not artifact_dir.exists()


def _native_hunt_boundaries(root: Path) -> tuple[Path, Path]:
    prompt = root / "prompts/native.md"
    prompt.parent.mkdir()
    prompt_bytes = b"generate a deterministic in-memory harness"
    _ = prompt.write_bytes(prompt_bytes)
    manifest = root / "configs/hunt.json"
    manifest.parent.mkdir()
    _ = manifest.write_text(
        HuntManifest(
            schema_version="dllm-hunt/v1",
            hunt_id="native-cli-test",
            target_id="parser",
            repository_url="https://github.com/example/parser.git",
            commit_sha="a" * 40,
            security_policy_url="https://github.com/example/parser/security/advisories/new",
            security_policy_sha256="b" * 64,
            prompt="prompts/native.md",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            image_id="sha256:" + ("c" * 64),
            allowed_includes=("cstddef",),
            required_symbols=("LLVMFuzzerTestOneInput",),
            seed_hex=("41",),
            fuzz_seconds=1,
            max_len=64,
            rss_limit_mb=256,
            timeout_seconds=5,
        ).model_dump_json(),
        encoding="utf-8",
    )
    source = (
        "#include <cstddef>\n"
        'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, std::size_t) '
        "{ return 0; }\n"
    )
    candidate = HarnessCandidate(
        candidate_id="candidate-01",
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        seed=20260819,
        nfe=1,
        steps=(
            DenoisingStep(
                step_index=1,
                masked_tokens_before=1,
                committed_tokens=1,
                mean_confidence=0.9,
                state_sha256="d" * 64,
            ),
        ),
    )
    second = candidate.model_copy(
        update={"candidate_id": "candidate-02", "source_sha256": candidate.source_sha256}
    )
    bank = root / "candidate-bank.json"
    _ = bank.write_text(
        CandidateBank(
            schema_version="dllm-candidate-bank/v1",
            prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
            model=LiveModelProvenance(
                model_id=MODEL_ID,
                revision=MODEL_REVISION,
                remote_code_sha256="e" * 64,
                python_version="3.12.13",
                transformers_version="5.3.0",
                torch_version="2.5.1+cu121",
                device="cuda:0",
                generation=GenerationEvidence(
                    steps=1,
                    gen_length=4,
                    block_length=1,
                    temperature=0.2,
                ),
            ),
            candidates=(candidate, second),
        ).model_dump_json(),
        encoding="utf-8",
    )
    return manifest, bank
