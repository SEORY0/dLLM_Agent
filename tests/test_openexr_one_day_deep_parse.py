"""Task 7 OpenEXR deep-parse runtime and evidence regressions.

# noqa: SIZE_OK -- one trust-boundary regression module keeps shared Docker fixtures local.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import TypeAdapter

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.post_search import (
    DeepParseEvidenceArtifact,
    DeepParseRequest,
    DeepParseResult,
    HashOnlyDeepParseReplayer,
    PostSearchEvidenceError,
    _load_or_build_deep_parse,
    validate_deep_parse_result_evidence,
)
from dllm_agent.targets.openexr.deep_parse import (
    DeepParseInfrastructureError,
    DockerDeepParseReplayer,
    InfrastructureFailureKind,
    OpenExrRuntimeBuildManifest,
)
from tests.openexr_one_day_deep_parse_fixtures import (
    BUILD_FILES,
    CHECKER_PATH,
    CHECKER_SHA256,
    IMAGE_DIGEST,
    MANIFEST_PATH,
    ScriptedDockerRunner,
    build_manifest_bytes,
    deep_parse_cell,
    deep_parse_request,
    final_coverage,
    private_cell_root,
)

if TYPE_CHECKING:
    from dllm_agent.core.models import Sha256

ROOT = Path(__file__).resolve().parents[1]
_EXIT_JSON = TypeAdapter(dict[str, int | str | None])


class _UnexpectedTestOnlyReplayer:
    """Production-shaped fake that unexpectedly emits test-only evidence."""

    checker_binary_sha256: Sha256 = "0" * 64

    def replay(self, request: DeepParseRequest) -> DeepParseResult:
        return HashOnlyDeepParseReplayer().replay(request)


def test_docker_replayer_maps_checker_codes_and_persists_exact_evidence(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "cell"
    request = deep_parse_request(root)
    runner = ScriptedDockerRunner()

    # When
    result = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
        request
    )

    # Then
    assert tuple(verdict.parse_success for verdict in result.verdicts) == (True, False)
    parse_calls = tuple(call for call in runner.calls if call[-2:-1] == (CHECKER_PATH,))
    assert len(parse_calls) == 2
    for index, (call, verdict) in enumerate(zip(parse_calls, result.verdicts, strict=True)):
        assert call[call.index("--network") :][:2] == ("--network", "none")
        assert call[call.index("--pull") :][:2] == ("--pull", "never")
        assert "--read-only" in call
        assert call[call.index("--cap-drop") :][:2] == ("--cap-drop", "ALL")
        assert call[call.index("--user") :][:2] == ("--user", "65534:65534")
        assert call[call.index("-v") :][:2] == ("-v", f"{root}:/work:ro")
        assert call[-2:] == (
            CHECKER_PATH,
            f"/work/post-search/deep-parse-evidence/runs/{index:06d}/input.exr",
        )
        _assert_artifact(result.evidence_root, verdict.input_evidence)
        _assert_artifact(result.evidence_root, verdict.argv_evidence)
        _assert_artifact(result.evidence_root, verdict.stdout_evidence)
        _assert_artifact(result.evidence_root, verdict.stderr_evidence)
        _assert_artifact(result.evidence_root, verdict.exit_evidence)
        assert (result.evidence_root / verdict.argv_evidence.relative_path).read_bytes() == (
            canonical_json_bytes(call)
        )
        assert (result.evidence_root / verdict.input_evidence.relative_path).read_bytes() == (
            request.input_paths[index].read_bytes()
        )
    assert result.success_count == 1
    assert result.total_count == 2
    assert result.replay_kind == "docker"
    assert result.build_manifest_evidence is not None
    _assert_artifact(result.evidence_root, result.build_manifest_evidence)
    assert (result.evidence_root / result.build_manifest_evidence.relative_path).read_bytes() == (
        build_manifest_bytes()
    )


@pytest.mark.parametrize(
    ("return_code", "stderr", "expected_kind"),
    [
        (2, b"cannot read input", "checker"),
        (1, b"ERROR: AddressSanitizer: heap-use-after-free", "sanitizer"),
    ],
)
def test_checker_infrastructure_failures_are_typed_and_evidenced(
    tmp_path: Path,
    return_code: int,
    stderr: bytes,
    expected_kind: str,
) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    runner = ScriptedDockerRunner(
        parse_codes=(return_code,),
        parse_stdout=(b"",),
        parse_stderr=(stderr,),
    )

    # When / Then
    with pytest.raises(DeepParseInfrastructureError) as captured:
        _ = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
            request
        )
    assert captured.value.kind == expected_kind
    assert captured.value.evidence is not None
    _assert_artifact(captured.value.evidence_root, captured.value.evidence.stderr_evidence)
    assert (
        captured.value.evidence_root / captured.value.evidence.stderr_evidence.relative_path
    ).read_bytes() == stderr


@pytest.mark.parametrize(
    ("failure", "expected_kind"), [("timeout", "timeout"), ("docker", "docker")]
)
def test_process_boundary_failures_are_typed_and_evidenced(
    tmp_path: Path,
    failure: Literal["timeout", "docker"],
    expected_kind: InfrastructureFailureKind,
) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    runner = ScriptedDockerRunner(
        parse_codes=(0,),
        parse_stdout=(b"",),
        parse_stderr=(b"",),
        failure=failure,
    )

    # When / Then
    with pytest.raises(DeepParseInfrastructureError) as captured:
        _ = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
            request
        )
    assert captured.value.kind == expected_kind
    assert captured.value.evidence is not None
    _assert_artifact(captured.value.evidence_root, captured.value.evidence.exit_evidence)


def test_replayer_rejects_input_hash_mismatch_before_docker(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",)).model_copy(
        update={"input_sha256": ("f" * 64,)}
    )
    runner = ScriptedDockerRunner(parse_codes=(0,), parse_stdout=(b"",), parse_stderr=(b"",))

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="input hash or size mismatch"):
        _ = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
            request
        )
    assert runner.calls == []


def test_replayer_rejects_input_path_outside_exact_root(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    outside = tmp_path / "outside.exr"
    _ = outside.write_bytes(b"one")
    escaped = request.model_copy(update={"input_paths": (outside,)})

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="escapes artifact root"):
        _ = DockerDeepParseReplayer(
            checker_binary_sha256=CHECKER_SHA256, runner=ScriptedDockerRunner()
        ).replay(escaped)


def test_replayer_rejects_symlinked_input(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    target = request.input_paths[0]
    link = target.with_name("link.exr")
    link.symlink_to(target)
    linked = request.model_copy(update={"input_paths": (link,)})

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="regular non-symlink file"):
        _ = DockerDeepParseReplayer(
            checker_binary_sha256=CHECKER_SHA256, runner=ScriptedDockerRunner()
        ).replay(linked)


def test_result_evidence_validation_rejects_tampering(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    runner = ScriptedDockerRunner(parse_codes=(0,), parse_stdout=(b"ok",), parse_stderr=(b"",))
    result = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
        request
    )
    stdout = result.verdicts[0].stdout_evidence
    _ = (result.evidence_root / stdout.relative_path).write_bytes(b"tampered")

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="hash or size mismatch"):
        validate_deep_parse_result_evidence(result)


def test_result_evidence_rejects_input_hash_detached_from_backing(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",))
    runner = ScriptedDockerRunner(parse_codes=(0,), parse_stdout=(b"",), parse_stderr=(b"",))
    result = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
        request
    )
    verdict = result.verdicts[0].model_copy(update={"input_sha256": "f" * 64})
    detached = result.model_copy(update={"verdicts": (verdict,)})

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="input backing mismatch"):
        validate_deep_parse_result_evidence(detached)


def test_replayer_rejects_checker_identity_mismatch_before_docker(tmp_path: Path) -> None:
    # Given
    request = deep_parse_request(
        tmp_path / "cell",
        inputs=(b"one",),
        checker_binary_sha256="e" * 64,
    )
    runner = ScriptedDockerRunner(parse_codes=(0,), parse_stdout=(b"",), parse_stderr=(b"",))

    # When / Then
    with pytest.raises(DeepParseInfrastructureError) as captured:
        _ = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
            request
        )
    assert captured.value.kind == "provenance"
    assert runner.calls == []


def test_replayer_rejects_manifest_checker_identity_mismatch(tmp_path: Path) -> None:
    # Given
    manifest = build_manifest_bytes(checker_sha256="e" * 64)
    request = deep_parse_request(tmp_path / "cell", inputs=(b"one",), manifest=manifest)
    runner = ScriptedDockerRunner(
        manifest=manifest, parse_codes=(0,), parse_stdout=(b"",), parse_stderr=(b"",)
    )

    # When / Then
    with pytest.raises(DeepParseInfrastructureError) as captured:
        _ = DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=runner).replay(
            request
        )
    assert captured.value.kind == "provenance"
    assert len(runner.calls) == 2


def test_existing_envelope_rejects_changed_current_request(tmp_path: Path) -> None:
    # Given
    root = private_cell_root(tmp_path)
    cell_request, cell_result = deep_parse_cell(root)
    coverage = final_coverage(root, b"first")
    manifest = build_manifest_bytes()
    path = root / "post-search/deep-parse-envelope.json"
    first_runner = ScriptedDockerRunner(parse_codes=(0,), parse_stdout=(b"",), parse_stderr=(b"",))
    _ = _load_or_build_deep_parse(
        path=path,
        cell_request=cell_request,
        cell_result=cell_result,
        coverage=coverage,
        image_digest=IMAGE_DIGEST,
        build_manifest_sha256=sha256_bytes(manifest),
        replayer=DockerDeepParseReplayer(checker_binary_sha256=CHECKER_SHA256, runner=first_runner),
    )

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="current request mismatch"):
        _ = _load_or_build_deep_parse(
            path=path,
            cell_request=cell_request,
            cell_result=cell_result,
            coverage=coverage,
            image_digest="sha256:" + "b" * 64,
            build_manifest_sha256=sha256_bytes(manifest),
            replayer=DockerDeepParseReplayer(
                checker_binary_sha256=CHECKER_SHA256, runner=ScriptedDockerRunner()
            ),
        )


def test_production_deep_parse_rejects_test_only_result(tmp_path: Path) -> None:
    # Given
    root = private_cell_root(tmp_path)
    cell_request, cell_result = deep_parse_cell(root)
    coverage = final_coverage(root, b"first")

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="test-only"):
        _ = _load_or_build_deep_parse(
            path=root / "post-search/deep-parse-envelope.json",
            cell_request=cell_request,
            cell_result=cell_result,
            coverage=coverage,
            image_digest=IMAGE_DIGEST,
            build_manifest_sha256="7" * 64,
            replayer=_UnexpectedTestOnlyReplayer(),
        )


def test_production_deep_parse_rejects_existing_test_only_result(tmp_path: Path) -> None:
    # Given
    root = private_cell_root(tmp_path)
    cell_request, cell_result = deep_parse_cell(root)
    coverage = final_coverage(root, b"first")
    path = root / "post-search/deep-parse-envelope.json"
    _ = _load_or_build_deep_parse(
        path=path,
        cell_request=cell_request,
        cell_result=cell_result,
        coverage=coverage,
        image_digest=IMAGE_DIGEST,
        build_manifest_sha256="7" * 64,
        replayer=HashOnlyDeepParseReplayer(),
    )

    # When / Then
    with pytest.raises(PostSearchEvidenceError, match="test-only"):
        _ = _load_or_build_deep_parse(
            path=path,
            cell_request=cell_request,
            cell_result=cell_result,
            coverage=coverage,
            image_digest=IMAGE_DIGEST,
            build_manifest_sha256="7" * 64,
            replayer=_UnexpectedTestOnlyReplayer(),
        )


def test_openexr_runtime_build_contract_has_ubsan_and_independent_coverage() -> None:
    # Given / When
    cmake = (ROOT / "tools/openexr_runtime/CMakeLists.txt").read_text(encoding="utf-8")
    dockerfile = (ROOT / "tools/openexr_runtime/Dockerfile").read_text(encoding="utf-8")
    checker = (ROOT / "tools/openexr_runtime/deep_parse_check.cc").read_text(encoding="utf-8")

    # Then
    assert "-fsanitize=fuzzer,address,undefined" in cmake
    assert "-fno-sanitize-recover=undefined" in cmake
    assert dockerfile.count("-fsanitize=address,undefined") >= 2
    assert "-fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping" in dockerfile
    assert "/opt/dllm/bin/openexr-deep-parse-check" in dockerfile
    assert "/opt/dllm/evidence/build-manifest.json" in dockerfile
    call = "OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, true)"
    assert checker.count(call) == 1
    assert "const bool has_error" in checker
    assert "return has_error ? 1 : 0;" in checker


def _assert_artifact(root: Path, artifact: DeepParseEvidenceArtifact) -> None:
    path = root / artifact.relative_path
    content = path.read_bytes()
    assert not artifact.relative_path.is_absolute()
    assert ".." not in artifact.relative_path.parts
    assert len(content) == artifact.size_bytes
    assert sha256_bytes(content) == artifact.sha256
    if artifact.relative_path.name == "exit.json":
        parsed = _EXIT_JSON.validate_json(content)
        assert set(parsed) in ({"return_code"}, {"failure", "return_code"})


def test_fixture_manifest_lists_every_provenance_file() -> None:
    # Given / When
    payload = OpenExrRuntimeBuildManifest.model_validate_json(build_manifest_bytes())

    # Then
    assert tuple(item.path for item in payload.files) == tuple(
        path for path, _content in BUILD_FILES
    )
    assert payload.deep_parse_checker_sha256 == CHECKER_SHA256
    assert sha256_bytes(build_manifest_bytes()) != sha256_bytes(b"")
    assert MANIFEST_PATH == "/opt/dllm/evidence/build-manifest.json"
    assert IMAGE_DIGEST.startswith("sha256:")
