"""External target-only OpenEXR coverage-series behavior.

Module-size opt-out ``# noqa: SIZE_OK`` keeps Task 4 coverage regressions in one owned test.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Generator, Mapping  # noqa: TC003
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final

import pytest
from pydantic import TypeAdapter

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue, append_event
from dllm_agent.experiments.openexr_one_day import coverage_series
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRecord,
    CorpusIndexRequest,
    CoverageSeriesRequest,
    CoverageStepResult,
    TargetCoverageCounts,
    build_coverage_series,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    persist_backing_bytes,
    validate_backing_file,
)
from dllm_agent.targets.openexr.probe_coverage import (
    CoverageBoundaryError,
    DockerCoverageReplayer,
    ProfileMergeRequest,
    build_incremental_profile_merge_argv,
    parse_target_coverage_export,
)
from tests.openexr_one_day_task4_fakes import (
    ScriptedCoverageReplayer,
    coverage_step,
    host_path,
)

SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def test_llvm_export_counts_only_canonical_openexr_target_files() -> None:
    # Given
    raw = _llvm_export(
        (
            ("/opt/openexr-src/src/lib/OpenEXR/ImfInputFile.cpp", 5, 15, 7, 20, 9, 30),
            ("/opt/harness/openexr_fuzzer.cc", 99, 100, 99, 100, 99, 100),
        )
    )

    # When
    counts = parse_target_coverage_export(raw)

    # Then
    assert (counts.branches_covered, counts.branches_total) == (5, 15)
    assert (counts.regions_covered, counts.regions_total) == (7, 20)
    assert (counts.lines_covered, counts.lines_total) == (9, 30)
    assert counts.target_file_count == 1


def test_llvm_export_rejects_percentage_only_and_missing_target_data() -> None:
    # Given
    percentage_only = json.dumps(
        {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/opt/openexr-src/src/lib/OpenEXR/A.cpp",
                            "summary": {"branches": {"percent": 50.0}},
                        }
                    ]
                }
            ]
        }
    ).encode()

    # When / Then
    with pytest.raises(CoverageBoundaryError, match="integer"):
        _ = parse_target_coverage_export(percentage_only)
    with pytest.raises(CoverageBoundaryError, match="target file"):
        _ = parse_target_coverage_export(
            _llvm_export((("/opt/harness/fuzzer.cc", 1, 2, 1, 2, 1, 2),))
        )


def test_corpus_index_uses_exact_schedule_and_deduplicates_renamed_bytes() -> None:
    # Given
    # When
    with _data_root() as root:
        request = _corpus_request(root)
        request.corpus_root.mkdir(parents=True)
        _ = (request.corpus_root / "seed-a").write_bytes(b"same")
        rows: list[CorpusIndexRecord] = []
        for scheduled in range(0, 10_801, 900):
            if scheduled == 900:
                _ = (request.corpus_root / "seed-a").rename(request.corpus_root / "renamed")
                _ = (request.corpus_root / "duplicate").write_bytes(b"same")
            if scheduled == 1_800:
                _ = (request.corpus_root / "new").write_bytes(b"different")
            rows.append(snapshot_corpus(request, scheduled, scheduled + 7.0))
        loaded = load_corpus_index(request)

        # Then
        assert tuple(row.scheduled_elapsed_seconds for row in loaded) == tuple(
            range(0, 10_801, 900)
        )
        assert loaded[0].cumulative_count == 1
        assert loaded[1].new_inputs == ()
        assert loaded[2].cumulative_count == 2
        assert loaded[-1].cumulative_count == 2
        assert all(row.observed_lateness_seconds == 7.0 for row in rows)
        assert loaded[-1].cumulative_hashes == tuple(sorted(loaded[-1].cumulative_hashes))


def test_qualification_smoke_corpus_index_uses_request_schedule_only() -> None:
    # Given
    with _data_root() as root:
        request = _corpus_request(
            root,
            execution_profile="qualification-smoke",
            schedule_seconds=(0, 300),
        )
        request.corpus_root.mkdir(parents=True)
        _ = (request.corpus_root / "seed").write_bytes(b"seed")

        # When
        _ = snapshot_corpus(request, 0, 0.0)
        rows = coverage_series.freeze_corpus_index(request, freeze_elapsed_seconds=10.0)

        # Then
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == (0, 300)
        assert tuple(row.observation_kind for row in rows) == ("observed", "frozen")


def test_coverage_series_replays_only_new_hashes_and_chains_profiles() -> None:
    # Given
    # When
    with _data_root() as root:
        index_request = _corpus_request(root)
        index_request.corpus_root.mkdir(parents=True)
        _ = (index_request.corpus_root / "a").write_bytes(b"a")
        for scheduled in range(0, 10_801, 900):
            if scheduled == 900:
                _ = (index_request.corpus_root / "b").write_bytes(b"b")
            _ = snapshot_corpus(index_request, scheduled, float(scheduled))
        first_step = _backed_step_result(index_request.artifact_root, 0, 3, 10, b"profile-0")
        second_step = _backed_step_result(index_request.artifact_root, 900, 5, 15, b"profile-900")
        replayer = ScriptedCoverageReplayer(
            (
                first_step,
                second_step,
            )
        )
        series = build_coverage_series(
            CoverageSeriesRequest(
                corpus_index=index_request,
                target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
                image_digest=f"sha256:{'6' * 64}",
                binary_sha256=SHA_A,
                build_manifest_sha256=SHA_B,
                cpu_index=8,
            ),
            replayer,
        )

        # Then
        assert len(replayer.requests) == 2
        assert replayer.requests[0].prior_profdata_sha256 is None
        assert replayer.requests[1].prior_profdata_sha256 == first_step.profdata_sha256
        assert replayer.requests[0].new_input_sha256 == (hashlib.sha256(b"a").hexdigest(),)
        assert replayer.requests[1].new_input_sha256 == (hashlib.sha256(b"b").hexdigest(),)
        assert tuple(point.scheduled_elapsed_seconds for point in series.points) == tuple(
            range(0, 10_801, 900)
        )
        assert series.points[-1].branches_covered == 5
        assert series.points[-1].branches_total == 15
        assert series.points[-1].corpus_set_sha256 == series.corpus_index[-1].cumulative_set_sha256


def test_coverage_series_rejects_cross_profile_corpus_evidence() -> None:
    # Given
    with _data_root() as root:
        main_index = _corpus_request(root)
        main_index.corpus_root.mkdir(parents=True)
        _ = (main_index.corpus_root / "seed").write_bytes(b"seed")
        for scheduled in range(0, 10_801, 900):
            _ = snapshot_corpus(main_index, scheduled, float(scheduled))
        smoke_index = main_index.model_copy(
            update={
                "execution_profile": "qualification-smoke",
                "schedule_seconds": (0, 300),
            }
        )
        replayer = ScriptedCoverageReplayer(
            (_backed_step_result(main_index.artifact_root, 0, 5, 15, b"profile"),)
        )

        # When / Then
        with pytest.raises(coverage_series.CoverageSeriesError, match="schedule"):
            _ = build_coverage_series(
                CoverageSeriesRequest(
                    execution_profile="qualification-smoke",
                    schedule_seconds=(0, 300),
                    corpus_index=smoke_index,
                    target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
                    image_digest=f"sha256:{'6' * 64}",
                    binary_sha256=SHA_A,
                    build_manifest_sha256=SHA_B,
                    cpu_index=8,
                ),
                replayer,
            )


def test_frozen_corpus_rows_produce_explicit_frozen_coverage_points() -> None:
    # Given
    with _data_root() as root:
        index_request = _corpus_request(root)
        index_request.corpus_root.mkdir(parents=True)
        _ = (index_request.corpus_root / "seed").write_bytes(b"seed")
        _ = snapshot_corpus(index_request, 0, 0.0)
        rows = coverage_series.freeze_corpus_index(index_request, freeze_elapsed_seconds=10.0)
        replayer = ScriptedCoverageReplayer(
            (_backed_step_result(index_request.artifact_root, 0, 5, 15, b"profile-frozen"),)
        )

        # When
        series = build_coverage_series(
            CoverageSeriesRequest(
                corpus_index=index_request,
                target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
                image_digest=f"sha256:{'6' * 64}",
                binary_sha256=SHA_A,
                build_manifest_sha256=SHA_B,
                cpu_index=8,
            ),
            replayer,
        )

        # Then
        assert len(rows) == 13
        assert len(replayer.requests) == 1
        assert tuple(point.observation_kind for point in series.points) == (
            "observed",
            *("frozen",) * 12,
        )
        assert tuple(point.source_elapsed_seconds for point in series.points) == (
            0.0,
            *(10.0,) * 12,
        )
        assert {point.branches_covered for point in series.points} == {5}


def test_coverage_series_rejects_hash_only_step_results_without_backing_evidence() -> None:
    # Given
    with _data_root() as root:
        index_request = _corpus_request(root)
        index_request.corpus_root.mkdir(parents=True)
        _ = (index_request.corpus_root / "seed").write_bytes(b"seed")
        _ = snapshot_corpus(index_request, 0, 0.0)
        _ = coverage_series.freeze_corpus_index(index_request, freeze_elapsed_seconds=1.0)
        replayer = ScriptedCoverageReplayer((_step_result(5, 15, SHA_C),))

        # When / Then
        with pytest.raises(coverage_series.CoverageSeriesError, match="dereferenceable evidence"):
            _ = build_coverage_series(
                CoverageSeriesRequest(
                    corpus_index=index_request,
                    target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
                    image_digest=f"sha256:{'6' * 64}",
                    binary_sha256=SHA_A,
                    build_manifest_sha256=SHA_B,
                    cpu_index=8,
                ),
                replayer,
            )


def test_corpus_index_reload_rejects_escaped_backing_path() -> None:
    # Given
    with _data_root() as root:
        request = _corpus_request(root)
        request.corpus_root.mkdir(parents=True)
        _ = (request.corpus_root / "seed").write_bytes(b"seed")
        record = snapshot_corpus(request, 0, 0.0)
        escaped_input = record.new_inputs[0].model_copy(update={"stored_path": "../seed"})
        escaped = record.model_copy(update={"new_inputs": (escaped_input,)})
        request.index_path.unlink()
        _ = append_event(
            request.index_path,
            timestamp_utc="tampered+00000",
            event_type="corpus_index",
            payload=_JSON_OBJECT.validate_python(escaped.model_dump(mode="json")),
        )

        # When / Then
        with pytest.raises(
            coverage_series.CoverageSeriesError,
            match=r"canonical relative|escapes",
        ):
            _ = load_corpus_index(request)


def test_backing_evidence_rejects_symlink_and_special_file_on_reload() -> None:
    # Given
    with _data_root() as root:
        evidence_root = root / "cell"
        real = persist_backing_bytes(evidence_root, "evidence/real.bin", b"real")
        symlink = evidence_root / "evidence/link.bin"
        symlink.symlink_to(evidence_root / real.relative_path)
        symlink_dir = evidence_root / "alias"
        symlink_dir.symlink_to(evidence_root / "evidence", target_is_directory=True)
        fifo = evidence_root / "evidence/fifo.bin"
        os.mkfifo(fifo)

        # When / Then
        with pytest.raises(EvidenceBackingError, match="symlink"):
            _ = validate_backing_file(
                evidence_root,
                ArtifactBackingEvidence(
                    relative_path="evidence/link.bin",
                    size_bytes=real.size_bytes,
                    sha256=real.sha256,
                ),
            )
        with pytest.raises(EvidenceBackingError, match="contains a symlink"):
            _ = validate_backing_file(
                evidence_root,
                ArtifactBackingEvidence(
                    relative_path="alias/real.bin",
                    size_bytes=real.size_bytes,
                    sha256=real.sha256,
                ),
            )
        with pytest.raises(EvidenceBackingError, match="regular file"):
            _ = validate_backing_file(
                evidence_root,
                ArtifactBackingEvidence(
                    relative_path="evidence/fifo.bin",
                    size_bytes=0,
                    sha256=sha256_bytes(b""),
                ),
            )


def test_incremental_merge_argv_includes_prior_profile_and_new_profraw() -> None:
    # Given
    request = ProfileMergeRequest(
        prior_profdata_path=Path("/work/prior.profdata"),
        new_profraw_paths=(Path("/work/a.profraw"), Path("/work/b.profraw")),
        output_profdata_path=Path("/work/current.profdata"),
    )

    # When
    argv = build_incremental_profile_merge_argv(request)

    # Then
    assert argv == (
        "/usr/local/bin/llvm-profdata",
        "merge",
        "-sparse",
        "/work/prior.profdata",
        "/work/a.profraw",
        "/work/b.profraw",
        "-o",
        "/work/current.profdata",
    )


def test_production_replayer_persists_incremental_profiles_and_target_report() -> None:
    # Given
    with _data_root() as root:
        corpus = _corpus_request(root)
        corpus.corpus_root.mkdir(parents=True)
        first_input = corpus.corpus_root / "a"
        second_input = corpus.corpus_root / "b"
        _ = first_input.write_bytes(b"a")
        _ = second_input.write_bytes(b"b")
        series_request = CoverageSeriesRequest(
            corpus_index=corpus,
            target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
            image_digest=f"sha256:{'6' * 64}",
            binary_sha256=SHA_A,
            build_manifest_sha256=SHA_B,
            cpu_index=8,
        )
        calls: list[tuple[str, ...]] = []
        merge_outputs = iter((b"profile-0", b"profile-900"))

        def runner(
            argv: tuple[str, ...], _environment: Mapping[str, str], _timeout: float
        ) -> CompletedProcess[bytes]:
            calls.append(argv)
            if "/usr/local/bin/llvm-profdata" in argv:
                output = host_path(corpus.artifact_root, argv[-1])
                _ = output.write_bytes(next(merge_outputs))
                return CompletedProcess(argv, 0, b"", b"")
            if "/usr/local/bin/llvm-cov" in argv:
                report = _llvm_export(
                    (("/opt/openexr-src/src/lib/OpenEXR/ImfInputFile.cpp", 5, 15, 7, 20, 9, 30),)
                )
                return CompletedProcess(argv, 0, report, b"")
            profile = next(item for item in argv if item.startswith("LLVM_PROFILE_FILE="))
            _ = host_path(corpus.artifact_root, profile.split("=", 1)[1]).write_bytes(
                f"raw-{len(calls)}".encode()
            )
            return CompletedProcess(argv, 0, b"run\n", b"")

        replayer = DockerCoverageReplayer(series_request, runner=runner)

        # When
        first = replayer.replay(coverage_step(0, first_input, prior=None))
        second = replayer.replay(coverage_step(900, second_input, prior=first.profdata_sha256))

        # Then
        coverage_root = corpus.artifact_root / "coverage"
        assert (coverage_root / "t00000/profile.profdata").read_bytes() == b"profile-0"
        assert (coverage_root / "t00900/profile.profdata").read_bytes() == b"profile-900"
        assert (coverage_root / "current.profdata").read_bytes() == b"profile-900"
        assert (coverage_root / "t00900/report.json").read_bytes().startswith(b'{"data"')
        assert first.run_argv_evidence[0].sha256 == first.run_argv_sha256[0]
        assert second.profraw_evidence[0].sha256 == second.profraw_sha256[0]
        assert second.merge_argv_evidence is not None
        assert second.merge_argv_evidence.sha256 == second.merge_argv_sha256
        assert second.profdata_evidence is not None
        assert second.profdata_evidence.sha256 == second.profdata_sha256
        assert second.report_argv_evidence is not None
        assert second.report_argv_evidence.sha256 == second.report_argv_sha256
        assert second.report_evidence is not None
        assert second.report_evidence.sha256 == second.report_sha256
        run_argv_bytes = json.dumps(calls[0], separators=(",", ":")).encode()
        assert (
            corpus.artifact_root / first.run_argv_evidence[0].relative_path
        ).read_bytes() == run_argv_bytes
        assert (
            (corpus.artifact_root / second.report_evidence.relative_path)
            .read_bytes()
            .startswith(b'{"data"')
        )
        assert second.counts.branches_covered == 5
        assert "/work/coverage/current.profdata" in calls[4]
        assert (coverage_root / "t00900/input-0000.profraw").read_bytes() == b"raw-4"

        _ = (coverage_root / "current.profdata").write_bytes(b"corrupt")
        _ = (third_input := corpus.corpus_root / "c").write_bytes(b"c")
        with pytest.raises(CoverageBoundaryError, match="prior profile hash"):
            _ = replayer.replay(coverage_step(1_800, third_input, prior=second.profdata_sha256))
        assert len(calls) == 6


@contextmanager
def _data_root() -> Generator[Path]:
    with tempfile.TemporaryDirectory(prefix="dllm-task4-coverage-", dir="/data") as directory:
        yield Path(directory)


def _corpus_request(
    root: Path,
    *,
    execution_profile: coverage_series.ExecutionProfile = "main",
    schedule_seconds: tuple[int, ...] = coverage_series.SCHEDULE_SECONDS,
) -> CorpusIndexRequest:
    artifact_root = root / "cell"
    return CorpusIndexRequest(
        execution_profile=execution_profile,
        schedule_seconds=schedule_seconds,
        artifact_root=artifact_root,
        corpus_root=artifact_root / "corpus",
        store_root=artifact_root / "corpus-inputs",
        index_path=artifact_root / "corpus-index.jsonl",
    )


def _step_result(covered: int, total: int, profile: str) -> CoverageStepResult:
    return CoverageStepResult(
        counts=TargetCoverageCounts(
            target_file_count=1,
            branches_covered=covered,
            branches_total=total,
            regions_covered=covered + 1,
            regions_total=total + 1,
            lines_covered=covered + 2,
            lines_total=total + 2,
        ),
        run_argv_sha256=(SHA_A,),
        profraw_sha256=(SHA_B,),
        merge_argv_sha256=SHA_C,
        profdata_sha256=profile,
        report_argv_sha256=SHA_A,
        report_sha256=SHA_B,
    )


def _backed_step_result(
    artifact_root: Path,
    scheduled: int,
    covered: int,
    total: int,
    profile_content: bytes,
) -> CoverageStepResult:
    run_argv_bytes = json.dumps(("run", scheduled), separators=(",", ":")).encode()
    profraw_bytes = f"profraw:{scheduled}".encode()
    merge_argv_bytes = json.dumps(("merge", scheduled), separators=(",", ":")).encode()
    report_argv_bytes = json.dumps(("report", scheduled), separators=(",", ":")).encode()
    report_bytes = f"report:{scheduled}".encode()
    return CoverageStepResult(
        counts=TargetCoverageCounts(
            target_file_count=1,
            branches_covered=covered,
            branches_total=total,
            regions_covered=covered + 1,
            regions_total=total + 1,
            lines_covered=covered + 2,
            lines_total=total + 2,
        ),
        run_argv_sha256=(sha256_bytes(run_argv_bytes),),
        profraw_sha256=(sha256_bytes(profraw_bytes),),
        merge_argv_sha256=sha256_bytes(merge_argv_bytes),
        profdata_sha256=sha256_bytes(profile_content),
        report_argv_sha256=sha256_bytes(report_argv_bytes),
        report_sha256=sha256_bytes(report_bytes),
        run_argv_evidence=(
            persist_backing_bytes(
                artifact_root,
                f"coverage/t{scheduled:05d}/run-argv-0000.bin",
                run_argv_bytes,
            ),
        ),
        profraw_evidence=(
            persist_backing_bytes(
                artifact_root,
                f"coverage/t{scheduled:05d}/input-0000.profraw",
                profraw_bytes,
            ),
        ),
        merge_argv_evidence=persist_backing_bytes(
            artifact_root,
            f"coverage/t{scheduled:05d}/merge-argv.bin",
            merge_argv_bytes,
        ),
        profdata_evidence=persist_backing_bytes(
            artifact_root,
            f"coverage/t{scheduled:05d}/profile.profdata",
            profile_content,
        ),
        report_argv_evidence=persist_backing_bytes(
            artifact_root,
            f"coverage/t{scheduled:05d}/report-argv.bin",
            report_argv_bytes,
        ),
        report_evidence=persist_backing_bytes(
            artifact_root,
            f"coverage/t{scheduled:05d}/report.json",
            report_bytes,
        ),
    )


def _llvm_export(files: tuple[tuple[str, int, int, int, int, int, int], ...]) -> bytes:
    return json.dumps(
        {
            "data": [
                {
                    "files": [
                        {
                            "filename": path,
                            "summary": {
                                "branches": {"covered": bc, "count": bt},
                                "regions": {"covered": rc, "count": rt},
                                "lines": {"covered": lc, "count": lt},
                            },
                        }
                        for path, bc, bt, rc, rt, lc, lt in files
                    ]
                }
            ]
        },
        separators=(",", ":"),
    ).encode()
