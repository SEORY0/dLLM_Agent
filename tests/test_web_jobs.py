"""Web job manager safety tests."""

from pathlib import Path
from typing import Final

import pytest

from dllm_agent.web.contracts import HuntJobSpec, JobRuntimeConfig, SampleJobSpec
from dllm_agent.web.jobs import (
    JobBusyError,
    JobManager,
    ResourceResolver,
    UnknownJobError,
)
from dllm_agent.web.processes import ProcessFactory, ProcessLaunch

MANIFEST_ID: Final = "res_manifest"
BANK_ID: Final = "res_bank"


class RecordingProcess:
    """Minimal controllable subprocess test double."""

    def __init__(self, return_code: int | None = None) -> None:
        self.return_code: int | None = return_code

    def poll(self) -> int | None:
        return self.return_code

    def communicate(self) -> tuple[bytes, bytes]:
        return (b"out\n", b"err\n")


def test_sample_job_launches_fixed_argv_with_server_owned_paths(tmp_path: Path) -> None:
    # Given
    calls: list[dict[str, str | bool | tuple[str, ...]]] = []
    config = _runtime(tmp_path)
    manager = JobManager(
        config,
        resolver=_resolver(tmp_path),
        process_factory=_recording_factory(calls),
    )
    spec = SampleJobSpec(manifest_id=MANIFEST_ID, count=2, seed=11)

    # When
    status = manager.start(spec)

    # Then
    call = calls[0]
    argv = call["argv"]
    assert isinstance(argv, tuple)
    assert argv[:3] == (config.python_executable, "-m", "dllm_agent")
    assert argv[3:] == (
        "sample",
        str(tmp_path / "repo/configs/hunt.json"),
        "--runtime-python",
        str(tmp_path / "runtime/bin/python"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / f"jobs/{status.job_id}/candidate-bank.json"),
        "--count",
        "2",
        "--seed",
        "11",
    )
    assert call["shell"] is False
    assert call["close_fds"] is True
    assert status.output_dir == str(tmp_path / f"jobs/{status.job_id}")
    assert (tmp_path / f"jobs/{status.job_id}/events.jsonl").is_file()


def test_hunt_job_rejects_browser_paths_and_uses_candidate_resource(tmp_path: Path) -> None:
    # Given
    calls: list[dict[str, str | bool | tuple[str, ...]]] = []
    manager = JobManager(
        _runtime(tmp_path),
        resolver=_resolver(tmp_path),
        process_factory=_recording_factory(calls),
    )

    # When
    status = manager.start(HuntJobSpec(manifest_id=MANIFEST_ID, candidate_bank_id=BANK_ID))

    # Then
    argv = calls[0]["argv"]
    assert isinstance(argv, tuple)
    assert "--source" in argv
    assert str(tmp_path / "source") in argv
    assert str(tmp_path / "repo/artifacts/live/bank.json") in argv
    browser_controlled = "browser-controlled"
    assert not any(browser_controlled in item for item in argv)
    assert status.kind == "hunt"


def test_job_manager_allows_only_one_running_worker(tmp_path: Path) -> None:
    # Given
    manager = JobManager(
        _runtime(tmp_path),
        resolver=_resolver(tmp_path),
        process_factory=_recording_factory([], return_code=None),
    )
    _ = manager.start(SampleJobSpec(manifest_id=MANIFEST_ID))

    # When / Then
    with pytest.raises(JobBusyError, match="already running"):
        _ = manager.start(HuntJobSpec(manifest_id=MANIFEST_ID, candidate_bank_id=BANK_ID))


def test_job_status_readers_use_existing_job_ids_not_browser_paths(tmp_path: Path) -> None:
    # Given
    manager = JobManager(
        _runtime(tmp_path),
        resolver=_resolver(tmp_path),
        process_factory=_recording_factory([], return_code=0),
    )
    started = manager.start(SampleJobSpec(manifest_id=MANIFEST_ID))
    completed = manager.refresh()
    assert completed is not None

    # When
    statuses = manager.list_statuses()
    status = manager.get_status(started.job_id)
    events = manager.events(started.job_id)

    # Then
    assert tuple(item.job_id for item in statuses) == (started.job_id,)
    assert status.state == "succeeded"
    assert tuple(event.state for event in events) == ("running", "succeeded")
    with pytest.raises(UnknownJobError, match="unknown job"):
        _ = manager.get_status("../escape")


def _runtime(tmp_path: Path) -> JobRuntimeConfig:
    return JobRuntimeConfig(
        repo_root=tmp_path / "repo",
        jobs_root=tmp_path / "jobs",
        runtime_python=tmp_path / "runtime/bin/python",
        cache_dir=tmp_path / "cache",
        source_root=tmp_path / "source",
        python_executable="/venv/bin/python",
    )


def _resolver(tmp_path: Path) -> ResourceResolver:
    def resolve(resource_id: str) -> Path:
        match resource_id:
            case "res_manifest":
                return tmp_path / "repo/configs/hunt.json"
            case "res_bank":
                return tmp_path / "repo/artifacts/live/bank.json"
            case unreachable:
                message = f"unexpected resource id {unreachable}"
                raise AssertionError(message)

    return resolve


def _recording_factory(
    calls: list[dict[str, str | bool | tuple[str, ...]]],
    *,
    return_code: int | None = None,
) -> ProcessFactory:
    def factory(request: ProcessLaunch) -> RecordingProcess:
        calls.append(
            {
                "argv": request.argv,
                "cwd": str(request.cwd),
                "env_path": request.env["PATH"],
                "stdout": str(request.stdout),
                "stderr": str(request.stderr),
                "shell": request.shell,
                "close_fds": request.close_fds,
            }
        )
        return RecordingProcess(return_code)

    return factory
