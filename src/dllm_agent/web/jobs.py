"""Typed subprocess job manager for the WebUI control plane."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dllm_agent.web.contracts import (
    HuntJobSpec,
    JobEvent,
    JobRuntimeConfig,
    JobStatus,
    SampleJobSpec,
)
from dllm_agent.web.processes import (
    ManagedProcess,
    ProcessFactory,
    ProcessLaunch,
    bounded_log,
    default_process_factory,
)

JobSpec = SampleJobSpec | HuntJobSpec
ResourceResolver = Callable[[str], Path]


class JobBusyError(Exception):
    """Raised when the single worker slot is already occupied."""

    def __init__(self) -> None:
        """Create a busy-worker error."""
        super().__init__("a job is already running")


class UnknownJobError(Exception):
    """Raised when a browser job id is not an existing job directory."""

    def __init__(self) -> None:
        """Create an unknown-job error."""
        Exception.__init__(self, "unknown job")


class JobManager:
    """Launch typed sample/hunt jobs with fixed argv and durable journals."""

    def __init__(
        self,
        runtime: JobRuntimeConfig,
        *,
        resolver: ResourceResolver,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        """Create a manager from trusted runtime settings and a resource resolver."""
        self._runtime: JobRuntimeConfig = runtime
        self._resolve: ResourceResolver = resolver
        self._process_factory: ProcessFactory = process_factory or default_process_factory
        self._active: tuple[str, ManagedProcess] | None = None

    def start(self, spec: JobSpec) -> JobStatus:
        """Start one typed job if the single worker is free."""
        _ = self.refresh()
        if self._active is not None:
            raise JobBusyError
        job_id = uuid4().hex
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        argv = self._argv(spec, job_id)
        status = JobStatus(
            job_id=job_id,
            kind=spec.kind,
            state="running",
            output_dir=self._relative_output_dir(job_id),
        )
        self._write_status(job_dir, status)
        self._append_event(job_dir, JobEvent(job_id=job_id, state="running", message="started"))
        process = self._process_factory(
            ProcessLaunch(
                argv,
                cwd=self._runtime.repo_root,
                env=self._minimal_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
            )
        )
        self._active = (job_id, process)
        return status

    def refresh(self) -> JobStatus | None:
        """Persist completion state for the active job, if any."""
        if self._active is None:
            return None
        job_id, process = self._active
        exit_code = process.poll()
        if exit_code is None:
            return None
        stdout, stderr = process.communicate()
        state = "succeeded" if exit_code == 0 else "failed"
        status = JobStatus(
            job_id=job_id,
            kind=self._read_kind(job_id),
            state=state,
            output_dir=self._relative_output_dir(job_id),
            exit_code=exit_code,
            log_tail=bounded_log(stdout + stderr),
        )
        job_dir = self._job_dir(job_id)
        self._write_status(job_dir, status)
        self._append_event(job_dir, JobEvent(job_id=job_id, state=state, message="finished"))
        self._active = None
        return status

    def list_statuses(self) -> tuple[JobStatus, ...]:
        """List durable job statuses from existing server-owned job dirs."""
        root = self._jobs_root()
        if not root.is_dir():
            return ()
        statuses: list[JobStatus] = []
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            status_path = path / "status.json"
            if path.is_dir() and status_path.is_file():
                statuses.append(JobStatus.model_validate_json(status_path.read_bytes()))
        return tuple(statuses)

    def get_status(self, job_id: str) -> JobStatus:
        """Load one durable job status by opaque job id."""
        return JobStatus.model_validate_json(self._job_status_path(job_id).read_bytes())

    def events(self, job_id: str) -> tuple[JobEvent, ...]:
        """Load one durable job event stream by opaque job id."""
        event_path = self._job_dir_from_existing_id(job_id) / "events.jsonl"
        if not event_path.is_file():
            raise UnknownJobError
        return tuple(
            JobEvent.model_validate_json(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if line
        )

    def _argv(self, spec: JobSpec, job_id: str) -> tuple[str, ...]:
        match spec:
            case SampleJobSpec():
                return self._sample_argv(spec, job_id)
            case HuntJobSpec():
                return self._hunt_argv(spec, job_id)

    def _sample_argv(self, spec: SampleJobSpec, job_id: str) -> tuple[str, ...]:
        output = self._job_dir(job_id) / "candidate-bank.json"
        return (
            self._runtime.python_executable,
            "-m",
            "dllm_agent",
            "sample",
            str(self._resolve(spec.manifest_id)),
            "--runtime-python",
            str(self._runtime.runtime_python),
            "--cache-dir",
            str(self._runtime.cache_dir),
            "--output",
            str(output),
            "--count",
            str(spec.count),
            "--seed",
            str(spec.seed),
        )

    def _hunt_argv(self, spec: HuntJobSpec, job_id: str) -> tuple[str, ...]:
        job_dir = self._job_dir(job_id)
        return (
            self._runtime.python_executable,
            "-m",
            "dllm_agent",
            "hunt",
            str(self._resolve(spec.manifest_id)),
            "--candidate-bank",
            str(self._resolve(spec.candidate_bank_id)),
            "--source",
            str(self._runtime.source_root),
            "--artifact-dir",
            str(job_dir / "artifacts"),
            "--output",
            str(job_dir / "hunt-report.json"),
        )

    def _minimal_env(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(self._runtime.repo_root / "src"),
            "PYTHONUNBUFFERED": "1",
        }

    def _job_dir(self, job_id: str) -> Path:
        return self._jobs_root() / job_id

    def _relative_output_dir(self, job_id: str) -> str:
        return str(self._job_dir(job_id))

    def _jobs_root(self) -> Path:
        return self._runtime.jobs_root.resolve()

    def _job_status_path(self, job_id: str) -> Path:
        status_path = self._job_dir_from_existing_id(job_id) / "status.json"
        if not status_path.is_file():
            raise UnknownJobError
        return status_path

    def _job_dir_from_existing_id(self, job_id: str) -> Path:
        job_dir = (self._jobs_root() / job_id).resolve()
        if job_dir.parent != self._jobs_root() or not job_dir.is_dir():
            raise UnknownJobError
        return job_dir

    def _write_status(self, job_dir: Path, status: JobStatus) -> None:
        _ = (job_dir / "status.json").write_text(status.model_dump_json(indent=2) + "\n")

    def _append_event(self, job_dir: Path, event: JobEvent) -> None:
        with (job_dir / "events.jsonl").open("a", encoding="utf-8") as file:
            _ = file.write(event.model_dump_json() + "\n")

    def _read_kind(self, job_id: str) -> Literal["sample", "hunt"]:
        path = self._job_dir(job_id) / "status.json"
        return JobStatus.model_validate_json(path.read_bytes()).kind
