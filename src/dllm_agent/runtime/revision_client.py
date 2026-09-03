"""Persistent resident selective-revision worker client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Self

from tools.runtime.selective_revision_json import (
    InitializeRequest,
    ReadyRecord,
    ReviseRequest,
    SamplerResponse,
    SelectorState,
    ShutdownResponse,
    WireProtocolError,
    WorkerRequest,
)
from tools.selective_revision_payloads import (
    initialize_worker_request,
    parse_ready_line,
    parse_response_line,
    request_hash,
    request_to_line,
    response_with_hash,
    revise_worker_request,
    sha256_token_ids,
    shutdown_request,
)

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import read_bytes, sha256_bytes
from dllm_agent.runtime.revision_checks import (
    read_line_with_timeout,
    to_revision_result,
    validate_controller_revision,
    validate_ready,
    validate_revision_request,
    validate_sampler_response,
)
from dllm_agent.runtime.revision_process import (
    CpuAffinityReader,
    CpuAffinitySetter,
    ProcessFactory,
    WorkerCpuAffinity,
    WorkerProcess,
    WorkerProcessFactory,
    pin_worker_process_to_cpu_affinity,
    read_process_cpu_affinity,
    reap_worker_process,
    revision_worker_command,
    set_process_cpu_affinity,
)

if TYPE_CHECKING:
    from types import TracebackType

    from dllm_agent.revision.controller_models import (
        InitialParticleRequest,
        RevisionRequest,
        RevisionResult,
    )

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


# Client mutation is the documented worker lifecycle state.
@dataclass(slots=True)
class RevisionWorkerClient:  # noqa: D101
    runtime_python: Path
    cache_dir: Path
    prompt_path: Path
    prompt_sha256: str
    device: str
    timeout_seconds: float
    read_only_cache: bool = field(default=False, kw_only=True)
    max_line_bytes: int = 1_000_000
    factory: ProcessFactory = field(default_factory=WorkerProcessFactory)
    cpu_affinity: WorkerCpuAffinity | None = field(default=None, kw_only=True)
    affinity_reader: CpuAffinityReader = field(
        default=read_process_cpu_affinity, repr=False, kw_only=True
    )
    affinity_setter: CpuAffinitySetter = field(
        default=set_process_cpu_affinity, repr=False, kw_only=True
    )
    _process: WorkerProcess | None = None
    _last_process_id: int | None = field(default=None, init=False)
    _actual_cpu_affinity: frozenset[int] | None = field(default=None, init=False)
    _next_id: int = 1
    ready: ReadyRecord | None = None

    @property
    def process_id(self) -> int:  # noqa: D102
        if (process_id := self.process_id_if_started) is None:
            _fail("revision worker must expose a positive PID")
        return process_id

    @property
    def process_id_if_started(self) -> int | None:  # noqa: D102
        return None if (process := self._process) is None or process.pid <= 0 else process.pid

    @property
    def last_process_id(self) -> int | None:  # noqa: D102
        return self._last_process_id

    @property
    def actual_cpu_affinity(self) -> frozenset[int] | None:  # noqa: D102
        return self._actual_cpu_affinity

    @property
    def is_alive(self) -> bool:  # noqa: D102
        return (process := self._process) is not None and process.pid > 0 and process.poll() is None

    def __enter__(self) -> Self:  # noqa: D105
        prompt = read_bytes(self.prompt_path, "revision worker prompt")
        if sha256_bytes(prompt) != self.prompt_sha256:
            _fail("revision worker prompt sha256 mismatch")
        process = self.factory.start(self.command(), self.environment())
        self._process = process
        self._last_process_id = process.pid if process.pid > 0 else None
        try:
            if self.cpu_affinity is not None:
                self._actual_cpu_affinity = pin_worker_process_to_cpu_affinity(
                    process.pid,
                    self.cpu_affinity,
                    reader=self.affinity_reader,
                    setter=self.affinity_setter,
                )
            self.ready = parse_ready_line(self._read_line(process))
            validate_ready(self.ready)
            if self.ready.prompt_sha256 != self.prompt_sha256:
                _fail("revision worker ready prompt sha256 mismatch")
        except WireProtocolError as error:
            self._discard_started_process(process)
            raise HuntSetupError(str(error)) from error
        except HuntSetupError:
            self._discard_started_process(process)
            raise
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:  # noqa: D102
        if (process := self._process) is None:
            return
        terminate = False
        try:
            if process.poll() is None:
                request_id = self._request_id("shutdown")
                request = shutdown_request(request_id)
                self._write(process, request_to_line(request))
                response = parse_response_line(self._read_line(process))
                if not isinstance(response, ShutdownResponse) or response.request_id != request_id:
                    _fail("revision worker returned mismatched shutdown response")
                if response.request_sha256 != request_hash(request):
                    _fail("revision worker returned mismatched shutdown request hash")
        except (HuntSetupError, WireProtocolError):
            terminate = True
        finally:
            try:
                reap_worker_process(
                    process,
                    self.timeout_seconds,
                    terminate=terminate,
                    timeout_detail="revision worker shutdown timeout",
                )
            finally:
                self._process = None
                self._actual_cpu_affinity = None

    def initial(self, request: InitialParticleRequest) -> RevisionResult:  # noqa: D102
        payload = InitializeRequest(
            request_id=self._request_id("init"),
            particle_id=request.particle_id,
            seed=request.seed,
            steps=request.steps,
        )
        response = self._round_trip(initialize_worker_request(payload))
        return to_revision_result(response)

    def revise(self, request: RevisionRequest) -> RevisionResult:  # noqa: D102
        validate_controller_revision(request)
        payload = ReviseRequest(
            request_id=self._request_id("revise"),
            particle_id=request.particle_id,
            parent=SelectorState(
                token_ids=request.parent_token_ids,
                slot_positions=request.slot_positions,
                selectors=request.parent_selectors,
                token_state_sha256=sha256_token_ids(request.parent_token_ids),
            ),
            parent_token_sha256=sha256_token_ids(request.parent_token_ids),
            remask_bitmap=request.remask_bitmap,
            frozen_bitmap=request.frozen_bitmap,
            revision_steps=request.revision_steps,
            seed=request.seed,
            rng_state_sha256=request.rng_state_sha256,
            feedback_digest=request.feedback_digest,
            mode=request.mode,
        )
        validate_revision_request(payload)
        response = self._round_trip(revise_worker_request(payload))
        return to_revision_result(response)

    def _round_trip(self, request: WorkerRequest) -> SamplerResponse:
        process = self._required_process()
        line = request_to_line(request)
        expected_hash = request_hash(request)
        self._write(process, line)
        try:
            response = parse_response_line(self._read_line(process))
        except WireProtocolError as error:
            raise HuntSetupError(str(error)) from error
        if not isinstance(response, SamplerResponse):
            _fail("revision worker returned a non-sampler response")
        if request.payload is None:
            _fail("revision worker request payload missing")
        if response.request_id != request.payload.request_id:
            _fail("revision worker returned mismatched request id")
        if response.request_sha256 != expected_hash:
            _fail("revision worker returned mismatched request hash")
        if response_with_hash(response).response_hash != response.response_hash:
            _fail("revision worker returned mismatched response hash")
        validate_sampler_response(response, request.payload)
        return response

    def _required_process(self) -> WorkerProcess:
        if self._process is None:
            _fail("revision worker is not running")
        return self._process

    def _discard_started_process(self, process: WorkerProcess) -> None:
        try:
            reap_worker_process(
                process,
                self.timeout_seconds,
                terminate=True,
                timeout_detail="revision worker startup cleanup timeout",
            )
        finally:
            self._process = None

    def _request_id(self, prefix: str) -> str:
        request_id = f"{prefix}-{self._next_id:06d}"
        self._next_id += 1
        return request_id

    def command(self) -> tuple[str, ...]:  # noqa: D102
        command = revision_worker_command(
            self.runtime_python,
            self.cache_dir,
            self.device,
            self.prompt_path,
            self.prompt_sha256,
            cpu_affinity=self.cpu_affinity,
        )
        return (*command, "--read-only-cache") if self.read_only_cache else command

    def environment(self) -> dict[str, str]:  # noqa: D102
        return {
            "HF_HOME": str(self.cache_dir),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(self.cache_dir / "modules"),
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": f"{_REPOSITORY_ROOT / 'src'}:{_REPOSITORY_ROOT}",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }

    def _read_line(self, process: WorkerProcess) -> str:
        if (stdout := process.stdout) is None:
            _fail("revision worker stdout unavailable")
        try:
            line = read_line_with_timeout(stdout, self.timeout_seconds)
        except HuntSetupError:
            process.terminate()
            raise
        if line == "":
            _fail("revision worker EOF")
        if len(line.encode()) > self.max_line_bytes:
            _fail("revision worker line length exceeded")
        return line

    def _write(self, process: WorkerProcess, line: str) -> None:
        if len(line.encode()) > self.max_line_bytes:
            _fail("revision worker request line length exceeded")
        if (stdin := process.stdin) is None:
            _fail("revision worker stdin unavailable")
        _ = stdin.write(line)
        stdin.flush()


def _fail(detail: str) -> NoReturn:
    raise HuntSetupError(detail)
