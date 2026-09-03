"""Process and argv helpers for the resident revision worker."""

# ruff: noqa: D102, D105

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from subprocess import TimeoutExpired
from typing import IO, TYPE_CHECKING, Final, NamedTuple, NoReturn, Protocol

from dllm_agent.core.errors import HuntSetupError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tools.stable_diffcoder_instrumentation import ModelLike, TensorLike

    from dllm_agent.experiments.openexr_one_day.models import ModelArm

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_WORKER_SCRIPT = _REPOSITORY_ROOT / "tools/selective_revision_worker.py"
_CPUS_ALLOWED_LIST_PREFIX: Final = "Cpus_allowed_list:"


class _EncodedInputLike(Protocol):
    input_ids: TensorLike


class TokenizerLike(Protocol):
    """Tokenizer surface consumed by Stable-DiffCoder generation."""

    eos_token_id: int | None

    def __len__(self) -> int: ...
    def __call__(self, text: str, *, return_tensors: str) -> _EncodedInputLike: ...
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str: ...
    def decode(
        self, token_ids: TensorLike | Sequence[int], *, skip_special_tokens: bool
    ) -> str: ...


class _CudaRuntime(Protocol):
    def is_available(self) -> bool: ...


class _TorchDType(Protocol):
    @property
    def is_floating_point(self) -> bool: ...


class TorchRuntime(NamedTuple):
    """Typed torch attributes loaded from its optional runtime module."""

    cuda: _CudaRuntime
    bfloat16: _TorchDType
    version: str
    set_num_threads: Callable[[int], None]
    set_num_interop_threads: Callable[[int], None]


class HubDownload(Protocol):
    """Pinned local-only Hugging Face download callable."""

    def __call__(
        self,
        repository: str,
        filename: str,
        *,
        revision: str,
        cache_dir: str | None,
        local_files_only: bool,
    ) -> str: ...


class _TokenizerFactory(Protocol):
    def from_pretrained(
        self,
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        cache_dir: str | None,
        local_files_only: bool,
    ) -> TokenizerLike: ...


class _ModelFactory(Protocol):
    def from_pretrained(  # noqa: PLR0913
        self,
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        cache_dir: str | None,
        local_files_only: bool,
        dtype: _TorchDType,
        use_safetensors: bool,
        low_cpu_mem_usage: bool,
        weights_only: bool,
        device_map: dict[str, str] | None,
    ) -> ModelLike: ...


class TransformersImports(NamedTuple):
    """Typed lazy exports loaded from Transformers."""

    tokenizer_factory: _TokenizerFactory
    model_factory: _ModelFactory
    version: str


class BankGenerationCost(NamedTuple):
    """Physical or logical model work spent on bank generation."""

    initial_requests: int
    nfe: int
    forward_calls: int
    gpu_ms: int


class SelectedLineage(NamedTuple):
    """Stable campaign lineage bound to one selected raw parent."""

    particle_id: str
    source_seed: int
    requested_seed: int
    steps: int


class ArmBankCostClaim(NamedTuple):
    """Task 5 adjustment after pipeline-visible selected initial costs."""

    claim_id: str
    arm: ModelArm
    pipeline_visible_selected_cost: BankGenerationCost
    task5_adjustment: BankGenerationCost
    logical_raw_bank_cost: BankGenerationCost


class BankAccounting(NamedTuple):
    """One physical cost plus one logical claim per model arm."""

    physical_campaign_cost: BankGenerationCost
    arm_claims: tuple[ArmBankCostClaim, ...]


def load_torch_runtime() -> TorchRuntime:
    """Load and validate torch's lazy runtime attributes."""
    module = import_module("torch")
    cuda: _CudaRuntime | None = getattr(module, "cuda", None)
    dtype: _TorchDType | None = getattr(module, "bfloat16", None)
    version: str | None = getattr(module, "__version__", None)
    set_threads: Callable[[int], None] | None = getattr(module, "set_num_threads", None)
    set_interop: Callable[[int], None] | None = getattr(module, "set_num_interop_threads", None)
    if (
        cuda is None
        or not callable(cuda.is_available)
        or dtype is None
        or not isinstance(version, str)
        or set_threads is None
        or not callable(set_threads)
        or set_interop is None
        or not callable(set_interop)
    ):
        message = "torch runtime does not expose its required surface"
        raise TypeError(message)
    return TorchRuntime(cuda, dtype, version, set_threads, set_interop)


def load_hub_download() -> HubDownload:
    """Load and validate the Hugging Face Hub lazy download export."""
    module = import_module("huggingface_hub")
    download: HubDownload | None = getattr(module, "hf_hub_download", None)
    if download is None or not callable(download):
        message = "huggingface_hub runtime does not expose hf_hub_download"
        raise TypeError(message)
    return download


def load_transformers_imports() -> TransformersImports:
    """Load and validate the two Transformers lazy factory exports."""
    module = import_module("transformers")
    tokenizer: _TokenizerFactory | None = getattr(module, "AutoTokenizer", None)
    model: _ModelFactory | None = getattr(module, "AutoModelForCausalLM", None)
    version: str | None = getattr(module, "__version__", None)
    if (
        tokenizer is None
        or model is None
        or not callable(tokenizer.from_pretrained)
        or not callable(model.from_pretrained)
        or not isinstance(version, str)
    ):
        message = "transformers runtime does not expose its required surface"
        raise TypeError(message)
    return TransformersImports(tokenizer, model, version)


class WorkerProcess(Protocol):
    """Process surface used by the revision client."""

    @property
    def stdin(self) -> IO[str] | None:
        """Writable stdin pipe."""
        ...

    @property
    def stdout(self) -> IO[str] | None:
        """Readable stdout pipe."""
        ...

    @property
    def stderr(self) -> IO[str] | None:
        """Readable stderr pipe."""
        ...

    @property
    def pid(self) -> int:
        """Positive operating-system process identifier."""
        ...

    def poll(self) -> int | None:
        """Return None only while the process is alive."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process exit."""
        ...

    def terminate(self) -> None:
        """Terminate the process."""
        ...

    def kill(self) -> None:
        """Kill a process that ignored graceful termination."""
        ...


class CpuAffinityReader(Protocol):
    """Read a process CPU affinity mask."""

    def __call__(self, pid: int) -> frozenset[int]:
        """Return the process CPU set."""
        ...


class CpuAffinitySetter(Protocol):
    """Set a process CPU affinity mask."""

    def __call__(self, pid: int, cpus: frozenset[int]) -> None:
        """Pin the process to exactly the provided CPU set."""
        ...


def _missing_sched_setaffinity(pid: int, cpus: frozenset[int]) -> None:
    del pid, cpus
    _fail("CPU affinity is unavailable on this platform")


def _missing_sched_getaffinity(pid: int) -> frozenset[int]:
    del pid
    _fail("CPU affinity is unavailable on this platform")


@dataclass(frozen=True, slots=True)
class WorkerCpuAffinity:
    """Validated CPU tuple assigned to one resident worker."""

    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cpus:
            _fail("worker CPU affinity requires at least one CPU")
        seen: set[int] = set()
        for cpu in self.cpus:
            if isinstance(cpu, bool) or cpu < 0:
                _fail("worker CPU affinity values must be non-negative integers")
            if cpu in seen:
                _fail("worker CPU affinity values must be unique")
            seen.add(cpu)

    @property
    def expected_set(self) -> frozenset[int]:
        """Return the exact process affinity mask expected after pinning."""
        return frozenset(self.cpus)


def worker_cpu_affinity_for_core(core_id: int, sibling_core_id: int) -> WorkerCpuAffinity:
    """Build the physical-core/sibling affinity tuple from a resource-map assignment."""
    return WorkerCpuAffinity((core_id, sibling_core_id))


def worker_cpu_affinity_from_text(value: str) -> WorkerCpuAffinity:
    """Parse the fixed worker argv CPU tuple form."""
    if value.strip() != value or not value:
        _fail("worker CPU affinity argv must be a comma-separated CPU tuple")
    parsed: list[int] = []
    for item in value.split(","):
        if not item:
            _fail("worker CPU affinity argv must not contain empty CPU ids")
        try:
            parsed.append(int(item))
        except ValueError as error:
            _fail("worker CPU affinity argv values must be integers", error)
    return WorkerCpuAffinity(tuple(parsed))


def worker_cpu_affinity_to_text(affinity: WorkerCpuAffinity) -> str:
    """Serialize the fixed worker argv CPU tuple form."""
    return ",".join(str(cpu) for cpu in affinity.cpus)


def read_process_cpu_affinity(pid: int) -> frozenset[int]:
    """Read a process affinity mask from procfs, falling back to sched_getaffinity."""
    status_path = Path(f"/proc/{pid}/status")
    try:
        status = status_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _read_sched_affinity(pid)
    except OSError as error:
        _fail("failed to read revision worker CPU affinity", error)
    for line in status.splitlines():
        if line.startswith(_CPUS_ALLOWED_LIST_PREFIX):
            return _parse_cpu_list(line.removeprefix(_CPUS_ALLOWED_LIST_PREFIX).strip())
    return _read_sched_affinity(pid)


def set_process_cpu_affinity(pid: int, cpus: frozenset[int]) -> None:
    """Set process affinity using the platform scheduler API."""
    setter = getattr(os, "sched_setaffinity", _missing_sched_setaffinity)
    try:
        _ = setter(pid, cpus)
    except OSError as error:
        _fail("failed to set revision worker CPU affinity", error)


def _read_sched_affinity(pid: int) -> frozenset[int]:
    getter = getattr(os, "sched_getaffinity", _missing_sched_getaffinity)
    try:
        cpus = getter(pid)
    except OSError as error:
        _fail("failed to read revision worker CPU affinity", error)
    return WorkerCpuAffinity(tuple(cpus)).expected_set


def pin_worker_process_to_cpu_affinity(
    pid: int,
    affinity: WorkerCpuAffinity,
    *,
    reader: CpuAffinityReader = read_process_cpu_affinity,
    setter: CpuAffinitySetter = set_process_cpu_affinity,
) -> frozenset[int]:
    """Set and verify worker CPU affinity before worker readiness is trusted."""
    expected = affinity.expected_set
    setter(pid, expected)
    actual = reader(pid)
    if actual != expected:
        message = (
            f"revision worker CPU affinity mismatch: expected {sorted(expected)}, "
            f"actual {sorted(actual)}"
        )
        _fail(message)
    return actual


def _parse_cpu_list(cpu_list: str) -> frozenset[int]:
    parsed: list[int] = []
    for raw_item in cpu_list.split(","):
        item = raw_item.strip()
        if not item:
            continue
        bounds = item.split("-", maxsplit=1)
        if len(bounds) == 1:
            parsed.append(_parse_cpu_id(bounds[0]))
            continue
        start = _parse_cpu_id(bounds[0])
        stop = _parse_cpu_id(bounds[1])
        if stop < start:
            _fail("revision worker CPU affinity status is invalid")
        parsed.extend(range(start, stop + 1))
    return WorkerCpuAffinity(tuple(parsed)).expected_set


def _parse_cpu_id(value: str) -> int:
    try:
        cpu = int(value)
    except ValueError as error:
        _fail("revision worker CPU affinity status is invalid", error)
    if cpu < 0:
        _fail("revision worker CPU affinity status is invalid")
    return cpu


def _fail(detail: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise HuntSetupError(detail)
    raise HuntSetupError(detail) from cause


class ProcessFactory(Protocol):
    """Factory for starting a worker process."""

    def start(self, command: tuple[str, ...], environment: dict[str, str]) -> WorkerProcess:
        """Start the fixed worker command."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerProcessFactory:
    """Production subprocess factory."""

    shell: bool = False

    def __post_init__(self) -> None:
        if self.shell:
            _fail("revision worker process factory does not support shell=True")

    def start(self, command: tuple[str, ...], environment: dict[str, str]) -> WorkerProcess:
        """Start the worker without shell interpolation."""
        try:
            return subprocess.Popen(  # noqa: S603 - fixed argv, shell disabled.
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=environment,
                shell=False,
            )
        except OSError as error:
            raise HuntSetupError(error.strerror or "failed to start revision worker") from error


def reap_worker_process(
    process: WorkerProcess,
    timeout_seconds: float,
    *,
    terminate: bool,
    timeout_detail: str,
) -> None:
    """Close worker pipes and wait, escalating to kill on timeout."""
    if terminate and process.poll() is None:
        process.terminate()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    try:
        _ = process.wait(timeout=timeout_seconds)
    except TimeoutExpired as error:
        if process.poll() is None:
            process.terminate()
        process.kill()
        try:
            _ = process.wait(timeout=timeout_seconds)
        except TimeoutExpired as reap_error:
            raise HuntSetupError(timeout_detail) from reap_error
        raise HuntSetupError(timeout_detail) from error


def revision_worker_command(  # noqa: PLR0913 - fixed historical argv contract plus affinity.
    runtime_python: Path,
    cache_dir: Path,
    device: str,
    prompt_path: Path,
    prompt_sha256: str,
    *,
    cpu_affinity: WorkerCpuAffinity | None = None,
) -> tuple[str, ...]:
    """Return the fixed resident worker argv used by client and manifests."""
    command = (
        str(runtime_python),
        str(_WORKER_SCRIPT),
        "--cache-dir",
        str(cache_dir),
        "--prompt-path",
        str(prompt_path),
        "--prompt-sha256",
        prompt_sha256,
        "--device",
        device,
    )
    if cpu_affinity is None:
        return command
    return (*command, "--cpu-affinity", worker_cpu_affinity_to_text(cpu_affinity))
