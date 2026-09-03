from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from types import ModuleType
from typing import TYPE_CHECKING, Self, override

from tools.selective_revision_payloads import ready_record
from tools.stable_diffcoder_payloads import GenerationConfig, JsonValue, ModelProvenance

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.experiments.openexr_one_day.models import GpuIndex
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    GIBIBYTE,
    ResidentWorkerBinding,
    ResourceSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tools.runtime.selective_revision_json import ReadyRecord

    from dllm_agent.experiments.openexr_one_day.resident_pool import ResourceTelemetryProbe
    from dllm_agent.revision.controller_models import (
        InitialParticleRequest,
        RevisionRequest,
        RevisionResult,
    )
    from dllm_agent.runtime.revision_process import WorkerProcess


class FakePipe(StringIO):
    def __init__(self, lines: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._lines: list[str] = list(lines)
        self.writes: list[str] = []
        self.was_closed: bool = False

    @override
    def readline(self, size: int = -1) -> str:
        del size
        return "" if not self._lines else self._lines.pop(0)

    @override
    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    @override
    def close(self) -> None:
        self.was_closed = True


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records process lifecycle.
class FakeProcess:
    stdout: FakePipe
    stdin: FakePipe = field(default_factory=FakePipe)
    stderr: FakePipe = field(default_factory=FakePipe)
    pid: int = 12_345
    returncode: int | None = None
    terminated: bool = False
    killed: bool = False
    wait_calls: int = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        self.returncode = 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


@dataclass(frozen=True, slots=True)
class FakeFactory:
    process: FakeProcess
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = field(default_factory=list)

    def start(self, command: tuple[str, ...], environment: dict[str, str]) -> WorkerProcess:
        self.calls.append((command, environment))
        return self.process


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records resident lifecycle.
class FakeResidentWorker:
    device: str
    cache_dir: Path
    events: list[str]
    read_only_cache: bool = True
    ready_device: str | None = None
    fail_start: bool = False
    ready: ReadyRecord | None = None
    process_id: int = 0
    pid_after_start: int | None = None
    alive: bool = True
    start_count: int = 0
    close_count: int = 0

    def __enter__(self) -> Self:
        self.start_count += 1
        self.events.append(f"start:{self.device}")
        default_pid = 10_000 + int(self.device.removeprefix("cuda:"))
        self.process_id = default_pid if self.pid_after_start is None else self.pid_after_start
        if self.fail_start:
            raise HuntSetupError("synthetic startup failure")  # noqa: EM101, TRY003
        self.ready = ready_record(self.ready_device or self.device, 1, "a" * 64)
        return self

    @property
    def process_id_if_started(self) -> int | None:
        return self.process_id or None

    @property
    def last_process_id(self) -> int | None:
        return self.process_id_if_started

    @property
    def is_alive(self) -> bool:
        return self.alive

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close:{self.device}")

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        del request
        raise AssertionError

    def revise(self, request: RevisionRequest) -> RevisionResult:
        del request
        raise AssertionError


@dataclass(frozen=True, slots=True)
class ReadOnlyStatVfs:
    f_flag: int


def resource_snapshot(host: float, gpu: float, swap: float, rss: float) -> ResourceSnapshot:
    scale = GIBIBYTE
    return ResourceSnapshot(
        int(host * scale), int(swap * scale), int(gpu * scale), int(rss * scale)
    )


def resident_bindings(
    workers: list[FakeResidentWorker],
) -> tuple[ResidentWorkerBinding, ...]:
    return tuple(
        ResidentWorkerBinding(GpuIndex(index), worker) for index, worker in enumerate(workers)
    )


@dataclass(frozen=True, slots=True)
class FakeTelemetryProbe:
    before: ResourceSnapshot
    after: ResourceSnapshot
    calls: list[tuple[GpuIndex, int | None]] = field(default_factory=list)

    def capture(self, gpu_index: GpuIndex, process_id: int | None) -> ResourceSnapshot:
        self.calls.append((gpu_index, process_id))
        return self.before if process_id is None else self.after


@dataclass(frozen=True, slots=True)
class ScriptedLoadMonitor:
    in_load_samples: tuple[ResourceSnapshot, ...]

    def load(
        self,
        binding: ResidentWorkerBinding,
        telemetry_probe: ResourceTelemetryProbe,
        record: Callable[[ResourceSnapshot], None],
    ) -> None:
        del telemetry_probe
        for sample in self.in_load_samples:
            record(sample)
        _ = binding.worker.__enter__()


class FakeCuda:
    def is_available(self) -> bool:
        return True


class FakeTorch(ModuleType):
    cuda: FakeCuda = FakeCuda()
    bfloat16: str = "bfloat16"
    __version__: str = "fake-torch"

    def set_num_threads(self, threads: int) -> None:
        assert threads == 1

    def set_num_interop_threads(self, threads: int) -> None:
        assert threads == 1


class LazyHub(ModuleType):
    def __init__(self, remote_code: Path) -> None:
        super().__init__("huggingface_hub")
        self._remote_code: Path = remote_code

    @override
    def __getattr__(self, name: str) -> Callable[..., str]:
        if name != "hf_hub_download":
            raise AttributeError(name)
        return self._download

    def _download(self, *args: str, **kwargs: str | bool | None) -> str:
        del args, kwargs
        return str(self._remote_code)


class FakeTokenizer:
    pass


class FakeModel:
    evaluated: bool = False
    moved_to: str | None = None

    def to(self, device: str) -> None:
        self.moved_to = device

    def eval(self) -> None:
        self.evaluated = True


class _FakeTokenizerFactory:
    @staticmethod
    def from_pretrained(*args: str, **kwargs: str | bool | None) -> FakeTokenizer:
        del args, kwargs
        return FakeTokenizer()


class _FakeModelFactory:
    @staticmethod
    def from_pretrained(*_args: str, **kwargs: str | bool | dict[str, str]) -> FakeModel:
        if kwargs.get("device_map") is not None:
            message = "accelerate is required for device_map"
            raise ValueError(message)
        return FakeModel()


class LazyTransformers(ModuleType):
    __version__: str = "fake-transformers"

    @override
    def __getattr__(self, name: str) -> type[_FakeTokenizerFactory | _FakeModelFactory] | str:
        if name == "AutoTokenizer":
            return _FakeTokenizerFactory
        if name == "AutoModelForCausalLM":
            return _FakeModelFactory
        if name == "__version__":
            return self.__version__
        raise AttributeError(name)


def stable_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
        revision="10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        remote_code_sha256="a" * 64,
        torch_version="2.8.0",
        transformers_version="4.55.0",
        device="cpu",
        generation_config=GenerationConfig(
            steps=8, gen_length=64, block_length=16, temperature=0.2
        ),
    )


def legacy_candidate_bank_v1() -> JsonValue:
    return {
        "schema_version": "dllm-candidate-bank/v1",
        "prompt_sha256": "0" * 64,
        "model": {
            "model_id": "ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
            "revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
            "remote_code_sha256": "a" * 64,
            "python_version": "3.12.13",
            "torch_version": "2.8.0",
            "transformers_version": "4.55.0",
            "device": "cpu",
            "generation": {"steps": 8, "gen_length": 64, "block_length": 16, "temperature": 0.2},
        },
        "candidates": [
            _legacy_candidate("candidate-01", 0, "b" * 64, "c" * 64),
            _legacy_candidate("candidate-02", 1, "d" * 64, "e" * 64),
        ],
    }


def _legacy_candidate(identifier: str, result: int, source_hash: str, state_hash: str) -> JsonValue:
    source = f"int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) {{return {result};}}"
    return {
        "candidate_id": identifier,
        "source": source,
        "source_sha256": source_hash,
        "seed": result + 1,
        "nfe": 1,
        "steps": [
            {
                "step_index": 1,
                "masked_tokens_before": 4,
                "committed_tokens": 1,
                "mean_confidence": 0.5,
                "state_sha256": state_hash,
            }
        ],
    }
