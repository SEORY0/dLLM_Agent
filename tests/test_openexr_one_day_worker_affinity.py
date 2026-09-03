"""Worker CPU-affinity contracts for OpenEXR one-day resident model workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING

import pytest
from tools import selective_revision_worker as worker_module
from tools.selective_revision_payloads import request_hash, sha256_bytes, shutdown_request

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.runtime.revision_client import RevisionWorkerClient
from dllm_agent.runtime.revision_process import (
    CpuAffinityReader,
    CpuAffinitySetter,
    WorkerCpuAffinity,
    WorkerProcessFactory,
    worker_cpu_affinity_from_text,
)
from tests.revision_client_fakes import FakeFactory, FakePipe, FakeProcess

if TYPE_CHECKING:
    from pathlib import Path

    from tools.stable_diffcoder_payloads import RunnerConfig


@dataclass(slots=True)
class RecordingAffinity:
    """Mutable test probe recording affinity attempts."""

    actual: frozenset[int]
    sets: list[tuple[int, frozenset[int]]] = field(default_factory=list)

    def set(self, pid: int, cpus: frozenset[int]) -> None:
        self.sets.append((pid, cpus))

    def read(self, pid: int) -> frozenset[int]:
        del pid
        return self.actual


@dataclass(frozen=True, slots=True)
class FakeTokenizer:
    """Tokenizer placeholder bypassed by the fake sampler."""


@dataclass(frozen=True, slots=True)
class FakeModel:
    """Model placeholder bypassed by the fake sampler."""


class FakeSampler:
    """Sampler placeholder that keeps worker startup focused on load ordering."""

    def __init__(
        self,
        tokenizer: FakeTokenizer,
        model: worker_module.RealSlotModel,
        *,
        mask_token_id: int,
        block_size: int,
        prompt_context: str = "",
    ) -> None:
        del tokenizer, model, mask_token_id, block_size, prompt_context


def test_worker_startup_records_verified_cpu_affinity_before_ready(
    tmp_path: Path,
) -> None:
    # Given
    process = FakeProcess(stdout=FakePipe((_ready(), _shutdown("shutdown-000001"))), pid=45_678)
    affinity = RecordingAffinity(frozenset({4, 28}))
    factory = FakeFactory(process)
    client = _client(
        tmp_path,
        factory,
        cpu_affinity=WorkerCpuAffinity((4, 28)),
        affinity_reader=affinity.read,
        affinity_setter=affinity.set,
    )

    # When
    with client as opened:
        actual = opened.actual_cpu_affinity

    # Then
    command, _environment = factory.calls[0]
    assert command[0] == str(tmp_path / "python")
    assert command[1].endswith("tools/selective_revision_worker.py")
    assert command[-2:] == ("--cpu-affinity", "4,28")
    assert affinity.sets == [(45_678, frozenset({4, 28}))]
    assert actual == frozenset({4, 28})
    assert process.stdin.writes[0].startswith('{"operation":"shutdown"')


def test_worker_startup_fails_closed_when_actual_affinity_mismatches(
    tmp_path: Path,
) -> None:
    # Given
    process = FakeProcess(stdout=FakePipe((_ready(),)), pid=45_678)
    affinity = RecordingAffinity(frozenset({4}))
    client = _client(
        tmp_path,
        FakeFactory(process),
        cpu_affinity=WorkerCpuAffinity((4, 28)),
        affinity_reader=affinity.read,
        affinity_setter=affinity.set,
    )

    # When / Then
    with pytest.raises(HuntSetupError, match="CPU affinity mismatch"):
        _ = client.__enter__()
    assert process.terminated is True
    assert process.stdin.writes == []


@pytest.mark.parametrize("cpus", [(), (-1,), (True,), (4, 4)])
def test_worker_cpu_affinity_rejects_invalid_core_tuples(cpus: tuple[int, ...]) -> None:
    # Given / When / Then
    with pytest.raises(HuntSetupError, match="CPU affinity"):
        _ = WorkerCpuAffinity(cpus)


@pytest.mark.parametrize("value", ["", "-1", "4,4", "4,,28", "core"])
def test_worker_cpu_affinity_rejects_invalid_argv_values(value: str) -> None:
    # Given / When / Then
    with pytest.raises(HuntSetupError, match="CPU affinity"):
        _ = worker_cpu_affinity_from_text(value)


def test_worker_process_factory_starts_fixed_argv_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    calls: list[tuple[tuple[str, ...], bool | None]] = []

    class CapturingPopen:
        def __init__(
            self,
            command: tuple[str, ...],
            **kwargs: bool | dict[str, str] | int | str,
        ) -> None:
            shell = kwargs.get("shell")
            if not isinstance(shell, bool):
                shell = None
            calls.append((command, shell))

    monkeypatch.setattr(subprocess, "Popen", CapturingPopen)

    # When
    _ = WorkerProcessFactory().start(("python", "worker.py", "--device", "cuda:0"), {})

    # Then
    assert calls == [(("python", "worker.py", "--device", "cuda:0"), False)]


def test_worker_process_factory_rejects_shell_override() -> None:
    # Given / When / Then
    shell_override = True
    with pytest.raises(HuntSetupError, match="shell"):
        _ = WorkerProcessFactory(shell=shell_override)


def test_worker_child_pins_affinity_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    prompt_path = tmp_path / "prompt.md"
    _ = prompt_path.write_text("prompt context", encoding="utf-8")
    events: list[str] = []
    current_affinity: set[frozenset[int]] = set()

    def fake_pin(pid: int, affinity: WorkerCpuAffinity) -> frozenset[int]:
        assert pid == os.getpid()
        actual = affinity.expected_set
        current_affinity.add(actual)
        events.append("pin")
        return actual

    def fake_load_model(config: RunnerConfig) -> tuple[FakeTokenizer, FakeModel, str]:
        assert config.device == "cuda:0"
        assert current_affinity == {frozenset({4, 28})}
        events.append("load")
        return FakeTokenizer(), FakeModel(), "cuda:0"

    monkeypatch.setattr(worker_module, "pin_worker_process_to_cpu_affinity", fake_pin)
    monkeypatch.setattr(worker_module, "load_model", fake_load_model)
    monkeypatch.setattr(worker_module, "SelectiveRevisionSampler", FakeSampler)
    monkeypatch.setattr(sys, "stdin", StringIO(""))

    # When
    worker_module.main(
        cache_dir=tmp_path / "cache",
        prompt_path=prompt_path,
        prompt_sha256=sha256_bytes(prompt_path.read_bytes()),
        device="cuda:0",
        read_only_cache=False,
        cpu_affinity="4,28",
    )

    # Then
    assert events == ["pin", "load"]


def test_worker_client_lifecycle_remains_unpinned_when_affinity_is_omitted(
    tmp_path: Path,
) -> None:
    # Given
    process = FakeProcess(stdout=FakePipe((_ready(), _shutdown("shutdown-000001"))))
    affinity = RecordingAffinity(frozenset({0, 1}))
    client = _client(
        tmp_path,
        FakeFactory(process),
        affinity_reader=affinity.read,
        affinity_setter=affinity.set,
    )

    # When
    with client as opened:
        actual = opened.actual_cpu_affinity

    # Then
    assert actual is None
    assert affinity.sets == []
    assert process.stdin.writes[0].startswith('{"operation":"shutdown"')


def _client(
    tmp_path: Path,
    factory: FakeFactory,
    *,
    cpu_affinity: WorkerCpuAffinity | None = None,
    affinity_reader: CpuAffinityReader | None = None,
    affinity_setter: CpuAffinitySetter | None = None,
) -> RevisionWorkerClient:
    prompt_path = tmp_path / "prompt.md"
    _ = prompt_path.write_text("prompt context", encoding="utf-8")
    if affinity_reader is not None and affinity_setter is not None:
        return RevisionWorkerClient(
            runtime_python=tmp_path / "python",
            cache_dir=tmp_path / "cache",
            prompt_path=prompt_path,
            prompt_sha256=sha256_bytes(prompt_path.read_bytes()),
            device="cuda:0",
            timeout_seconds=1.0,
            max_line_bytes=4096,
            factory=factory,
            cpu_affinity=cpu_affinity,
            affinity_reader=affinity_reader,
            affinity_setter=affinity_setter,
        )
    return RevisionWorkerClient(
        runtime_python=tmp_path / "python",
        cache_dir=tmp_path / "cache",
        prompt_path=prompt_path,
        prompt_sha256=sha256_bytes(prompt_path.read_bytes()),
        device="cuda:0",
        timeout_seconds=1.0,
        max_line_bytes=4096,
        factory=factory,
        cpu_affinity=cpu_affinity,
    )


def _ready() -> str:
    return (
        json.dumps(
            {
                "kind": "ready",
                "schema_version": "dllm-selective-revision-ready/v1",
                "model_id": "ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
                "revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
                "remote_code_sha256": (
                    "298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104"
                ),
                "device": "cuda:0",
                "load_count": 1,
                "prompt_sha256": sha256_bytes(b"prompt context"),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _shutdown(request_id: str) -> str:
    return (
        json.dumps(
            {
                "kind": "shutdown",
                "request_id": request_id,
                "request_sha256": request_hash(shutdown_request(request_id)),
                "status": "ok",
            },
            sort_keys=True,
        )
        + "\n"
    )
