"""Resident revision client validation and timeout tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from io import StringIO
from subprocess import TimeoutExpired
from typing import TYPE_CHECKING, override

import pytest
from tools.selective_revision_payloads import (
    request_hash,
    sha256_bytes,
    shutdown_request,
)
from tools.selective_revision_worker import WorkerCacheError, prepare_worker_cache

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.revision.controller_models import RevisionRequest
from dllm_agent.runtime.revision_client import RevisionWorkerClient
from dllm_agent.targets.openexr.particle import FIELD_ORDER, FieldName

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.runtime.revision_process import WorkerProcess


class SlowPipe(StringIO):
    """Pipe that can delay reads to exercise timeout cleanup."""

    def __init__(self, lines: tuple[str, ...], *, delay_seconds: float = 0.0) -> None:
        super().__init__()
        self._lines: list[str] = list(lines)
        self._delay_seconds: float = delay_seconds
        self.writes: list[str] = []

    @override
    def readline(self, size: int = -1) -> str:
        del size
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        if not self._lines:
            return ""
        return self._lines.pop(0)

    @override
    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    @override
    def flush(self) -> None:
        return None


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records lifecycle outcomes.
class FakeProcess:
    """Mutable process fake that records lifecycle outcomes."""

    stdout: SlowPipe
    stdin: SlowPipe = field(default_factory=lambda: SlowPipe(()))
    stderr: SlowPipe = field(default_factory=lambda: SlowPipe(()))
    pid: int = 12_345
    returncode: int | None = None
    terminated: bool = False
    killed: bool = False
    wait_timeouts_remaining: int = 0
    wait_calls: int = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_timeouts_remaining > 0:
            self.wait_timeouts_remaining -= 1
            raise TimeoutExpired(cmd="revision-worker", timeout=timeout or 0.0)
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
    """Single-process factory."""

    process: FakeProcess

    def start(self, command: tuple[str, ...], environment: dict[str, str]) -> WorkerProcess:
        del command, environment
        return self.process


def test_client_terminates_worker_on_ready_timeout(tmp_path: Path) -> None:
    process = FakeProcess(stdout=SlowPipe((_ready(),), delay_seconds=0.1))
    client = _client(tmp_path, process, timeout_seconds=0.01)

    with pytest.raises(HuntSetupError, match="timeout"):
        _ = client.__enter__()

    assert process.terminated is True


def test_client_rejects_unpinned_ready_record(tmp_path: Path) -> None:
    process = FakeProcess(stdout=SlowPipe((_ready(load_count=2),)))
    client = _client(tmp_path, process)

    with pytest.raises(HuntSetupError, match="load_count"):
        _ = client.__enter__()

    assert process.terminated is True


def test_client_closes_all_pipes_and_reaps_process_after_ready_failure(tmp_path: Path) -> None:
    # Given
    process = FakeProcess(stdout=SlowPipe((_ready(load_count=2),)))
    client = _client(tmp_path, process)

    # When / Then
    with pytest.raises(HuntSetupError, match="load_count"):
        _ = client.__enter__()
    assert process.terminated is True
    assert process.wait_calls == 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_client_requires_positive_pid_and_reports_polled_liveness(tmp_path: Path) -> None:
    # Given
    process = FakeProcess(stdout=SlowPipe((_ready(),)), pid=0)
    client = _client(tmp_path, process)
    opened = client.__enter__()

    # When / Then
    with pytest.raises(HuntSetupError, match="positive"):
        _ = opened.process_id
    process.pid = 12_345
    assert opened.is_alive is True
    process.returncode = 9
    assert opened.is_alive is False
    opened.close()


def test_client_rejects_spoofed_ready_prompt_hash(tmp_path: Path) -> None:
    process = FakeProcess(stdout=SlowPipe((_ready(prompt_sha256="0" * 64),)))
    client = _client(tmp_path, process)

    with pytest.raises(HuntSetupError, match="prompt sha256"):
        _ = client.__enter__()

    assert process.terminated is True


def test_client_rejects_changed_field_outside_remask_fields(tmp_path: Path) -> None:
    process = FakeProcess(stdout=SlowPipe((_ready(), _shutdown("shutdown-000001"))))
    with (
        _client(tmp_path, process) as client,
        pytest.raises(HuntSetupError, match="changed field"),
    ):
        _ = client.revise(_revision(changed_field="width", remask_fields=("height",)))

    assert process.stdin.writes == [
        (
            '{"operation":"shutdown","payload":null,"request_id":"shutdown-000001",'
            '"schema_version":"dllm-selective-revision-request/v1"}\n'
        )
    ]


def test_client_terminates_worker_for_mismatched_shutdown_request_hash(tmp_path: Path) -> None:
    # Given
    process = FakeProcess(
        stdout=SlowPipe(
            (
                _ready(),
                _shutdown("shutdown-000001", request_sha256="0" * 64),
            )
        )
    )
    client = _client(tmp_path, process)
    opened = client.__enter__()

    # When
    opened.close()

    # Then
    assert process.terminated is True


def test_client_terminates_and_raises_when_shutdown_wait_times_out(tmp_path: Path) -> None:
    # Given
    process = FakeProcess(
        stdout=SlowPipe((_ready(), _shutdown("shutdown-000001"))),
        wait_timeouts_remaining=1,
    )
    client = _client(tmp_path, process)
    opened = client.__enter__()

    # When / Then
    with pytest.raises(HuntSetupError, match=r"shutdown.*timeout"):
        opened.close()
    assert process.terminated is True
    assert process.stdin.closed is True


def test_campaign_client_uses_offline_read_only_cache_and_single_thread_limits(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, FakeProcess(stdout=SlowPipe(())), read_only_cache=True)
    command = client.command()
    environment = client.environment()

    assert command[-1] == "--read-only-cache"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_read_only_worker_cache_must_be_prewarmed_and_is_never_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing-cache"
    with pytest.raises(WorkerCacheError, match="pre-warmed"):
        _ = prepare_worker_cache(missing, read_only_cache=True)
    assert not missing.exists()


def _client(
    tmp_path: Path,
    process: FakeProcess,
    *,
    timeout_seconds: float = 1.0,
    read_only_cache: bool = False,
) -> RevisionWorkerClient:
    prompt_path = tmp_path / "prompt.md"
    _ = prompt_path.write_text("prompt context", encoding="utf-8")
    return RevisionWorkerClient(
        runtime_python=tmp_path / "python",
        cache_dir=tmp_path / "cache",
        prompt_path=prompt_path,
        prompt_sha256=sha256_bytes(prompt_path.read_bytes()),
        device="cuda:0",
        timeout_seconds=timeout_seconds,
        read_only_cache=read_only_cache,
        factory=FakeFactory(process),
    )


def _ready(load_count: int = 1, prompt_sha256: str | None = None) -> str:
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
                "load_count": load_count,
                "prompt_sha256": (
                    sha256_bytes(b"prompt context") if prompt_sha256 is None else prompt_sha256
                ),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _shutdown(request_id: str, *, request_sha256: str | None = None) -> str:
    return (
        json.dumps(
            {
                "kind": "shutdown",
                "request_id": request_id,
                "request_sha256": (
                    request_hash(shutdown_request(request_id))
                    if request_sha256 is None
                    else request_sha256
                ),
                "status": "ok",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _revision(
    *,
    changed_field: FieldName = "width",
    remask_fields: tuple[FieldName, ...] = ("width",),
) -> RevisionRequest:
    width_bit = 1 << FIELD_ORDER.index("width")
    return RevisionRequest(
        particle_id="particle-1",
        parent_token_ids=tuple(range(10)),
        parent_selectors=(0, 1, 2, 3, 0, 1, 2, 3, 0, 1),
        slot_positions=tuple(range(10)),
        remask_fields=remask_fields,
        changed_field=changed_field,
        remask_bitmap=width_bit,
        frozen_bitmap=((1 << len(FIELD_ORDER)) - 1) ^ width_bit,
        revision_steps=8,
        seed=2,
        rng_state_sha256="e" * 64,
        feedback_digest="f" * 64,
        mode="selective-revision/v1",
    )
