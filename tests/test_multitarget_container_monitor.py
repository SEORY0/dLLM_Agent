"""Container monitoring contracts for Magma multitarget campaigns."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerMonitorError,
    launch_container,
    wait_for_container,
)
from dllm_agent.experiments.multitarget.docker_runtime import (
    ContainerRequest,
    runtime_profile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

IMAGE_ID = "sha256:" + ("a" * 64)


@dataclass(slots=True)
class FakeRunner:
    responses: Mapping[tuple[str, str], subprocess.CompletedProcess[str] | BaseException]
    calls: list[tuple[tuple[str, ...], int | float | None]] = field(default_factory=list)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float | None,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        command = tuple(argv)
        self.calls.append((command, timeout))
        response = self.responses[(command[0], command[1])]
        if isinstance(response, BaseException):
            raise response
        return response


def test_launch_container_returns_exact_detached_container_id(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runner = FakeRunner(
        {
            ("docker", "run"): subprocess.CompletedProcess(
                args=("docker", "run"),
                returncode=0,
                stdout="7f3b8fd1\n",
                stderr="",
            )
        }
    )

    launch = launch_container(request, runner=runner)

    assert launch.container_id == "7f3b8fd1"
    assert runner.calls[0][0][:3] == ("docker", "run", "--detach")
    assert runner.calls[0][1] == 30


def test_launch_container_rejects_mixed_stdout_instead_of_guessing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runner = FakeRunner(
        {
            ("docker", "run"): subprocess.CompletedProcess(
                args=("docker", "run"),
                returncode=0,
                stdout="7f3b8fd1\nextra log\n",
                stderr="",
            )
        }
    )

    with pytest.raises(ContainerMonitorError, match="exactly one container id"):
        _ = launch_container(request, runner=runner)


def test_wait_for_container_collects_logs_inspect_monitor_rows_and_findings(
    tmp_path: Path,
) -> None:
    _write_monitor(tmp_path / "monitor" / "0", "BUG_A_R,BUG_A_T,BUG_B_R,BUG_B_T\n1,0,0,0\n")
    _write_monitor(tmp_path / "monitor" / "30", "BUG_A_R,BUG_A_T,BUG_B_R,BUG_B_T\n2,1,0,0\n")
    finding = tmp_path / "findings" / "crash-a"
    finding.parent.mkdir(parents=True)
    _ = finding.write_bytes(b"boom")
    logs = (
        "2026-08-21T00:00:00Z #1 INITED cov: 2 ft: 5 exec/s: 10\n"
        "2026-08-21T00:00:30Z #44 DONE cov: 14 ft: 33 corp: 7/1Kb exec/s 200\n"
        "2026-08-21T00:00:31Z stat::number_of_executed_units: 9001"
    )
    runner = FakeRunner(
        {
            ("docker", "wait"): subprocess.CompletedProcess(
                args=("docker", "wait"), returncode=0, stdout="0\n", stderr=""
            ),
            ("docker", "logs"): subprocess.CompletedProcess(
                args=("docker", "logs"), returncode=0, stdout=logs, stderr=""
            ),
            ("docker", "inspect"): subprocess.CompletedProcess(
                args=("docker", "inspect"),
                returncode=0,
                stdout='{"Status":"exited","ExitCode":0,"OOMKilled":false}\n',
                stderr="",
            ),
        }
    )

    result = wait_for_container("7f3b8fd1", shared_dir=tmp_path, runner=runner, timeout_seconds=45)

    assert result.exit_code == 0
    assert result.docker_status == "exited"
    assert result.oom_killed is False
    assert result.timed_out is False
    assert result.process_error is None
    assert result.logs_text == logs
    assert result.log_sha256 == hashlib.sha256(logs.encode()).hexdigest()
    assert result.metrics.coverage_edges == 14
    assert result.metrics.value_profile_features == 33
    assert result.metrics.executions_per_second == 200
    assert result.metrics.executed_units == 9001
    assert result.canaries["BUG_A"].reached == ((0, 1), (30, 2))
    assert result.canaries["BUG_A"].triggered == ((0, 0), (30, 1))
    assert result.findings[0].path == str(finding)
    assert result.findings[0].sha256 == hashlib.sha256(b"boom").hexdigest()
    assert result.findings[0].size_bytes == 4
    assert [call[0][:2] for call in runner.calls] == [
        ("docker", "wait"),
        ("docker", "logs"),
        ("docker", "inspect"),
    ]


def test_wait_timeout_is_reported_without_claiming_success(tmp_path: Path) -> None:
    runner = FakeRunner(
        {
            ("docker", "wait"): subprocess.TimeoutExpired(
                cmd=("docker", "wait", "7f3b8fd1"), timeout=3
            ),
            ("docker", "logs"): subprocess.CompletedProcess(
                args=("docker", "logs"), returncode=0, stdout="partial log", stderr=""
            ),
            ("docker", "inspect"): subprocess.CompletedProcess(
                args=("docker", "inspect"),
                returncode=0,
                stdout='{"Status":"running","ExitCode":137,"OOMKilled":true}\n',
                stderr="",
            ),
        }
    )

    result = wait_for_container("7f3b8fd1", shared_dir=tmp_path, runner=runner, timeout_seconds=3)

    assert result.timed_out is True
    assert result.process_error == "docker wait timed out after 3s"
    assert result.exit_code == 137
    assert result.oom_killed is True
    assert result.log_sha256 == hashlib.sha256(b"partial log").hexdigest()


def test_fork_mode_uses_last_progress_unit_when_final_stat_is_zero(tmp_path: Path) -> None:
    logs = (
        "#4458: cov: 626 ft: 775 corp: 10 exec/s 2229 time: 2s\n"
        "#8123: cov: 630 ft: 780 corp: 12 exec/s 2400 time: 4s\n"
        "stat::number_of_executed_units: 0\n"
    )
    runner = FakeRunner(
        {
            ("docker", "wait"): subprocess.CompletedProcess(
                args=("docker", "wait"), returncode=0, stdout="0\n", stderr=""
            ),
            ("docker", "logs"): subprocess.CompletedProcess(
                args=("docker", "logs"), returncode=0, stdout=logs, stderr=""
            ),
            ("docker", "inspect"): subprocess.CompletedProcess(
                args=("docker", "inspect"),
                returncode=0,
                stdout='{"Status":"exited","ExitCode":0,"OOMKilled":false}\n',
                stderr="",
            ),
        }
    )

    result = wait_for_container("forked", shared_dir=tmp_path, runner=runner, timeout_seconds=10)

    assert result.metrics.executed_units == 8123


def _request(tmp_path: Path) -> ContainerRequest:
    return ContainerRequest(
        name="dllm-libpng-selective-r0",
        profile=runtime_profile("libpng"),
        image_id=IMAGE_ID,
        shared_dir=tmp_path / "shared",
        corpus_dir=tmp_path / "corpus",
        duration_seconds=30,
        fuzzer_seed=0,
        cpu_index=0,
        poll_seconds=5,
    )


def _write_monitor(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")
