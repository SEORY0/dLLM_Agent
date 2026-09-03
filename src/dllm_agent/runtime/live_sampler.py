"""External Stable-DiffCoder process boundary for live candidate generation.

The runtime package owns launching and isolating external model processes.
The hunt package owns candidate-bank semantics and validation.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import parse_json, read_bytes
from dllm_agent.hunt.models import CandidateBank
from dllm_agent.hunt.validation import validate_candidate_bank

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_SCRIPT = _REPOSITORY_ROOT / "tools/stable_diffcoder_runner.py"


class ProcessRunner(Protocol):
    """Narrow process boundary used by the live sampler."""

    def run(
        self, command: tuple[str, ...], timeout_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one fixed-argv model process."""
        ...


@dataclass(frozen=True, slots=True)
class SubprocessRunner:
    """Production process runner with a minimal environment."""

    def run(
        self, command: tuple[str, ...], timeout_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        """Execute the pinned model runner without a shell."""
        cache_dir = command[command.index("--cache-dir") + 1]
        environment = {
            "HF_HOME": cache_dir,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(Path(cache_dir) / "modules"),
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
        try:
            return subprocess.run(  # noqa: S603 - fixed script and argv, shell disabled
                command,
                capture_output=True,
                check=False,
                env=environment,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            detail = "Stable-DiffCoder sampler timed out"
            raise HuntSetupError(detail) from error
        except OSError as error:
            raise HuntSetupError(error.strerror or "Stable-DiffCoder sampler failed") from error


@dataclass(frozen=True, slots=True)
class StableDiffCoderProcessSampler:
    """Invoke the real dLLM in an isolated Python dependency environment."""

    runtime_python: Path
    prompt_path: Path
    cache_dir: Path
    count: int
    seed: int
    steps: int
    gen_length: int
    block_length: int
    temperature: float
    device: str
    timeout_seconds: float
    runner: ProcessRunner = SubprocessRunner()

    def sample(self) -> CandidateBank:
        """Generate and strictly parse one immutable live candidate bank."""
        runtime = _required_file(self.runtime_python, "model runtime")
        prompt_path = _required_file(self.prompt_path, "model prompt")
        prompt = read_bytes(prompt_path, "model prompt")
        try:
            cache_dir = self.cache_dir.resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise HuntSetupError(error.strerror or "model cache directory unavailable") from error
        with tempfile.TemporaryDirectory(prefix="dllm-live-sampler-") as directory:
            output = Path(directory) / "candidate-bank.json"
            command = self._command(runtime, prompt_path, cache_dir, output)
            result = self.runner.run(command, self.timeout_seconds)
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace")[-1000:].strip()
                message = f"Stable-DiffCoder sampler exited {result.returncode}: {detail}"
                raise HuntSetupError(message)
            bank = parse_json(output, CandidateBank, "candidate bank")
            validate_candidate_bank(bank, prompt, output)
            return bank

    def _command(
        self, runtime: Path, prompt: Path, cache_dir: Path, output: Path
    ) -> tuple[str, ...]:
        """Build the fixed argument vector passed to the reviewed runner."""
        return (
            str(runtime),
            str(_RUNNER_SCRIPT),
            "--prompt",
            str(prompt),
            "--output",
            str(output),
            "--count",
            str(self.count),
            "--seed",
            str(self.seed),
            "--steps",
            str(self.steps),
            "--gen-length",
            str(self.gen_length),
            "--block-length",
            str(self.block_length),
            "--temperature",
            str(self.temperature),
            "--cache-dir",
            str(cache_dir),
            "--device",
            self.device,
        )


def _required_file(path: Path, label: str) -> Path:
    try:
        absolute = path.absolute()
    except (OSError, RuntimeError) as error:
        detail = f"{label} does not exist: {path}"
        raise HuntSetupError(detail) from error
    if not absolute.exists() or not absolute.is_file():
        detail = f"{label} is not a regular file: {path}"
        raise HuntSetupError(detail)
    return absolute
