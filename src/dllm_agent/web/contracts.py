"""Typed browser-facing contracts for the WebUI core."""

import sys
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, NewType

from pydantic import BaseModel, ConfigDict, Field

from dllm_agent.hunt.models import HarnessCandidate

ResourceId = NewType("ResourceId", str)
JobId = NewType("JobId", str)


class WebModel(BaseModel):
    """Immutable strict WebUI schema base."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)


class ResourceKind(StrEnum):
    """Catalog resource kinds exposed to browser code."""

    CONFIG = "config"
    CANDIDATE_BANK = "candidate_bank"
    HUNT_REPORT = "hunt_report"
    PROMPT = "prompt"
    DOC = "doc"
    ARTIFACT = "artifact"


class ResourceDescriptor(WebModel):
    """Opaque catalog entry for one approved repository file."""

    resource_id: str = Field(min_length=1)
    kind: ResourceKind
    relative_path: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


class TrajectoryView(WebModel):
    """Browser-safe candidate trajectory adapter for a candidate bank."""

    resource_id: str = Field(min_length=1)
    candidates: tuple[HarnessCandidate, ...]


class CatalogListing(WebModel):
    """Complete catalog snapshot."""

    resources: tuple[ResourceDescriptor, ...]


class SampleJobSpec(WebModel):
    """Browser contract for live candidate-bank generation."""

    kind: Literal["sample"] = "sample"
    manifest_id: str = Field(min_length=1)
    count: int = Field(default=2, ge=2, le=8)
    seed: int = Field(default=20_260_819, ge=0)


class HuntJobSpec(WebModel):
    """Browser contract for live target hunting."""

    kind: Literal["hunt"] = "hunt"
    manifest_id: str = Field(min_length=1)
    candidate_bank_id: str = Field(min_length=1)


class JobRuntimeConfig(WebModel):
    """Trusted server-side paths never accepted from browser requests."""

    repo_root: Path
    jobs_root: Path
    runtime_python: Path
    cache_dir: Path
    source_root: Path
    python_executable: str = sys.executable


class JobStatus(WebModel):
    """Durable browser-visible job status."""

    job_id: str
    kind: Literal["sample", "hunt"]
    state: Literal["queued", "running", "succeeded", "failed"]
    output_dir: str = Field(min_length=1)
    exit_code: int | None = None
    log_tail: str = ""


class JobEvent(WebModel):
    """One durable journal event."""

    job_id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    message: str = Field(min_length=1)
