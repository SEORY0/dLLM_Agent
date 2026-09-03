"""Target-owned manifest and evidence contracts."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.targets.profiles import TargetProfile

GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
DockerImageId = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
HexSeed = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{2})+$", max_length=131_072)]


class HuntManifest(FrozenModel):
    """Pinned target, model, sandbox, and fuzzing controls."""

    schema_version: Literal["dllm-hunt/v1"]
    hunt_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    commit_sha: GitCommit
    security_policy_url: str = Field(min_length=1)
    security_policy_sha256: Sha256
    prompt: str = Field(min_length=1)
    model_id: Literal["ByteDance-Seed/Stable-DiffCoder-8B-Instruct"]
    model_revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"]
    image_id: DockerImageId
    target_profile: TargetProfile = "header-only"
    sanitizers: Literal["address,undefined", "address"] = "address,undefined"
    allowed_includes: tuple[str, ...] = Field(min_length=1)
    required_symbols: tuple[str, ...] = Field(min_length=1)
    seed_hex: tuple[HexSeed, ...] = Field(min_length=1)
    fuzz_seconds: int = Field(ge=1, le=3600)
    max_len: int = Field(ge=1, le=1_048_576)
    rss_limit_mb: int = Field(ge=256, le=8192)
    timeout_seconds: int = Field(ge=1, le=60)

    @model_validator(mode="after")
    def _validate_target_sanitizer(self) -> Self:
        match self.target_profile:
            case "header-only":
                return self
            case "openexr-v3.4":
                if self.sanitizers != "address":
                    msg = "openexr-v3.4 requires address sanitizer"
                    raise ValueError(msg)
                return self


class TargetEvidence(FrozenModel):
    """Pinned clean Git worktree and disclosure evidence."""

    repository_url: str
    commit_sha: GitCommit
    tree_oid: GitCommit
    source_archive_sha256: Sha256
    security_policy_url: str
    security_policy_sha256: Sha256
