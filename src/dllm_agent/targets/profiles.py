"""Closed target profiles for reviewed build and coverage behavior."""

from typing import Literal

type TargetProfile = Literal["header-only", "openexr-v3.4"]


def require_target_profile(value: str) -> TargetProfile:
    """Parse one runtime string into the closed target-profile type."""
    match value:
        case "header-only":
            return "header-only"
        case "openexr-v3.4":
            return "openexr-v3.4"
        case unsupported:
            msg = f"unsupported target profile: {unsupported}"
            raise ValueError(msg)
