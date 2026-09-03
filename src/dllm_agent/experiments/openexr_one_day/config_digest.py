"""Detached digest verification for OpenEXR one-day configs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.coordinator import CampaignConfig, CoordinatorError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_DIGEST_FIELD: Final = "config_sha256"
_PLACEHOLDER_HASHES: Final = frozenset({char * 64 for char in "0123456789abcdef"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def config_payload_sha256(payload: Mapping[str, JsonValue]) -> str:
    """Hash normalized config content with its detached digest field removed."""
    normalized_input = dict(payload)
    if not isinstance(normalized_input.get(_DIGEST_FIELD), str):
        normalized_input[_DIGEST_FIELD] = "0" * 64
    config = CampaignConfig.model_validate(normalized_input)
    normalized = config.model_dump(mode="json", exclude={_DIGEST_FIELD})
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(canonical)


def load_verified_campaign_config(path: Path) -> CampaignConfig:
    """Load a config only when its detached digest is present and exact."""
    try:
        payload = _JSON_OBJECT.validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason=f"invalid one-day config: {error}") from error
    digest = payload.get(_DIGEST_FIELD)
    if not isinstance(digest, str):
        raise CoordinatorError(reason="config digest missing")
    if digest in _PLACEHOLDER_HASHES:
        raise CoordinatorError(reason="config digest placeholder hash")
    expected = config_payload_sha256(payload)
    if digest != expected:
        raise CoordinatorError(reason="config digest mismatch")
    try:
        return CampaignConfig.model_validate(payload)
    except ValidationError as error:
        raise CoordinatorError(reason=f"invalid one-day config: {error}") from error
