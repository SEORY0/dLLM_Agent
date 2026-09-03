"""Selective-revision request payload boundary tests."""

from __future__ import annotations

import json

import pytest
from tools.runtime.selective_revision_json import WireProtocolError
from tools.selective_revision_payloads import parse_request_line


@pytest.mark.parametrize(
    "line",
    [
        json.dumps(
            {
                "operation": "initialize",
                "request_id": "top-level-id",
                "schema_version": "dllm-selective-revision-request/v1",
                "payload": {
                    "request_id": "payload-id",
                    "particle_id": "particle-1",
                    "seed": 1,
                    "steps": 8,
                },
            }
        ),
        json.dumps(
            {
                "operation": "revise",
                "request_id": "top-level-id",
                "schema_version": "dllm-selective-revision-request/v1",
                "payload": {
                    "request_id": "payload-id",
                    "particle_id": "particle-1",
                    "parent": {
                        "token_ids": list(range(10)),
                        "slot_positions": list(range(10)),
                        "selectors": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                        "token_state_sha256": "a" * 64,
                    },
                    "parent_token_sha256": "b" * 64,
                    "remask_bitmap": 1,
                    "frozen_bitmap": (1 << 10) - 2,
                    "revision_steps": 8,
                    "seed": 2,
                    "rng_state_sha256": "c" * 64,
                    "feedback_digest": "d" * 64,
                    "mode": "selective-revision/v1",
                },
            }
        ),
    ],
    ids=("initialize", "revise"),
)
def test_parse_request_line_rejects_mismatched_payload_request_id(line: str) -> None:
    # Given a structurally valid request whose envelope and payload IDs differ

    # When / Then
    with pytest.raises(WireProtocolError, match="request_id"):
        _ = parse_request_line(line)
