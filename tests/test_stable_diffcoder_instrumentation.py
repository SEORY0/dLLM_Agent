"""Focused Stable-DiffCoder transfer instrumentation tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.stable_diffcoder_payloads import (  # noqa: E402
    build_token_transfer_evidence,
)


def test_build_token_transfer_evidence_maps_prompt_residual_block_positions() -> None:
    # Given
    decoded_tokens = {101: "A", 202: "B"}

    def decode_token(token_id: int) -> str:
        return decoded_tokens[token_id]

    # When
    transfers = build_token_transfer_evidence(
        block_start=4,
        prompt_length=6,
        local_positions=(3, 2),
        token_ids=(202, 101),
        confidences=(0.25, 0.9),
        decode_token=decode_token,
    )

    # Then
    assert transfers[0].generated_position == 0
    assert transfers[0].token_id == 101
    assert transfers[0].text == "A"
    assert transfers[0].confidence == 0.9
    assert transfers[1].generated_position == 1
    assert transfers[1].token_id == 202
    assert transfers[1].text == "B"
    assert transfers[1].confidence == 0.25
