"""Resident bank generation orchestration contracts."""

from dataclasses import dataclass
from pathlib import Path

from dllm_agent.experiments.multitarget.bank_generation import (
    BankRevisionRequest,
    InitialBankRequest,
    RevisionMode,
    generate_initial_bank,
    revise_bank,
)
from dllm_agent.experiments.multitarget.candidate_bank import (
    ProbeObservation,
)
from dllm_agent.experiments.multitarget.slot_sampler import BlockDenoiseResult


@dataclass(slots=True)
class RecordingTokenizer:
    """Tokenizer that exposes numeric selector tokens and records prompt reads."""

    encoded_texts: list[str]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        self.encoded_texts.append(text)
        return [30 + (sum(text.encode("utf-8")) % 67)]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        decoded: dict[int, str] = {10: "0", 11: "1", 12: "2", 13: "3"}
        return decoded.get(token_ids[0], "x")

    def __len__(self) -> int:
        return 32


class RecordingModel:
    """Resident fake model that proves calls go through one shared object."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...], int, int]] = []
        self.initial_call_count: int = 0
        self.revision_call_count: int = 0

    def denoise_block(  # noqa: PLR0913, PLR0917
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        del token_ids
        self.calls.append((block_index, mask_positions, seed, step_index))
        if all(token_id is None for token_id in previous_token_ids):
            self.initial_call_count += 1
        else:
            self.revision_call_count += 1
        selected = tuple(
            _pick_token(allowed_token_ids, previous_token, seed, position, step_index)
            for position, previous_token in zip(mask_positions, previous_token_ids, strict=True)
        )
        return BlockDenoiseResult(
            token_ids=selected,
            confidences=tuple(0.6 + ((position + seed) % 3) * 0.1 for position in mask_positions),
            gpu_ms=5,
        )


def test_initial_bank_generation_reuses_resident_model_and_freezes_sixteen_unique(
    tmp_path: Path,
) -> None:
    # Given
    prompt_path = tmp_path / "libpng.md"
    _ = prompt_path.write_text("# libpng prompt\nkeep PNG structure\n", encoding="utf-8")
    tokenizer = RecordingTokenizer(encoded_texts=[])
    model = RecordingModel()

    # When
    initial = generate_initial_bank(
        InitialBankRequest(
            target_id="libpng",
            prompt_path=prompt_path,
            initial_seeds=tuple(range(32)),
            initial_steps=8,
        ),
        tokenizer=tokenizer,
        model=model,
    )
    result = revise_bank(
        initial,
        BankRevisionRequest(mode=RevisionMode.FULL, revision_steps=2),
        tokenizer=tokenizer,
        model=model,
    )

    # Then
    assert result.target_id == "libpng"
    assert result.prompt_sha256 != ""
    assert tokenizer.encoded_texts[0] == "# libpng prompt\nkeep PNG structure\n"
    assert len(initial.raw_samples) == 32
    assert all(sample.trajectory_steps for sample in initial.raw_samples)
    assert result.bank.raw_count == 32
    assert result.bank.unique_count >= 16
    assert len(result.selected) == 16
    assert len(result.revised) == 16
    assert len({item.snapshot.sha256 for item in result.revised}) == 16
    assert result.cost.physical_total.forward_calls == model.revision_call_count
    assert result.cost.logical_total.forward_calls == len(model.calls)


def test_three_revision_modes_reuse_shared_initial_bank_without_new_initial_calls(
    tmp_path: Path,
) -> None:
    # Given
    prompt_path = tmp_path / "libxml2.md"
    _ = prompt_path.write_text("# libxml2 prompt\n", encoding="utf-8")
    tokenizer = RecordingTokenizer(encoded_texts=[])
    model = RecordingModel()
    initial = generate_initial_bank(
        InitialBankRequest(
            target_id="libxml2",
            prompt_path=prompt_path,
            initial_seeds=tuple(range(32)),
            initial_steps=8,
        ),
        tokenizer=tokenizer,
        model=model,
    )
    initial_call_count = model.initial_call_count
    parent_hashes = tuple(candidate.snapshot.sha256 for candidate in initial.selected)
    observations = tuple(
        _observation(candidate.snapshot.sha256, coverage=index * 10)
        for index, candidate in enumerate(initial.selected)
    )

    # When
    full = revise_bank(
        initial,
        BankRevisionRequest(mode=RevisionMode.FULL, revision_steps=2),
        tokenizer=tokenizer,
        model=model,
    )
    selective = revise_bank(
        initial,
        BankRevisionRequest(
            revision_steps=2,
            mode=RevisionMode.SELECTIVE,
            observations=observations,
        ),
        tokenizer=tokenizer,
        model=model,
    )
    pseudo = revise_bank(
        initial,
        BankRevisionRequest(
            revision_steps=2,
            mode=RevisionMode.PSEUDO,
            observations=observations,
            pseudo_seed=3,
        ),
        tokenizer=tokenizer,
        model=model,
    )

    # Then
    assert (
        InitialBankRequest(
            target_id="libxml2",
            prompt_path=prompt_path,
            initial_seeds=tuple(range(32)),
            initial_steps=8,
        ).mask_token_id
        == 5
    )
    assert model.initial_call_count == initial_call_count
    assert tuple(candidate.snapshot.sha256 for candidate in full.selected) == parent_hashes
    assert tuple(candidate.snapshot.sha256 for candidate in selective.selected) == parent_hashes
    assert tuple(candidate.snapshot.sha256 for candidate in pseudo.selected) == parent_hashes
    assert all(
        item.parent.snapshot.sha256 == parent_hashes[index]
        for index, item in enumerate(selective.revised)
    )
    assert all(
        observation.feedback_source_sha256 != observation.candidate_sha256
        for observation in pseudo.feedback_observations
    )
    assert len(pseudo.remask_indexes) == 16


def test_revision_duplicate_falls_back_to_unique_parent(tmp_path: Path) -> None:
    # Given
    prompt_path = tmp_path / "libpng.md"
    _ = prompt_path.write_text("# libpng prompt\n", encoding="utf-8")
    tokenizer = RecordingTokenizer(encoded_texts=[])
    model = RecordingModel()
    initial = generate_initial_bank(
        InitialBankRequest(
            target_id="libpng",
            prompt_path=prompt_path,
            initial_seeds=tuple(range(32)),
            initial_steps=8,
        ),
        tokenizer=tokenizer,
        model=model,
    )
    observations = tuple(
        _observation(candidate.snapshot.sha256, coverage=0) for candidate in initial.selected
    )

    # When
    result = revise_bank(
        initial,
        BankRevisionRequest(
            revision_steps=1,
            mode=RevisionMode.SELECTIVE,
            observations=observations,
        ),
        tokenizer=tokenizer,
        model=model,
    )

    # Then
    assert len({item.snapshot.sha256 for item in result.revised}) == 16
    assert any(not item.used_revision for item in result.revised)


def _pick_token(
    allowed_token_ids: tuple[int, ...],
    previous_token_id: int | None,
    seed: int,
    position: int,
    step_index: int,
) -> int:
    del step_index
    slot_index = max(0, (position - 6) // 3)
    if previous_token_id is None:
        divisor = 1
        for _ in range(slot_index):
            divisor *= 4
        selected_index = (seed // divisor) % len(allowed_token_ids)
        return allowed_token_ids[selected_index]
    return allowed_token_ids[(seed + position + previous_token_id) % len(allowed_token_ids)]


def _observation(candidate_sha256: str, coverage: int) -> ProbeObservation:
    return ProbeObservation(
        candidate_sha256=candidate_sha256,
        feedback_source_sha256=candidate_sha256,
        canary_triggered=False,
        sanitizer_observed=False,
        coverage_edges=coverage,
        value_profile_features=coverage,
        executions_per_second=1,
        log_sha256="0" * 64,
    )
