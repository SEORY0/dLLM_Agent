"""Append-only per-iteration harness identity ledger (Task 6, spec §10)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import (
    AuditLedgerError,
    HarnessSynthesisLedgerRecord,
    append,
    append_many,
    load,
    records_sha256,
)

_HARNESS_SHA = sha256_bytes(b"harness v1")
_KB_SHA = sha256_bytes(b"kb v1")


def _record(
    *, iteration: int, arm: str = "feedback", compiled: bool = True
) -> HarnessSynthesisLedgerRecord:
    return HarnessSynthesisLedgerRecord(
        run_id="run-1",
        iteration=iteration,
        arm=arm,
        seed=2000 + iteration,
        harness_source_sha256=_HARNESS_SHA,
        knowledge_base_sha256=_KB_SHA,
        usage_example_names=("checkOpenEXRFile",),
        compiled=compiled,
        recorded_at_unix_seconds=1_800_000_000.0 + iteration,
    )


def test_load_returns_empty_tuple_when_file_is_absent(tmp_path: Path) -> None:
    assert load(tmp_path / "missing.jsonl") == ()


def test_round_trip_append_and_load(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    record = _record(iteration=0)

    append(path, record)

    assert load(path) == (record,)


def test_append_accumulates_multiple_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _record(iteration=0)
    second = _record(iteration=1, arm="random")

    append(path, first)
    append(path, second)

    assert load(path) == (first, second)


def test_load_rejects_a_corrupted_ledger_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    _ = path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(AuditLedgerError, match="line 1"):
        _ = load(path)


def test_append_under_concurrent_writers_loses_no_record(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    writer_count = 20
    records = tuple(_record(iteration=index) for index in range(writer_count))

    with ThreadPoolExecutor(max_workers=writer_count) as pool:
        futures = [pool.submit(append, path, record) for record in records]
        for future in futures:
            future.result()

    assert len(load(path)) == writer_count


def test_append_many_writes_all_records_in_one_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _record(iteration=0)
    second = _record(iteration=1)

    append_many(path, (first, second))

    assert load(path) == (first, second)


def test_append_many_with_no_records_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"

    append_many(path, ())

    assert not path.exists()


def test_records_sha256_is_none_for_an_empty_set() -> None:
    assert records_sha256(()) is None


def test_records_sha256_is_deterministic_over_the_same_records() -> None:
    records = (_record(iteration=0), _record(iteration=1))

    assert records_sha256(records) == records_sha256(records)


def test_a_failed_compile_records_the_error_and_no_coverage(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    record = HarnessSynthesisLedgerRecord(
        run_id="run-1",
        iteration=0,
        arm="feedback",
        seed=2000,
        harness_source_sha256=_HARNESS_SHA,
        knowledge_base_sha256=_KB_SHA,
        usage_example_names=(),
        compiled=False,
        compile_error="'ImfStream.h' file not found",
        coverage_edges=None,
        recorded_at_unix_seconds=1_800_000_000.0,
    )

    append(path, record)

    loaded = load(path)[0]
    assert loaded.compiled is False
    assert loaded.compile_error == "'ImfStream.h' file not found"
    assert loaded.coverage_edges is None
