# ruff: noqa: D100,D103

from __future__ import annotations

from tools.poc_revision_experiment_json import (
    JsonArray,
    JsonObject,
    require_array,
    require_bool,
    require_int,
    require_object,
)


def live_seed_binding_error(schedule: JsonObject, cli_manifest: JsonObject) -> str | None:
    errors: list[str] = []
    fuzzer = require_object(schedule, "fuzzer_seeds")
    if not require_bool(fuzzer, "consumed"):
        errors.append("fuzzer seed consumption")
    if require_array(fuzzer, "values") != _int_array(cli_manifest, "fuzzer_seeds"):
        errors.append("manifest fuzzer seed mismatch")
    particle_base = require_object(schedule, "particle_seed_base")
    if not require_bool(particle_base, "consumed"):
        errors.append("particle seed consumption")
    base_value = require_int(particle_base, "value")
    if base_value != require_int(cli_manifest, "particle_seed_base"):
        errors.append("particle seed mismatch")
    particle_values = require_array(require_object(schedule, "particle_seeds"), "values")
    particle_count = require_int(cli_manifest, "particle_count")
    if particle_values != [base_value + index for index in range(particle_count)]:
        errors.append("particle tuple mismatch")
    proof_seed = require_object(schedule, "proof_seed")
    if not require_bool(proof_seed, "consumed"):
        errors.append("proof seed consumption")
    if require_int(proof_seed, "value") != require_int(cli_manifest, "proof_seed"):
        errors.append("proof seed mismatch")
    if require_bool(require_object(schedule, "retrieval_seed"), "consumed"):
        errors.append("retrieval seed consumption mismatch")
    return errors[0] if errors else None


def _int_array(payload: JsonObject, key: str) -> JsonArray:
    values = require_array(payload, key)
    for value in values:
        if not (isinstance(value, int) and not isinstance(value, bool)):
            return []
    return values
