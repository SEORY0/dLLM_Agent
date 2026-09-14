"""Shell-safe CLI for the OpenEXR one-day campaign.

# --- How to run ---
# uv run python tools/run_openexr_one_day.py status --root /data/dllm-agent/openexr-one-day
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol, cast

import typer
from pydantic import TypeAdapter, ValidationError

from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.config_digest import (
    config_payload_sha256,
    load_verified_campaign_config,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CoordinatorError,
    ResidentPoolSurface,
    campaign_status,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    ConfigArtifactMeasurements,
    build_qualification_smoke_plan,
    finalize_config_payload_from_artifacts,
    write_record_backed_qualification_from_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_collectors import (
    collect_all_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_runtime import (
    QualificationRuntimePaths,
    build_production_collector_inputs,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    WRAPPER_NAME,
    QualificationSmokeArtifact,
    QualificationSmokeError,
    QualificationSmokePlan,
    QualificationSmokeResult,
    load_smoke_artifact_file,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_production import (
    execute_production_qualification_smoke,
)
from dllm_agent.experiments.openexr_one_day.recovery import RecoveryError, resume_campaign
from dllm_agent.experiments.openexr_one_day.runtime import (
    OneDayRuntimeFactory,
    ProductionQualificationSmokeAdapter,
    ProductionRuntimeFactory,
)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _RunOnlyFactory(Protocol):
    def run(self, config: CampaignConfig) -> CampaignOutcome:
        """Run a qualified campaign."""
        ...


app = typer.Typer(no_args_is_help=True)


@app.command()
def qualify(
    config: Annotated[Path, typer.Option()],
    root: Annotated[Path, typer.Option()],
) -> None:
    """Record draft-only qualification evidence without starting the main campaign."""
    payload = qualify_verified(config, root)
    _ = typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def qualify_verified(config: Path, root: Path) -> dict[str, JsonValue]:
    """Verify the detached config digest before writing qualification evidence."""
    loaded = load_verified_campaign_config(config)
    payload: dict[str, JsonValue] = {
        "state": "qualified-draft-only",
        "config_id": loaded.config_id,
        "config_sha256": loaded.config_sha256,
    }
    _write_once_bytes(root / "qualification.json", _canonical_json(payload))
    return payload


@app.command("qualify-live")
def qualify_live(
    config: Annotated[Path, typer.Option()],
    measurements_root: Annotated[Path, typer.Option()],
    evidence_root: Annotated[Path, typer.Option()],
) -> None:
    """Write record-backed qualification evidence from measured gate artifacts."""
    loaded = load_verified_campaign_config(config)
    try:
        result = write_record_backed_qualification_from_measurements(
            config=loaded,
            evidence_root=evidence_root,
            measurements_root=measurements_root,
        )
    except CoordinatorError as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _ = typer.echo(
        json.dumps(
            {
                "state": result.state,
                "evidence_sha256": result.evidence.evidence_sha256,
                "failing_gate": result.start_gates.failing_gate(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@app.command("collect")
def collect(  # noqa: PLR0913, PLR0917
    config: Annotated[Path, typer.Option()],
    target_harness_path: Annotated[Path, typer.Option()],
    remote_code_path: Annotated[Path, typer.Option()],
    search_binary_path: Annotated[Path, typer.Option()],
    search_image_ref: Annotated[str, typer.Option()],
    coverage_binary_path: Annotated[Path, typer.Option()],
    coverage_image_ref: Annotated[str, typer.Option()],
    coverage_build_manifest_path: Annotated[Path, typer.Option()],
    c0_source_path: Annotated[Path, typer.Option()],
    dictionary_path: Annotated[Path, typer.Option()],
    smoke_artifact_path: Annotated[Path, typer.Option()],
) -> None:
    """Run all 12 production qualification collectors and publish measurements.

    Writes canonical per-gate measurement JSON under `<artifact_root>/measurements`
    (real nvidia-smi/procfs/Docker-coverage evidence, not synthetic test doubles).
    Run this after `finalize-config --qualification-state qualified` and before
    `qualify-live`.
    """
    loaded = load_verified_campaign_config(config)
    paths = QualificationRuntimePaths(
        target_harness_path=target_harness_path,
        remote_code_path=remote_code_path,
        search_binary_path=search_binary_path,
        coverage_binary_path=coverage_binary_path,
        coverage_build_manifest_path=coverage_build_manifest_path,
        c0_source_path=c0_source_path,
        dictionary_path=dictionary_path,
        search_image_ref=search_image_ref,
        coverage_image_ref=coverage_image_ref,
        smoke_artifact_path=smoke_artifact_path,
    )
    try:
        inputs = build_production_collector_inputs(config=loaded, paths=paths)
        evidence_path = collect_all_measurements(inputs)
    except CoordinatorError as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _ = typer.echo(
        json.dumps(
            {
                "measurements_root": str(inputs.measurements_root),
                "evidence_path": str(evidence_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@app.command("smoke-plan")
def smoke_plan(
    config: Annotated[Path, typer.Option()],
    smoke_root: Annotated[Path, typer.Option()],
) -> None:
    """Emit the isolated 2-pair x 5-arm x 300-second qualification smoke plan."""
    loaded = load_verified_campaign_config(config)
    plan = build_qualification_smoke_plan(config=loaded, smoke_root=smoke_root)
    _ = typer.echo(json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))


@app.command("smoke-run")
def smoke_run(
    config: Annotated[Path, typer.Option()],
    smoke_root: Annotated[Path, typer.Option()],
) -> None:
    """Run production qualification smoke and immediately validate completed resume."""
    loaded = load_verified_campaign_config(config)
    plan = build_qualification_smoke_plan(config=loaded, smoke_root=smoke_root)
    try:
        completed = _load_completed_smoke_artifact(plan)
        if completed is not None:
            _emit_smoke_artifact(completed)
            return
        with _production_factory(loaded).build(loaded) as graph, graph.resident_pool:
            adapter = ProductionQualificationSmokeAdapter(
                smoke_root=smoke_root,
                resident_pool=graph.resident_pool,
                cell_runner=graph.cell_runner,
                model_executor=graph.model_executor,
                post_search=graph.post_search,
            )
            first = execute_production_qualification_smoke(loaded, plan, adapter)
            resumed = execute_production_qualification_smoke(loaded, plan, adapter)
        _require_matching_smoke_resume(first, resumed)
        artifact = load_smoke_artifact_file(plan)
    except (CoordinatorError, QualificationSmokeError) as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _emit_smoke_artifact(artifact)


def _load_completed_smoke_artifact(
    plan: QualificationSmokePlan,
) -> QualificationSmokeArtifact | None:
    path = plan.smoke_root / WRAPPER_NAME
    if not path.exists():
        return None
    return load_smoke_artifact_file(plan, path)


def _emit_smoke_artifact(artifact: QualificationSmokeArtifact) -> None:
    _ = typer.echo(
        json.dumps(artifact.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )


def _require_matching_smoke_resume(
    first: QualificationSmokeResult,
    resumed: QualificationSmokeResult,
) -> None:
    if resumed != first:
        raise QualificationSmokeError(reason="qualification smoke resume audit mismatch")


@app.command("finalize-config")
def finalize_config(  # noqa: PLR0913, PLR0917
    template: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    target_harness_path: Annotated[Path, typer.Option()],
    prompt_path: Annotated[Path, typer.Option()],
    remote_code_path: Annotated[Path, typer.Option()],
    search_binary_path: Annotated[Path, typer.Option()],
    search_image_digest: Annotated[str, typer.Option()],
    coverage_binary_path: Annotated[Path, typer.Option()],
    coverage_image_digest: Annotated[str, typer.Option()],
    coverage_build_manifest_path: Annotated[Path, typer.Option()],
    c0_source_path: Annotated[Path, typer.Option()],
    dictionary_path: Annotated[Path, typer.Option()],
    artifact_root: Annotated[Path, typer.Option()],
    config_id: Annotated[str, typer.Option()],
    qualification_state: Annotated[str, typer.Option()] = "qualified",
) -> None:
    """Regenerate a campaign config from measured prompt/source/binary artifacts."""
    try:
        payload = _JSON_OBJECT.validate_json(template.read_bytes())
        parsed_state = _parse_qualification_state(qualification_state)
        finalized = finalize_config_payload_from_artifacts(
            payload,
            ConfigArtifactMeasurements(
                prompt_path=prompt_path,
                target_harness_path=target_harness_path,
                remote_code_path=remote_code_path,
                search_binary_path=search_binary_path,
                search_image_digest=search_image_digest,
                coverage_binary_path=coverage_binary_path,
                coverage_image_digest=coverage_image_digest,
                coverage_build_manifest_path=coverage_build_manifest_path,
                c0_source_path=c0_source_path,
                dictionary_path=dictionary_path,
                artifact_root=artifact_root,
                config_id=config_id,
                qualification_state=parsed_state,
            ),
        )
        _write_once_bytes(output, _canonical_json(finalized))
    except (OSError, ValidationError, CoordinatorError) as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _ = typer.echo(json.dumps(finalized, sort_keys=True, separators=(",", ":")))


def _parse_qualification_state(state: str) -> Literal["draft", "qualified"]:
    match state:
        case "draft" | "qualified":
            return state
        case _:
            raise CoordinatorError(reason="invalid qualification state")


def _canonical_json(payload: dict[str, JsonValue]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_once_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CoordinatorError(reason=f"artifact mismatch: {path.name}")
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise CoordinatorError(reason=f"artifact mismatch: {path.name}") from None
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@app.command()
def run(config: Annotated[Path, typer.Option()]) -> None:
    """Run a qualified campaign with production dependency composition."""
    loaded = load_verified_campaign_config(config)
    if loaded.qualification_state != "qualified":
        _ = typer.echo("draft config cannot start the main campaign", err=True)
        raise typer.Exit(2)
    try:
        outcome = run_with_factory(loaded, _production_factory(loaded))
    except CoordinatorError as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _ = typer.echo(
        json.dumps(outcome.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )


@app.command()
def resume(root: Annotated[Path, typer.Option()]) -> None:
    """Validate a campaign root before attempting resume."""
    try:
        config = _load_config_from_root(root)
        outcome = resume_with_factory(root, _production_factory(config))
    except RecoveryError as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    except CoordinatorError as error:
        _ = typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    _ = typer.echo(
        json.dumps(outcome.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )


@app.command()
def status(root: Annotated[Path, typer.Option()]) -> None:
    """Emit deterministic read-only campaign status JSON."""
    result = campaign_status(root)
    _ = typer.echo(
        json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )


def run_with_factory(config: CampaignConfig, factory: _RunOnlyFactory) -> CampaignOutcome:
    """Run through an injected or production one-day runtime factory."""
    if config.qualification_state != "qualified":
        raise CoordinatorError(reason="draft config cannot start the main campaign")
    return factory.run(config)


def resume_with_factory(root: Path, factory: OneDayRuntimeFactory) -> CampaignOutcome:
    """Resume a campaign using dependencies supplied by the runtime factory."""
    config = _load_config_from_root(root)
    with factory.build(config) as graph:
        return resume_campaign(
            root,
            resident_pool=cast("ResidentPoolSurface", cast("object", graph.resident_pool)),
            cell_runner=graph.cell_runner,
            post_search=graph.post_search,
            gate_runner=graph.gate_runner,
            model_executor=graph.model_executor,
            now_monotonic_seconds=time.monotonic(),
        )


def _production_factory(config: CampaignConfig) -> ProductionRuntimeFactory:
    return ProductionRuntimeFactory(
        runtime_python=config.runtime_python,
        source_root=config.artifact_root,
        cache_dir=config.model_cache_dir,
        prompt_path=config.prompt_path,
        qualification_evidence_path=_resolve_qualification_evidence_path(config),
    )


def _resolve_qualification_evidence_path(config: CampaignConfig) -> Path:
    root = config.artifact_root.resolve(strict=False)
    configured = config.qualification_evidence_path
    candidate = configured if configured.is_absolute() else root / configured
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise CoordinatorError(reason="qualification evidence path escapes artifact root")
    return resolved


def _load_config_from_root(root: Path) -> CampaignConfig:
    try:
        payload = _JSON_OBJECT.validate_json((root / "campaign-manifest.json").read_bytes())
        config = _JSON_OBJECT.validate_python(payload["config"])
    except (OSError, KeyError, ValidationError) as error:
        raise CoordinatorError(reason="campaign is not resumable") from error
    digest = config.get("config_sha256")
    if not isinstance(digest, str) or digest != config_payload_sha256(config):
        raise CoordinatorError(reason="config digest mismatch")
    return CampaignConfig.model_validate(config)


if __name__ == "__main__":
    app()
