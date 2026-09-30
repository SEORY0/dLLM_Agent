from __future__ import annotations

import ast
import pkgutil
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[1]
SRC_ROOT: Final = ROOT / "src"
PACKAGE_ROOT: Final = SRC_ROOT / "dllm_agent"
PYPROJECT: Final = ROOT / "pyproject.toml"

type PythonSource = tuple[str, Path, str]
type ImportHit = tuple[Path, int, str]
type DependencyViolation = tuple[Path, str, str, str, str]


def module_set(raw: str) -> frozenset[str]:
    return frozenset(raw.split())


def grouped_sets(raw: str) -> dict[str, frozenset[str]]:
    groups: dict[str, frozenset[str]] = {}
    for row in raw.strip().splitlines():
        key, values = row.split(":", maxsplit=1)
        owner = key.strip()
        groups[owner] = groups.get(owner, frozenset()) | module_set(values)
    return groups


REMOVED_FLAT_FILES_BY_OWNER: Final = grouped_sets(
    """
    core: errors.py io.py models.py
    replay: pipeline.py sampler.py scoring.py validator.py
    targets: target_profiles.py target_staging.py openexr_particle.py
    fuzzing: fuzz_types.py fuzz_execution.py fuzz_runner.py
    fuzzing: coverage_types.py coverage_execution.py coverage_runner.py
    hunt: hunt.py hunt_models.py hunt_artifacts.py hunt_scoring.py hunt_pipeline.py
    runtime: live_sampler.py
    native: native_artifacts.py native_models.py native_pipeline.py native_scoring.py
    revision: poc_revision_artifacts.py poc_revision_command.py
    revision: poc_revision_models.py poc_revision_policy.py
    """
)
REMOVED_FLAT_MODULES_BY_OWNER: Final = {
    owner: frozenset(f"dllm_agent.{Path(file_name).stem}" for file_name in file_names)
    for owner, file_names in REMOVED_FLAT_FILES_BY_OWNER.items()
}
REMOVED_FLAT_MODULES: Final = frozenset(
    module for modules in REMOVED_FLAT_MODULES_BY_OWNER.values() for module in modules
)
REMOVED_IMPORT_SPECS: Final = REMOVED_FLAT_MODULES - {"dllm_agent.hunt"}
REMOVED_FLAT_FILE_PATHS: Final = frozenset(
    PACKAGE_ROOT / file_name
    for file_names in REMOVED_FLAT_FILES_BY_OWNER.values()
    for file_name in file_names
)
OWNER_BY_REMOVED_MODULE: Final = {
    module: owner for owner, modules in REMOVED_FLAT_MODULES_BY_OWNER.items() for module in modules
}

FINAL_RUNTIME_MODULES: Final = module_set(
    """
    dllm_agent dllm_agent.__main__ dllm_agent.cli
    dllm_agent.core dllm_agent.core.errors dllm_agent.core.io dllm_agent.core.models
    dllm_agent.experiments
    dllm_agent.experiments.multitarget
    dllm_agent.experiments.multitarget.ar_slot_sampler
    dllm_agent.experiments.multitarget.artifacts
    dllm_agent.experiments.multitarget.bank_generation
    dllm_agent.experiments.multitarget.budget
    dllm_agent.experiments.multitarget.campaign_coordinator
    dllm_agent.experiments.multitarget.campaign_core
    dllm_agent.experiments.multitarget.campaign_evidence
    dllm_agent.experiments.multitarget.campaign_execution
    dllm_agent.experiments.multitarget.campaign_preparation
    dllm_agent.experiments.multitarget.campaign_recording
    dllm_agent.experiments.multitarget.campaign_recovery
    dllm_agent.experiments.multitarget.candidate_bank
    dllm_agent.experiments.multitarget.container_monitor
    dllm_agent.experiments.multitarget.corpus
    dllm_agent.experiments.multitarget.docker_runtime
    dllm_agent.experiments.multitarget.ledger
    dllm_agent.experiments.multitarget.model_adapter
    dllm_agent.experiments.multitarget.probe_runner
    dllm_agent.experiments.multitarget.proof
    dllm_agent.experiments.multitarget.proof_batch
    dllm_agent.experiments.multitarget.schedule
    dllm_agent.experiments.multitarget.slot_sampler
    dllm_agent.experiments.multitarget.slot_sampler_contracts
    dllm_agent.experiments.multitarget.slot_sampler_utils
    dllm_agent.experiments.openexr_one_day
    dllm_agent.experiments.openexr_one_day.analysis
    dllm_agent.experiments.openexr_one_day.c0_source
    dllm_agent.experiments.openexr_one_day.candidate_bank
    dllm_agent.experiments.openexr_one_day.cell_runner
    dllm_agent.experiments.openexr_one_day.config_digest
    dllm_agent.experiments.openexr_one_day.coordinator
    dllm_agent.experiments.openexr_one_day.corpus_seed
    dllm_agent.experiments.openexr_one_day.coverage_series
    dllm_agent.experiments.openexr_one_day.evidence_backing
    dllm_agent.experiments.openexr_one_day.failure_taxonomy
    dllm_agent.experiments.openexr_one_day.model_executor
    dllm_agent.experiments.openexr_one_day.models
    dllm_agent.experiments.openexr_one_day.post_search
    dllm_agent.experiments.openexr_one_day.proof_batch
    dllm_agent.experiments.openexr_one_day.qualification
    dllm_agent.experiments.openexr_one_day.qualification_banks
    dllm_agent.experiments.openexr_one_day.qualification_collectors
    dllm_agent.experiments.openexr_one_day.qualification_runtime
    dllm_agent.experiments.openexr_one_day.qualification_smoke
    dllm_agent.experiments.openexr_one_day.qualification_smoke_models
    dllm_agent.experiments.openexr_one_day.qualification_smoke_production
    dllm_agent.experiments.openexr_one_day.recovery
    dllm_agent.experiments.openexr_one_day.resident_evidence
    dllm_agent.experiments.openexr_one_day.resident_pool
    dllm_agent.experiments.openexr_one_day.runtime
    dllm_agent.experiments.openexr_one_day.schedule
    dllm_agent.harness_synthesis
    dllm_agent.harness_synthesis.api_knowledge
    dllm_agent.harness_synthesis.audit_ledger
    dllm_agent.harness_synthesis.fitness
    dllm_agent.harness_synthesis.power_calc
    dllm_agent.harness_synthesis.regions
    dllm_agent.harness_synthesis.remask
    dllm_agent.harness_synthesis.token_trace
    dllm_agent.harness_synthesis.trajectory_hook
    dllm_agent.replay dllm_agent.replay.pipeline dllm_agent.replay.sampler
    dllm_agent.replay.scoring dllm_agent.replay.validator
    dllm_agent.targets dllm_agent.targets.models dllm_agent.targets.profiles
    dllm_agent.targets.staging
    dllm_agent.targets.libpng dllm_agent.targets.libpng.particle
    dllm_agent.targets.libtiff dllm_agent.targets.libtiff.particle
    dllm_agent.targets.libxml2 dllm_agent.targets.libxml2.particle
    dllm_agent.targets.openexr dllm_agent.targets.openexr.deep_parse
    dllm_agent.targets.openexr.particle dllm_agent.targets.openexr.probe
    dllm_agent.targets.openexr.probe_coverage dllm_agent.targets.openexr.proof
    dllm_agent.targets.openexr.proof_execution
    dllm_agent.targets.openexr.harness_reference dllm_agent.targets.structured
    dllm_agent.fuzzing dllm_agent.fuzzing.contracts dllm_agent.fuzzing.execution
    dllm_agent.fuzzing.runner dllm_agent.fuzzing.coverage
    dllm_agent.fuzzing.coverage.contracts dllm_agent.fuzzing.coverage.execution
    dllm_agent.fuzzing.coverage.runner
    dllm_agent.hunt dllm_agent.hunt.validation dllm_agent.hunt.models
    dllm_agent.hunt.artifacts dllm_agent.hunt.scoring dllm_agent.hunt.pipeline
    dllm_agent.runtime dllm_agent.runtime.ar_model_loader dllm_agent.runtime.live_sampler
    dllm_agent.runtime.revision_checks
    dllm_agent.runtime.revision_client dllm_agent.runtime.revision_process
    dllm_agent.native dllm_agent.native.artifacts dllm_agent.native.models
    dllm_agent.native.pipeline dllm_agent.native.scoring
    dllm_agent.revision dllm_agent.revision.archive dllm_agent.revision.artifacts
    dllm_agent.revision.command dllm_agent.revision.controller_helpers
    dllm_agent.revision.controller_models dllm_agent.revision.failure_registry
    dllm_agent.revision.models dllm_agent.revision.pipeline dllm_agent.revision.policy
    dllm_agent.revision.portfolio dllm_agent.revision.proof_archive
    dllm_agent.revision.search_tree
    dllm_agent.revision.trajectory_policy
    dllm_agent.web dllm_agent.web.app dllm_agent.web.catalog dllm_agent.web.contracts
    dllm_agent.web.jobs dllm_agent.web.processes
    """
)
ALLOWED_DEPENDENCIES: Final = grouped_sets(
    """
    core: core
    replay: core replay
    targets: core experiments targets harness_synthesis
    fuzzing: core targets fuzzing
    hunt: core targets fuzzing hunt
    runtime: core experiments hunt runtime revision targets
    native: core targets fuzzing hunt native
    revision: core targets revision
    experiments: core experiments revision runtime targets
    harness_synthesis: core harness_synthesis
    web: core hunt web
    cli: cli core experiments replay targets fuzzing hunt runtime native revision web
    """
)


def discover_modules() -> frozenset[str]:
    modules = pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="dllm_agent.")
    return frozenset({"dllm_agent", *(module.name for module in modules)})


def repo_sources() -> tuple[PythonSource, ...]:
    roots = (SRC_ROOT, ROOT / "tests", ROOT / "tools")
    return tuple(
        (module_name(path), path, path.read_text(encoding="utf-8"))
        for root in roots
        if root.exists()
        for path in sorted(root.rglob("*.py"))
    )


def find_stale_flat_imports(
    sources: tuple[PythonSource, ...],
    forbidden_modules: frozenset[str] = REMOVED_IMPORT_SPECS,
) -> tuple[ImportHit, ...]:
    return tuple(
        (source[1], line, module)
        for source in sources
        for line, module in project_imports(source[0], source[2])
        if module in forbidden_modules
    )


def dependency_violations(
    sources: tuple[PythonSource, ...],
) -> tuple[DependencyViolation, ...]:
    violations: list[DependencyViolation] = []
    for source_module, path, text in sources:
        source_owner = module_owner(source_module)
        if source_owner is None:
            continue
        for _line, imported_module in project_imports(source_module, text):
            imported_owner = module_owner(imported_module)
            if imported_owner is None:
                continue
            if imported_owner not in ALLOWED_DEPENDENCIES[source_owner]:
                violations.append(
                    (path, source_module, imported_module, source_owner, imported_owner)
                )
    return tuple(violations)


def dependency_edges(sources: tuple[PythonSource, ...]) -> frozenset[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source_module, _path, text in sources:
        source_owner = module_owner(source_module)
        if source_owner is None:
            continue
        for _line, imported_module in project_imports(source_module, text):
            imported_owner = module_owner(imported_module)
            if imported_owner is not None and source_owner != imported_owner:
                edges.add((source_owner, imported_owner))
    return frozenset(edges)


def project_imports(source_module: str, text: str) -> tuple[tuple[int, str], ...]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imports.extend(
                (node.lineno, alias.name) for alias in node.names if is_project_module(alias.name)
            )
            continue
        if isinstance(node, ast.ImportFrom):
            imports.extend(imports_from_node(source_module, node))
    return tuple(imports)


def imports_from_node(source_module: str, node: ast.ImportFrom) -> tuple[tuple[int, str], ...]:
    if node.level == 0 and node.module == "dllm_agent":
        return tuple(
            (node.lineno, f"dllm_agent.{alias.name}") for alias in node.names if alias.name != "*"
        )
    if node.level == 0 and node.module is not None and is_project_module(node.module):
        return ((node.lineno, node.module),)
    if node.level > 0:
        resolved = resolve_relative_import(source_module, node.level, node.module)
        if node.module is None:
            return tuple(
                (node.lineno, f"{resolved}.{alias.name}")
                for alias in node.names
                if alias.name != "*"
            )
        if is_project_module(resolved):
            return ((node.lineno, resolved),)
    return ()


def module_owner(module: str) -> str | None:
    if module in {"dllm_agent.cli", "dllm_agent.__main__"}:
        return "cli"
    if module in {"dllm_agent", "dllm_agent.core"}:
        return "core"
    if module in REMOVED_IMPORT_SPECS:
        return OWNER_BY_REMOVED_MODULE[module]
    if not module.startswith("dllm_agent."):
        return None
    package = module.removeprefix("dllm_agent.").split(".", maxsplit=1)[0]
    return package if package in ALLOWED_DEPENDENCIES else None


def module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT if path.is_relative_to(SRC_ROOT) else ROOT)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative_import(source_module: str, level: int, module: str | None) -> str:
    package_parts = source_package(source_module).split(".")
    prefix = package_parts[: len(package_parts) - level + 1]
    if module is not None:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def source_package(source_module: str) -> str:
    module_path = SRC_ROOT / Path(*source_module.split("."))
    if (module_path / "__init__.py").exists():
        return source_module
    return source_module.rsplit(".", 1)[0]


def is_project_module(module: str) -> bool:
    return module == "dllm_agent" or module.startswith("dllm_agent.")
