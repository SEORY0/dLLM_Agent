from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

from tests.source_layout_support import (
    ALLOWED_DEPENDENCIES,
    FINAL_RUNTIME_MODULES,
    PACKAGE_ROOT,
    PYPROJECT,
    REMOVED_FLAT_FILE_PATHS,
    REMOVED_IMPORT_SPECS,
    ROOT,
    dependency_edges,
    dependency_violations,
    discover_modules,
    find_stale_flat_imports,
    project_imports,
    repo_sources,
)


def test_current_module_inventory_is_exact_final_layout() -> None:
    # Given / When
    modules = discover_modules()

    # Then
    assert modules == FINAL_RUNTIME_MODULES
    assert REMOVED_IMPORT_SPECS.isdisjoint(modules)
    assert root_python_files() == ("__init__.py", "__main__.py", "cli.py")


def test_runtime_import_sweep_imports_every_module_except_main() -> None:
    # Given
    modules = sorted(FINAL_RUNTIME_MODULES - {"dllm_agent.__main__"})

    # When
    imported = {importlib.import_module(module).__name__ for module in modules}

    # Then
    assert imported == set(modules)


def test_root_cli_and_web_entrypoints_resolve_without_executing_main() -> None:
    # Given / When / Then
    assert importlib.util.find_spec("dllm_agent") is not None
    assert importlib.util.find_spec("dllm_agent.cli") is not None
    assert importlib.util.find_spec("dllm_agent.__main__") is not None
    assert importlib.util.find_spec("dllm_agent.web.app") is not None
    assert 'dllm-agent = "dllm_agent.cli:app"' in PYPROJECT.read_text(encoding="utf-8")
    assert "from dllm_agent.cli import app" in (PACKAGE_ROOT / "__main__.py").read_text(
        encoding="utf-8"
    )


def test_removed_flat_modules_and_files_are_absent() -> None:
    # Given / When
    missing_flat_files = tuple(path for path in sorted(REMOVED_FLAT_FILE_PATHS) if path.exists())
    resolved_old_specs = tuple(
        module for module in sorted(REMOVED_IMPORT_SPECS) if importlib.util.find_spec(module)
    )

    # Then
    assert missing_flat_files == ()
    assert resolved_old_specs == ()
    assert importlib.util.find_spec("dllm_agent.hunt") is not None
    assert not (PACKAGE_ROOT / "hunt.py").exists()


def test_stale_flat_import_detector_reports_exact_module(tmp_path: Path) -> None:
    # Given
    source_path = tmp_path / "stale_importer.py"
    stale_module = "dllm_agent." + "hunt_models"
    source = (
        "stale_importer",
        source_path,
        (
            "from dllm_agent import hunt\n"
            f"from dllm_agent import {stale_module.removeprefix('dllm_agent.')}\n"
            "import dllm_agent.hunt.models\n"
            f"import {stale_module}\n"
        ),
    )

    # When
    stale_imports = find_stale_flat_imports((source,))

    # Then
    assert stale_imports == (
        (source_path, 2, stale_module),
        (source_path, 4, stale_module),
    )


def test_no_stale_flat_imports_in_src_tests_or_tools() -> None:
    # Given / When / Then
    assert find_stale_flat_imports(repo_sources()) == ()


def test_package_dependency_direction_matches_final_matrix() -> None:
    # Given
    relative_source = (
        "dllm_agent.hunt.pipeline",
        Path("hunt/pipeline.py"),
        "from ..native.models import NativeHuntReport\nfrom .models import HuntReport\n",
    )
    sources = (
        relative_source,
        (
            "dllm_agent.targets.staging",
            Path("targets/staging.py"),
            "from dllm_agent.hunt.models import HuntReport\n",
        ),
        (
            "dllm_agent.fuzzing.execution",
            Path("fuzzing/execution.py"),
            (
                "from dllm_agent.hunt.models import HuntReport\n"
                "from dllm_agent.native.models import NativeHuntReport\n"
            ),
        ),
    )

    # When / Then
    assert project_imports(relative_source[0], relative_source[2]) == (
        (1, "dllm_agent.native.models"),
        (2, "dllm_agent.hunt.models"),
    )
    assert ("hunt", "targets") in dependency_edges(repo_sources())
    assert ("targets", "hunt") not in dependency_edges(repo_sources())
    assert dependency_violations(repo_sources()) == ()
    assert dependency_violations(sources) == (
        (
            Path("hunt/pipeline.py"),
            "dllm_agent.hunt.pipeline",
            "dllm_agent.native.models",
            "hunt",
            "native",
        ),
        (
            Path("targets/staging.py"),
            "dllm_agent.targets.staging",
            "dllm_agent.hunt.models",
            "targets",
            "hunt",
        ),
        (
            Path("fuzzing/execution.py"),
            "dllm_agent.fuzzing.execution",
            "dllm_agent.hunt.models",
            "fuzzing",
            "hunt",
        ),
        (
            Path("fuzzing/execution.py"),
            "dllm_agent.fuzzing.execution",
            "dllm_agent.native.models",
            "fuzzing",
            "native",
        ),
    )
    assert set(ALLOWED_DEPENDENCIES) == {
        "core",
        "replay",
        "targets",
        "fuzzing",
        "hunt",
        "runtime",
        "native",
        "revision",
        "experiments",
        "web",
        "cli",
    }


def test_source_layout_docs_and_active_poc_plan_are_final() -> None:
    # Given
    layout_doc = (ROOT / "docs/source-layout.md").read_text(encoding="utf-8")
    active_plan = (
        ROOT / "docs/superpowers/plans/2026-08-20-feedback-conditioned-poc-revision.md"
    ).read_text(encoding="utf-8")

    # When / Then
    assert "```mermaid" in layout_doc
    assert "PDF" not in layout_doc
    assert "DOCX" not in layout_doc
    assert "src/dllm_agent/targets/openexr/probe.py" in active_plan
    assert "src/dllm_agent/" + "poc_probe.py" not in active_plan
    assert "src/dllm_agent/" + "openexr_particle.py" not in active_plan
    assert "src/dllm_agent/" + "poc_revision_models.py" not in active_plan


def test_package_init_files_are_docstring_only() -> None:
    # Given / When
    violations = tuple(
        violation
        for path in sorted(PACKAGE_ROOT.rglob("__init__.py"))
        for violation in init_executable_statements(path)
    )

    # Then
    assert violations == ()


def root_python_files() -> tuple[str, ...]:
    return tuple(path.name for path in sorted(PACKAGE_ROOT.glob("*.py")) if path.is_file())


def init_executable_statements(path: Path) -> tuple[tuple[Path, int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    first_executable_index = 1 if ast.get_docstring(tree) is not None else 0
    return tuple(
        (path, node.lineno, type(node).__name__) for node in tree.body[first_executable_index:]
    )
