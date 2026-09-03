"""Pinned Git target verification and non-root-readable archive staging."""

import io
import os
import posixpath
import subprocess
import tarfile
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import read_bytes, sha256_bytes
from dllm_agent.targets.models import HuntManifest, TargetEvidence


@dataclass(frozen=True, slots=True)
class VerifiedTarget:
    """Evidence plus the exact committed archive used for execution."""

    evidence: TargetEvidence
    archive: bytes


def verify_target_archive(source_root: Path, manifest: HuntManifest) -> VerifiedTarget:
    """Verify exact clean Git bytes and return the pinned commit archive."""
    try:
        root = source_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        detail = f"target source does not exist: {source_root}"
        raise HuntSetupError(detail) from error
    if not root.is_dir():
        detail = f"target source is not a directory: {source_root}"
        raise HuntSetupError(detail)
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if commit != manifest.commit_sha:
        detail = "target commit does not match hunt manifest"
        raise HuntSetupError(detail)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        detail = "target worktree is not clean"
        raise HuntSetupError(detail)
    remote = _git(root, "remote", "get-url", "origin").decode().strip()
    if remote != manifest.repository_url:
        detail = "target origin does not match hunt manifest"
        raise HuntSetupError(detail)
    security = read_bytes(root / "SECURITY.md", "target security policy")
    if sha256_bytes(security) != manifest.security_policy_sha256:
        detail = "target SECURITY.md does not match hunt manifest"
        raise HuntSetupError(detail)
    tree_oid = _git(root, "rev-parse", f"{manifest.commit_sha}^{{tree}}").decode().strip()
    archive = _git(root, "archive", "--format=tar", manifest.commit_sha)
    evidence = TargetEvidence(
        repository_url=remote,
        commit_sha=commit,
        tree_oid=tree_oid,
        source_archive_sha256=sha256_bytes(archive),
        security_policy_url=manifest.security_policy_url,
        security_policy_sha256=manifest.security_policy_sha256,
    )
    return VerifiedTarget(evidence=evidence, archive=archive)


@contextmanager
def staged_target(target: VerifiedTarget) -> Generator[Path]:
    """Extract verified committed bytes with permissions readable by UID 65534."""
    if sha256_bytes(target.archive) != target.evidence.source_archive_sha256:
        detail = "target archive changed after verification"
        raise HuntSetupError(detail)
    with tempfile.TemporaryDirectory(prefix="dllm-target-") as directory:
        temp_root = Path(directory)
        temp_root.chmod(0o755)
        source_root = temp_root / "source"
        source_root.mkdir(mode=0o755)
        try:
            with tarfile.open(fileobj=io.BytesIO(target.archive), mode="r:") as archive:
                _validate_members(archive.getmembers())
                archive.extractall(source_root, filter="data")
            _make_world_readable(source_root)
        except HuntSetupError:
            raise
        except (OSError, tarfile.TarError) as error:
            detail = "could not stage verified target archive"
            raise HuntSetupError(detail) from error
        yield source_root


def _validate_members(members: Sequence[tarfile.TarInfo]) -> None:
    by_path = {PurePosixPath(member.name): member for member in members}
    for member_path, member in by_path.items():
        if member.islnk():
            detail = "target archive contains unsafe hard link"
            raise HuntSetupError(detail)
        if not member.issym():
            continue
        link_path = PurePosixPath(member.linkname)
        normalized = PurePosixPath(posixpath.normpath(str(member_path.parent / link_path)))
        target = by_path.get(normalized)
        unsafe_ancestor = any(
            (ancestor_member := by_path.get(ancestor)) is not None
            and (ancestor_member.issym() or ancestor_member.islnk())
            for ancestor in (*member_path.parents[:-1], *normalized.parents[:-1])
        )
        if (
            not member.linkname
            or link_path.is_absolute()
            or normalized == PurePosixPath("..")
            or str(normalized).startswith("../")
            or target is None
            or target.issym()
            or target.islnk()
            or not (target.isfile() or target.isdir())
            or unsafe_ancestor
        ):
            detail = "target archive contains unsafe symbolic link"
            raise HuntSetupError(detail)


def _make_world_readable(root: Path) -> None:
    root.chmod(0o755)
    resolved_root = root.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                detail = f"unsafe target archive link: {path.relative_to(root)}"
                raise HuntSetupError(detail) from error
            if not resolved.is_relative_to(resolved_root):
                detail = f"unsafe target archive link: {path.relative_to(root)}"
                raise HuntSetupError(detail)
        elif path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            executable = path.stat().st_mode & 0o111
            path.chmod(0o755 if executable else 0o644)
        else:
            detail = f"unsupported target archive entry: {path.relative_to(root)}"
            raise HuntSetupError(detail)


def _git(root: Path, *arguments: str) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": "/usr/bin:/bin",
    }
    command = ("git", "-C", str(root), *arguments)
    try:
        result = subprocess.run(  # noqa: S603 - fixed Git executable and argument forms.
            command, capture_output=True, check=False, env=environment, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        detail = "could not inspect target Git repository"
        raise HuntSetupError(detail) from error
    if result.returncode != 0:
        detail = "target Git repository inspection failed"
        raise HuntSetupError(detail)
    return result.stdout
