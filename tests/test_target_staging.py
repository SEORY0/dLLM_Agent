"""Pinned target archive link-boundary tests."""

import io
import tarfile
from pathlib import Path

import pytest

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import sha256_bytes
from dllm_agent.targets.models import TargetEvidence
from dllm_agent.targets.staging import VerifiedTarget, staged_target


def _symlink(name: str, linkname: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.SYMTYPE
    member.linkname = linkname
    return member


def _hardlink(name: str, linkname: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.LNKTYPE
    member.linkname = linkname
    return member


def test_staged_target_preserves_openexr_style_internal_directory_link() -> None:
    # Given
    target = _target(
        (
            _directory("src/test/oss-fuzz"),
            _file("src/test/oss-fuzz/fuzzer.cc", b"x"),
            _directory("src/test/OpenEXRFuzzTest"),
            _symlink("src/test/OpenEXRFuzzTest/oss-fuzz", "../oss-fuzz"),
        )
    )

    # When
    with staged_target(target) as source:
        link = source / "src/test/OpenEXRFuzzTest/oss-fuzz"
        content = (link / "fuzzer.cc").read_bytes()

        # Then
        assert link.is_symlink()
        assert link.readlink() == Path("../oss-fuzz")
        assert content == b"x"


@pytest.mark.parametrize(
    "member",
    [
        _symlink("links/absolute", "/etc/passwd"),
        _symlink("links/escape", "../../outside"),
        _symlink("links/dangling", "missing"),
        _hardlink("links/hard", "real/file"),
    ],
    ids=("absolute", "escape", "dangling", "hardlink"),
)
def test_staged_target_rejects_unsafe_or_unresolved_links(member: tarfile.TarInfo) -> None:
    # Given
    target = _target((_directory("links"), _directory("real"), _file("real/file", b"x"), member))

    # When / Then
    with pytest.raises(HuntSetupError, match="link"), staged_target(target):
        pass


def test_staged_target_rejects_chained_symbolic_link() -> None:
    # Given
    target = _target(
        (
            _directory("links"),
            _directory("real"),
            _file("real/file", b"x"),
            _symlink("links/target", "../real"),
            _symlink("links/chained", "target"),
        )
    )

    # When / Then
    with pytest.raises(HuntSetupError, match="link"), staged_target(target):
        pass


def _target(members: tuple[tarfile.TarInfo, ...]) -> VerifiedTarget:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for member in members:
            archive.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
    raw = stream.getvalue()
    return VerifiedTarget(
        evidence=TargetEvidence(
            repository_url="https://github.com/example/target.git",
            commit_sha="a" * 40,
            tree_oid="b" * 40,
            source_archive_sha256=sha256_bytes(raw),
            security_policy_url="https://github.com/example/target/security/advisories/new",
            security_policy_sha256="c" * 64,
        ),
        archive=raw,
    )


def _directory(name: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    member.mode = 0o755
    return member


def _file(name: str, content: bytes) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = 0o644
    return member
