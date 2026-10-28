"""Functional-repair verdict for an RQ4b candidate: compiles, constructs, inspects, runs.

"It compiles" scored deletion as repair in the RQ4 pilot. This module decides from two
artefacts the compile-only image produces (``tools/harness_synthesis/_functional_eval.py``
runs it) whether a repaired body performs a ``FunctionalSpec``:

- clang's JSON AST of ``LLVMFuzzerTestOneInput``. A call written inside a comment is not
  in the AST at all. A call on some other object does not trace back to the constructed
  one. Each required call is traced through variables to the object constructed from
  ``stream`` (``_Provenance``).
- llvm-cov region counts from one run of the coverage binary on a valid EXR input. A
  call in a dead branch, after an early ``return``, or in a lambda that never runs has
  count 0.

clang's source coverage does not start a new region after a call, so if a call throws,
the code after it still shows count 1. Three conditions close that hole. The fixed
scaffold's ``catch`` block must have count 0 (no ``std::exception`` escaped the body). The
body must contain no ``try`` of its own (which could swallow one). The process must exit
0 (anything else escaping would abort). Together they mean every counted region ran to
completion.

Limitations: it does not check argument values (``file.header(1)`` on a one-part file
throws, so that case is caught by the exception rule, not here), nor what a result is
used for. A benchmark-specific use of a value, such as "the data window's extent appears
in an expression", is not checked.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from dllm_agent.harness_synthesis.repair_benchmark import FunctionalSpec

type AstNode = Mapping[str, object]
type Segment = Sequence[int | bool]

_TRANSPARENT = frozenset(
    {
        "ImplicitCastExpr",
        "MaterializeTemporaryExpr",
        "ExprWithCleanups",
        "ParenExpr",
        "CXXBindTemporaryExpr",
        "ConstantExpr",
        "CXXFunctionalCastExpr",
    }
)
_ENTRY_POINT = "LLVMFuzzerTestOneInput"
_STREAM = "stream"
_MAX_DEPTH = 32


@dataclass(frozen=True, slots=True)
class CallSite:
    """A member call traced to a constructed object: ``chain`` from it, each call's offset."""

    root_var_id: str
    chain: tuple[str, ...]
    call_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Construction:
    """A body variable of the spec's class, and whether it is constructed from ``stream``."""

    var_id: str
    var_name: str
    offset: int
    from_stream: bool


@dataclass(frozen=True, slots=True)
class AstFacts:
    """What ``ast_facts`` extracted from the entry point's AST."""

    constructions: tuple[Construction, ...]
    calls: tuple[CallSite, ...]
    body_has_try: bool
    scaffold_catch_offset: int | None


@dataclass(frozen=True, slots=True)
class FunctionalVerdict:
    """Primary endpoint (``success``) plus the components it was computed from."""

    compiled: bool
    success: bool
    constructed_from_stream: bool = False
    executed_chains: tuple[tuple[str, ...], ...] = ()
    chains: Mapping[str, bool] = field(default_factory=dict[str, bool])
    reasons: tuple[str, ...] = ()


def _map(node: AstNode, key: str) -> AstNode:
    value = node.get(key)
    return cast("AstNode", value) if isinstance(value, Mapping) else {}


def _children(node: AstNode) -> list[AstNode]:
    value = node.get("inner")
    if not isinstance(value, list):
        return []
    return [cast("AstNode", c) for c in cast("list[object]", value) if isinstance(c, Mapping)]


def _text(node: AstNode, key: str) -> str | None:
    value = node.get(key)
    return value if isinstance(value, str) else None


def _offset(node: AstNode) -> int | None:
    begin = _map(_map(node, "range"), "begin")
    for loc in (begin, _map(begin, "expansionLoc"), _map(begin, "spellingLoc")):
        value = loc.get("offset")
        if isinstance(value, int):
            return value
    return None


def _walk(node: AstNode) -> list[AstNode]:
    out: list[AstNode] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(_children(current)))
    return out


def _qual_types(node: AstNode) -> set[str]:
    type_info = _map(node, "type")
    return {
        text.removeprefix("const ").strip()
        for key in ("qualType", "desugaredQualType")
        if (text := _text(type_info, key)) is not None
    }


def entry_point(nodes: Sequence[AstNode]) -> AstNode | None:
    """The ``LLVMFuzzerTestOneInput`` definition (the one with a body), if present."""
    for node in nodes:
        for candidate in _walk(node):
            if (
                _text(candidate, "kind") == "FunctionDecl"
                and _text(candidate, "name") == _ENTRY_POINT
                and any(_text(c, "kind") == "CompoundStmt" for c in _children(candidate))
            ):
                return candidate
    return None


def _strip(node: AstNode) -> AstNode:
    """Skip implicit wrappers and single-argument copy/move constructions."""
    current = node
    while True:
        inner = _children(current)
        kind = _text(current, "kind")
        if (kind in _TRANSPARENT and inner) or (kind == "CXXConstructExpr" and len(inner) == 1):
            current = inner[0]
        else:
            return current


@dataclass(frozen=True, slots=True)
class _Provenance:
    var_decls: Mapping[str, AstNode]
    roots: frozenset[str]

    def resolve(self, node: AstNode, depth: int = 0) -> CallSite | None:  # noqa: PLR0911 - one exit per AST shape.
        if depth > _MAX_DEPTH:
            return None
        expr = _strip(node)
        kind = _text(expr, "kind")
        if kind == "DeclRefExpr":
            var_id = str(_map(expr, "referencedDecl").get("id"))
            if var_id in self.roots:
                return CallSite(var_id, (), ())
            decl = self.var_decls.get(var_id)
            init = [] if decl is None or not decl.get("init") else _children(decl)
            return self.resolve(init[0], depth + 1) if init else None
        inner = _children(expr)
        if kind != "CXXMemberCallExpr" or not inner:
            return None
        member = inner[0]
        receiver = _children(member)
        if _text(member, "kind") != "MemberExpr" or not receiver:
            return None
        base = self.resolve(receiver[0], depth + 1)
        offset = _offset(expr)
        if base is None or offset is None:
            return None
        return CallSite(
            base.root_var_id,
            (*base.chain, str(_text(member, "name"))),
            (*base.call_offsets, offset),
        )


def _constructs_from_stream(init: AstNode) -> bool:
    expr = _strip(init)
    args = _children(expr)
    if _text(expr, "kind") != "CXXConstructExpr" or not args:
        return False
    first = _strip(args[0])
    return (
        _text(first, "kind") == "DeclRefExpr"
        and _text(_map(first, "referencedDecl"), "name") == _STREAM
    )


def _catch_offset(tries: Sequence[AstNode], in_body: Callable[[AstNode], bool]) -> int | None:
    scaffold_try = next((t for t in tries if not in_body(t)), None)
    if scaffold_try is None:
        return None
    for catch in _children(scaffold_try):
        if _text(catch, "kind") == "CXXCatchStmt":
            for handler in _children(catch):
                if _text(handler, "kind") == "CompoundStmt":
                    return _offset(handler)
    return None


def ast_facts(
    nodes: Sequence[AstNode], spec: FunctionalSpec, *, body_start: int, body_end: int
) -> AstFacts:
    """Constructions, traced member calls and ``try`` structure inside ``[body_start, body_end)``.

    Offsets are byte offsets into the compiled harness source, which the caller knows as
    ``len(prefix)`` and ``len(prefix) + len(body)``.
    """
    function = entry_point(nodes)
    if function is None:
        return AstFacts((), (), body_has_try=False, scaffold_catch_offset=None)
    all_nodes = _walk(function)

    def in_body(node: AstNode) -> bool:
        offset = _offset(node)
        return offset is not None and body_start <= offset < body_end

    def kind_is(node: AstNode, kind: str) -> bool:
        return _text(node, "kind") == kind

    var_decls = {str(n.get("id")): n for n in all_nodes if kind_is(n, "VarDecl")}
    constructions = tuple(
        Construction(
            var_id=str(n.get("id")),
            var_name=str(_text(n, "name")),
            offset=_offset(n) or -1,
            from_stream=bool(_children(n)) and _constructs_from_stream(_children(n)[0]),
        )
        for n in all_nodes
        if kind_is(n, "VarDecl") and in_body(n) and spec.object_qualified_type in _qual_types(n)
    )
    provenance = _Provenance(var_decls, frozenset(c.var_id for c in constructions if c.from_stream))
    calls = tuple(
        site
        for n in all_nodes
        if kind_is(n, "CXXMemberCallExpr") and in_body(n)
        if (site := provenance.resolve(n)) is not None
    )
    tries = [n for n in all_nodes if kind_is(n, "CXXTryStmt")]
    return AstFacts(
        constructions=constructions,
        calls=calls,
        body_has_try=any(in_body(t) for t in tries),
        scaffold_catch_offset=_catch_offset(tries, in_body),
    )


def line_col(source: str, offset: int) -> tuple[int, int]:
    """1-based ``(line, column)`` of a byte offset (sources here are ASCII)."""
    line = source.count("\n", 0, offset) + 1
    return line, offset - (source.rfind("\n", 0, offset) + 1) + 1


def coverage_count(segments: Sequence[Segment], line: int, col: int) -> int | None:
    """Execution count at ``(line, col)`` from llvm-cov export segments, ``None`` if unmapped.

    A segment is ``[line, col, count, has_count, is_region_entry, is_gap_region]`` and
    holds until the next one; segments are sorted by position.
    """
    current: Segment | None = None
    for segment in segments:
        if (int(segment[0]), int(segment[1])) <= (line, col):
            current = segment
        else:
            break
    if current is None or not current[3]:
        return None
    return int(current[2])


def evaluate(  # noqa: PLR0913, C901 - the verdict's inputs and its flat list of checks.
    spec: FunctionalSpec,
    *,
    source: str,
    compiled: bool,
    facts: AstFacts | None,
    segments: Sequence[Segment],
    exit_code: int | None,
) -> FunctionalVerdict:
    """Decide functional success for one compiled-and-run candidate."""
    if not compiled or facts is None:
        return FunctionalVerdict(compiled=False, success=False, reasons=("not-compiled",))

    def executed(offset: int) -> bool:
        count = coverage_count(segments, *line_col(source, offset))
        return count is not None and count > 0

    reasons: list[str] = []
    if exit_code != 0:
        reasons.append("nonzero-exit")
    if facts.body_has_try:
        reasons.append("body-contains-try")
    if facts.scaffold_catch_offset is None:
        reasons.append("scaffold-catch-not-found")
    elif executed(facts.scaffold_catch_offset):
        reasons.append("exception-reached-scaffold-catch")
    roots = [c for c in facts.constructions if c.from_stream]
    constructed = any(executed(c.offset) for c in roots)
    if not roots:
        reasons.append("no-construction-from-stream")
    elif not constructed:
        reasons.append("construction-not-executed")

    executed_chains = tuple(
        sorted({site.chain for site in facts.calls if all(executed(o) for o in site.call_offsets)})
    )
    chains: dict[str, bool] = {}
    for chain in spec.required_chains:
        name = ">".join(chain)
        present = any(site.chain == chain for site in facts.calls)
        chains[name] = chain in executed_chains
        if not present:
            reasons.append(f"chain-missing:{name}")
        elif not chains[name]:
            reasons.append(f"chain-not-executed:{name}")
    return FunctionalVerdict(
        compiled=True,
        success=not reasons,
        constructed_from_stream=constructed,
        executed_chains=executed_chains,
        chains=chains,
        reasons=tuple(reasons),
    )
