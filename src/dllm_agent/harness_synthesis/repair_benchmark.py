"""RQ4b controlled single-defect repair benchmark for the fixed-scaffold OpenEXR harness.

The RQ4 pilot repaired whatever the model happened to get wrong. That mixed several
things: how many lines were wrong, whether code (or only a comment) followed the error,
and whether inspection was required at all (the prompt said "optionally"). Here each
sample starts from a known-good body, written by hand and checked against the installed
OpenEXR 3.4 headers. Exactly one line is then broken in a recorded way. The repair target
is the original line, and the task requires the inspection explicitly.

Every reference body:

- constructs the object from ``stream``;
- performs a required inspection chain (``FunctionalSpec.required_chains``);
- has executable code after every mutated line (no mutation targets the last line).

``tools/harness_synthesis/rq4b_validate_benchmark.py`` checks every property that needs a
compiler or a run (each mutant fails with all diagnostics on the mutated line, each
reference passes the functional evaluator) and records the result. Nothing in this module
calls a compiler.

Leakage. The model never sees ``target_line``: the infill and suffix arms mask the
mutated line, and the task prompt (``repair_prompt``) describes the required operations
in words, never as code. ``leaked_fragments`` checks the second half mechanically.
"""

import re
from dataclasses import dataclass
from typing import Literal

type MutationKind = Literal[
    "constructor-argument",
    "invalid-method",
    "argument-count",
    "argument-type",
    "namespace-or-type",
    "missing-call-parentheses",
    "syntax-missing-semicolon",
]

_IDENTIFIER_CALL = re.compile(r"[A-Za-z_]\w*\s*\(|::\s*[A-Za-z_]\w*|[A-Za-z_]\w*\s*::")


@dataclass(frozen=True, slots=True)
class FunctionalSpec:
    """What a repaired body must do, checked from clang's AST plus execution coverage.

    ``required_chains``: each entry is a sequence of method names called on the object
    of type ``class_name`` constructed from ``stream``, either directly or through
    variables initialised from earlier calls in the chain. ``("header", "dataWindow")``
    is satisfied by ``file.header().dataWindow()`` and by ``const Header& h =
    file.header(); h.dataWindow();``. Every call in the chain must actually run on the
    validation input, with no exception reaching the scaffold's catch.
    """

    class_name: str
    object_qualified_type: str
    required_chains: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class Mutation:
    """One controlled defect: replace ``original`` (once, on ``line_number``) by ``mutated``."""

    sample_id: str
    kind: MutationKind
    line_number: int
    original: str
    mutated: str


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """A reference body plus one mutation applied to it."""

    sample_id: str
    example: str
    reference_body: str
    mutation: Mutation

    @property
    def mutated_body(self) -> str:
        """The reference with the mutation applied (exactly one occurrence, checked)."""
        return apply_mutation(self.reference_body, self.mutation)

    @property
    def target_line(self) -> str:
        """The known-good text of the mutated line (the repair target)."""
        return self.reference_body.split("\n")[self.mutation.line_number - 1]

    @property
    def defective_line(self) -> str:
        """The mutated line as the repair arms receive it."""
        return self.mutated_body.split("\n")[self.mutation.line_number - 1]


REFERENCE_BODIES: dict[str, str] = {
    "InputFile": (
        "    OPENEXR_IMF_NAMESPACE::InputFile file(stream);\n"
        "    const OPENEXR_IMF_NAMESPACE::Header& header = file.header();\n"
        "    const IMATH_NAMESPACE::Box2i& dataWindow = header.dataWindow();\n"
        "    static_cast<void>(dataWindow.max.x - dataWindow.min.x);\n"
        "  "
    ),
    "MultiPartInputFile": (
        "    OPENEXR_IMF_NAMESPACE::MultiPartInputFile file(stream);\n"
        "    const int partCount = file.parts();\n"
        "    const OPENEXR_IMF_NAMESPACE::Header& header = file.header(0);\n"
        "    static_cast<void>(partCount + header.dataWindow().max.x);\n"
        "  "
    ),
}

FUNCTIONAL_SPECS: dict[str, FunctionalSpec] = {
    "InputFile": FunctionalSpec(
        class_name="InputFile",
        object_qualified_type="Imf::InputFile",
        required_chains=(("header", "dataWindow"),),
    ),
    "MultiPartInputFile": FunctionalSpec(
        class_name="MultiPartInputFile",
        object_qualified_type="Imf::MultiPartInputFile",
        required_chains=(("parts",), ("header", "dataWindow")),
    ),
}

# Each was compiled against the pinned image before being listed here. Every clang error
# lands on ``line_number`` (rq4b_validate_benchmark.py re-checks this). Rejected
# candidates, kept as a record: a class-name typo (``InputFil``) and an unclosed call
# (``dataWindow(;``) both cascade into errors on other lines, so they are not single-line
# defects. ``IMATH_NAMESPACE::Header`` (for MP line 3) failed the leakage check: the
# correct ``OPENEXR_IMF_NAMESPACE::`` is visible in the prompt and on line 1, so the
# fix can be copied from context. ``PartHeader`` replaced it.
MUTATIONS: tuple[Mutation, ...] = (
    Mutation("IF-ctor-pointer", "constructor-argument", 1, "file(stream)", "file(&stream)"),
    Mutation("IF-bad-method", "invalid-method", 2, "file.header()", "file.readHeader()"),
    Mutation("IF-extra-arg", "argument-count", 2, "file.header()", "file.header(0)"),
    Mutation(
        "IF-box-namespace",
        "namespace-or-type",
        3,
        "const IMATH_NAMESPACE::Box2i&",
        "const OPENEXR_IMF_NAMESPACE::Box2i&",
    ),
    Mutation("IF-no-semicolon", "syntax-missing-semicolon", 1, "file(stream);", "file(stream)"),
    Mutation(
        "IF-uncalled-member",
        "missing-call-parentheses",
        3,
        "header.dataWindow()",
        "header.dataWindow",
    ),
    Mutation("MP-ctor-pointer", "constructor-argument", 1, "file(stream)", "file(&stream)"),
    Mutation("MP-bad-method", "invalid-method", 2, "file.parts()", "file.numParts()"),
    Mutation("MP-missing-arg", "argument-count", 3, "file.header(0)", "file.header()"),
    Mutation(
        "MP-header-type",
        "namespace-or-type",
        3,
        "OPENEXR_IMF_NAMESPACE::Header&",
        "OPENEXR_IMF_NAMESPACE::PartHeader&",
    ),
    Mutation("MP-no-semicolon", "syntax-missing-semicolon", 2, "file.parts();", "file.parts()"),
    Mutation("MP-string-arg", "argument-type", 3, "file.header(0)", 'file.header("0")'),
)

_REQUIREMENTS: dict[str, str] = {
    "InputFile": (
        "(1) construct an OPENEXR_IMF_NAMESPACE::InputFile named `file` that reads from "
        "`stream`; (2) obtain the file's header and, from that header, its data window; "
        "(3) use the data window's extent in an expression so the result is not unused."
    ),
    "MultiPartInputFile": (
        "(1) construct an OPENEXR_IMF_NAMESPACE::MultiPartInputFile named `file` that "
        "reads from `stream`; (2) obtain the number of parts in the file; (3) obtain the "
        "header of the first part and, from it, its data window; (4) use the part count "
        "and the data window in an expression so the results are not unused."
    ),
}


def apply_mutation(body: str, mutation: Mutation) -> str:
    """Apply ``mutation`` to ``body``; the original text must occur exactly once, on its line."""
    lines = body.split("\n")
    target = lines[mutation.line_number - 1]
    if body.count(mutation.original) != 1 or mutation.original not in target:
        reason = f"{mutation.sample_id}: original must occur once, on line {mutation.line_number}"
        raise ValueError(reason)
    lines[mutation.line_number - 1] = target.replace(mutation.original, mutation.mutated)
    return "\n".join(lines)


def samples() -> tuple[BenchmarkSample, ...]:
    """Every benchmark sample, in the frozen order the pilot iterates over."""
    out: list[BenchmarkSample] = []
    for mutation in MUTATIONS:
        example = "InputFile" if mutation.sample_id.startswith("IF-") else "MultiPartInputFile"
        out.append(
            BenchmarkSample(mutation.sample_id, example, REFERENCE_BODIES[example], mutation)
        )
    return tuple(out)


def repair_prompt(example: str, *, diagnostic: str | None = None) -> str:
    """The task prompt every arm gets; ``diagnostic`` adds the compiler-feedback factor.

    Unlike the RQ4 pilot's prompt, inspection is required, not optional. It says nothing
    about the argument form of the constructor: the pilot's "pass it as ``stream``, not
    ``&stream``" would hand the ``*-ctor-pointer`` samples their fix. ``diagnostic`` must
    be the same string for every arm within one condition (see the spec, Phase F).
    """
    prompt = (
        "You are writing the body of a fixed C++ libFuzzer harness for OpenEXR's "
        f"{example}. Inside the try block, an object named `stream` of a class derived "
        "from OPENEXR_IMF_NAMESPACE::IStream already exists. The body must: "
        f"{_REQUIREMENTS[example]} Do not repeat any surrounding code, includes, or the "
        "class definition, and do not write the closing brace. Output only the "
        "statements themselves, each ending in a semicolon."
    )
    if diagnostic is not None:
        prompt += f" A previous version of this body failed to compile with: {diagnostic}"
    return prompt


def _fragments(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(0)) for m in _IDENTIFIER_CALL.finditer(text)}


def leaked_fragments(sample: BenchmarkSample, model_visible_text: str) -> tuple[str, ...]:
    """Code fragments of the repair target that the model could copy from its input.

    The target line itself must never appear (whitespace-normalised). Neither may any
    call- or scope-shaped fragment (``name(``, ``ns::``, ``::name``) that the target has
    but the defective line does not, so that copying it would perform the repair.
    Fragments the target shares with the defective line are not leaks: the model sees
    those anyway, or they are masked in both cases.
    """
    visible = re.sub(r"\s+", "", model_visible_text)
    leaks: list[str] = []
    if re.sub(r"\s+", "", sample.target_line) in visible:
        leaks.append(sample.target_line.strip())
    delta = _fragments(sample.target_line) - _fragments(sample.defective_line)
    leaks.extend(sorted(fragment for fragment in delta if fragment in visible))
    return tuple(leaks)


CONTINUATION_TAIL = "}\n```"


def continuation_text(body: str) -> str:
    """The post-prefix text a remask round starts from: the body, the try block's ``}``, fence.

    It matches what the RQ4 pilot's own generations looked like after the body. The
    infill arm keeps the tail fixed. The suffix arm masks it along with the rest, so the
    model has to close the block itself.
    """
    return body + CONTINUATION_TAIL
