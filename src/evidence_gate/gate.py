"""Exact-substring verification of model-supplied evidence.

The gate answers one question and refuses to answer any other: does every
quote a model attached to a finding appear, byte for byte, in text the host
actually showed it?

It never judges whether a finding is correct, important, or well written.
It has no model, no network, and no notion of code quality.  It decides
admissibility, not truth.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "EvidenceSource",
    "Finding",
    "GateError",
    "REASONS",
    "RECEIPT_SCHEMA",
    "check",
]

RECEIPT_SCHEMA = "evidence-gate-receipt-v1"

#: Every way the gate can refuse.  The set is closed: a caller may branch on
#: these names, and a new release that adds one is a breaking change.
REASONS = frozenset(
    {
        "finding_not_object",
        "finding_key_set_mismatch",
        "finding_text_invalid",
        "requirement_not_exact_quote",
        "evidence_too_short",
        "evidence_too_long",
        "absence_pattern_too_short",
        "absence_pattern_too_broad",
        "absence_pattern_present",
        "evidence_covers_whole_source",
        "evidence_kind_not_allowed",
        "evidence_not_in_context",
        "evidence_line_endings_differ",
    }
)

#: A quote is a pointer into text, not a copy of it.  Both bounds exist
#: because a substring test has a degenerate escape at each end: a quote of
#: three characters matches by accident, and a quote of the entire file
#: matches by construction.  A model that pastes the whole source back has
#: shown nothing about where the defect is.
DEFAULT_MAX_EVIDENCE_CHARS = 2000
DEFAULT_MAX_SOURCE_FRACTION = 0.5

#: An absence claim has the same two degenerate escapes as a quote, mirrored.
#: A pattern of one character is absent from almost nothing, so a floor exists.
#: A pattern as long as the file is absent from it *by construction* -- any
#: invented sentence of seventy characters is missing from an eighty-eight byte
#: file whether or not the requirement is met -- so a ceiling exists too.  Both
#: were found by measurement: the ceiling after three local models produced
#: patterns of 70, 57 and 35 characters against that very file.
DEFAULT_MAX_ABSENCE_SOURCE_FRACTION = 0.5

#: The fraction above was fitted to an 88-byte file, and a fraction dissolves
#: as the subject grows: against two real modules of ~9,000 characters each it
#: permits an absence pattern of 4,500 characters.  A 70-character invented
#: statement is absent by construction from a file of any size, so the ceiling
#: that actually does the work at realistic scale is an absolute one.  Both
#: apply; the binding one is whichever is smaller.
#:
#: Forty characters is roughly where "a name or a call" ends and "a statement
#: I wrote myself" begins: ``record(``, ``raise ValueError`` and
#: ``audit.log_withdrawal(account, amount)`` all fit, while a two-line invented
#: block does not.
#:
#: Note where the two bounds cross.  The fraction permits ``0.5 * len(source)``
#: and the cap permits 40, so the fraction is the binding one only for sources
#: shorter than 80 characters.  Above that -- which is all real code -- the cap
#: does the work alone.  The fraction is kept because it is the only thing
#: standing between a 39-character pattern and a 40-character file, and that
#: degenerate case is exactly the kind this library exists to refuse.
DEFAULT_MAX_ABSENCE_PATTERN_CHARS = 40

REQUIRED_KEYS = frozenset({"requirement", "evidence", "detail"})


@dataclass(frozen=True)
class EvidenceSource:
    """One block of text the host showed the model.

    ``kind`` is a caller-chosen label used for scoping.  It lets a caller say
    "in this review part, only text of kind ``source`` may be quoted", which
    is how a model is prevented from answering a question about source code
    by quoting something else it was shown.
    """

    name: str
    kind: str
    text: str


@dataclass(frozen=True)
class Finding:
    """A model's allegation plus the quote it offered as support."""

    requirement: str
    evidence: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "requirement": self.requirement,
            "evidence": self.evidence,
            "detail": self.detail,
        }


class GateError(Exception):
    """A refusal.

    ``diagnostic`` is content-free by construction: it carries a reason, an
    index, hashes and byte counts, never the rejected text.  This matters
    because a rejected quote is frequently fabricated text, and a log that
    echoes it turns the log into an unverified-claim store.
    """

    def __init__(self, reason: str, **diagnostic: object) -> None:
        if reason not in REASONS:
            raise AssertionError(f"undeclared gate reason: {reason}")
        super().__init__(reason)
        self.reason = reason
        self.diagnostic: dict[str, object] = {"reason": reason, **diagnostic}


def _digest(text: str) -> dict[str, object]:
    raw = text.encode("utf-8")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _field(prefix: str, text: str) -> dict[str, object]:
    measured = _digest(text)
    return {
        f"{prefix}_sha256": measured["sha256"],
        f"{prefix}_bytes": measured["bytes"],
    }


def _clauses(requirements: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(requirements, str):
        return (requirements,)
    clauses = tuple(requirements)
    if not all(isinstance(clause, str) for clause in clauses):
        raise TypeError("requirements must be str or a sequence of str")
    return clauses


def _states(
    requirement: str, requirements: str | Sequence[str], whole_clause: bool
) -> bool:
    """Does the caller's requirement list actually contain this requirement?

    With ``whole_clause`` the quoted requirement must equal one clause.  A
    plain substring test over a joined brief lets a model quote a fragment
    ("A balance must never be") or a span crossing two unrelated rules, and
    then argue against something nobody wrote.
    """
    clauses = _clauses(requirements)
    if whole_clause:
        return requirement in clauses
    return any(requirement in clause for clause in clauses)


def _one_newline(text: str) -> str:
    """Line endings unified, for telling a platform artefact from invention.

    Used ONLY to pick a refusal reason, never to admit anything.  A quote that
    needs this to match is still refused; the gate just stops calling it
    fabrication.

    Why it exists: a repository checked out on Windows has CRLF line endings,
    a model that reflows its answer emits LF, and every multi-line quote then
    failed with ``evidence_not_in_context`` -- the same reason as a quote of
    code that was never shown.  That is this project's own central error
    committed against itself: reporting a transport artefact as a claim about
    the model.  The README argues at length that a gate reporting *format*
    when you asked about *evidence* is measuring the harness, and this gate
    was doing exactly that on any CRLF source.

    Lone ``\\r`` is folded too.  It is rare, it costs nothing here, and a
    refusal reason that is right for two of the three conventions would be a
    worse trap than no special case at all.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _offsets(haystack: str, needle: str) -> list[int]:
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def check(
    findings: Sequence[object],
    *,
    requirements: str,
    sources: Sequence[EvidenceSource],
    allowed_kinds: frozenset[str] | set[str] | None = None,
    min_evidence_chars: int = 8,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_source_fraction: float | None = DEFAULT_MAX_SOURCE_FRACTION,
    absence_label: str | None = None,
    absence_prefix: str | None = None,
    min_absence_pattern_chars: int = 3,
    max_absence_source_fraction: float | None = DEFAULT_MAX_ABSENCE_SOURCE_FRACTION,
    max_absence_pattern_chars: int | None = DEFAULT_MAX_ABSENCE_PATTERN_CHARS,
    require_whole_clause: bool = False,
) -> dict[str, object]:
    """Admit or refuse a batch of findings.  Fail-closed on the first refusal.

    Returns a content-free receipt.  Raises :class:`GateError` otherwise.

    The gate is deliberately unforgiving in three ways.  It compares exactly,
    with no normalisation of whitespace, case, quotes or Unicode form: a model
    that cannot reproduce a quote exactly has not read it.  It stops at the
    first refusal rather than collecting every problem, because an answer with
    one inadmissible quote is already inadmissible.  And it treats a short
    quote as no quote at all, because a quote of two or three characters
    matches some source by accident, which makes the check meaningless.
    """
    if min_evidence_chars < 1:
        raise ValueError("min_evidence_chars must be at least 1")
    if max_evidence_chars < min_evidence_chars:
        raise ValueError("max_evidence_chars must not be below min_evidence_chars")
    if max_source_fraction is not None and not 0 < max_source_fraction <= 1:
        raise ValueError("max_source_fraction must be in (0, 1]")
    if isinstance(requirements, (str, list, tuple)) is False:
        raise TypeError("requirements must be str or a sequence of str")
    clauses = _clauses(requirements)
    if absence_prefix is not None and not absence_prefix:
        raise ValueError("absence_prefix must not be empty")
    if min_absence_pattern_chars < 1:
        raise ValueError("min_absence_pattern_chars must be at least 1")
    if max_absence_source_fraction is not None and not 0 < max_absence_source_fraction <= 1:
        raise ValueError("max_absence_source_fraction must be in (0, 1]")
    if (
        max_absence_pattern_chars is not None
        and max_absence_pattern_chars < min_absence_pattern_chars
    ):
        raise ValueError(
            "max_absence_pattern_chars must not be below min_absence_pattern_chars"
        )

    scoped = [
        source
        for source in sources
        if allowed_kinds is None or source.kind in allowed_kinds
    ]
    out_of_scope = [source for source in sources if source not in scoped]

    rows: list[dict[str, object]] = []
    for index, raw in enumerate(findings):
        finding = raw.as_dict() if isinstance(raw, Finding) else raw
        if not isinstance(finding, dict):
            raise GateError("finding_not_object", index=index)

        present = frozenset(finding)
        if present != REQUIRED_KEYS:
            raise GateError(
                "finding_key_set_mismatch",
                index=index,
                missing_keys=sorted(REQUIRED_KEYS - present),
                unexpected_keys=sorted(present - REQUIRED_KEYS),
            )
        if any(
            not isinstance(finding[key], str) or not finding[key].strip()
            for key in sorted(REQUIRED_KEYS)
        ):
            raise GateError("finding_text_invalid", index=index)

        requirement = finding["requirement"]
        evidence = finding["evidence"]

        # A model may only allege against a requirement the caller actually
        # wrote.  Paraphrasing a requirement is how a review drifts into
        # reviewing a rule nobody asked for.  Checked before anything else,
        # because it applies to every form a finding can take.
        if not _states(requirement, requirements, require_whole_clause):
            raise GateError(
                "requirement_not_exact_quote",
                index=index,
                **_field("requirement", requirement),
            )

        # The strong form of an absence claim.  Instead of asserting that
        # something is missing, the model names the literal text it says is
        # absent, and the gate searches for it.  Zero occurrences is a
        # verified observation; one occurrence refutes the model.  This turns
        # the one claim the gate previously had to take on trust into the one
        # claim it can settle without reading the code.
        if absence_prefix is not None and evidence.startswith(absence_prefix):
            pattern = evidence[len(absence_prefix) :]
            # Deliberately a lower floor than a quote's.  The two bounds
            # guard against opposite failures: a short *quote* matches by
            # accident and proves nothing, whereas a short *pattern* is
            # harder to be absent, so its absence is the stronger claim.
            # Applying the quote floor here would reject ``record(`` -- seven
            # characters and exactly the right thing to search for.
            if len(pattern) < min_absence_pattern_chars:
                raise GateError(
                    "absence_pattern_too_short",
                    index=index,
                    pattern_chars=len(pattern),
                    min_absence_pattern_chars=min_absence_pattern_chars,
                    **_field("evidence", evidence),
                )
            # Search before measuring breadth.  If the pattern is there, the
            # model is refuted outright, and that is a stronger and more useful
            # thing to report than "your pattern was too broad to be
            # informative".  Refutation first, quality second.
            found = [
                {"source": source.name, "kind": source.kind, "offset": offset}
                for source in scoped
                for offset in _offsets(source.text, pattern)
            ]
            if found:
                raise GateError(
                    "absence_pattern_present",
                    index=index,
                    occurrences=len(found),
                    sources=sorted({item["source"] for item in found}),
                    **_field("evidence", evidence),
                )
            # Two ceilings, because one of them stops working at scale.  The
            # fraction is measured against the *smallest* source searched --
            # the claim is "absent from everything you showed me", and it is
            # only as strong as its weakest search -- and it is what catches a
            # pattern the size of a small file.  The absolute cap is what still
            # works when the file is 9,000 characters and every fraction is
            # negligible.  Whichever binds first, binds.
            smallest = (
                min(max(len(source.text), 1) for source in scoped) if scoped else None
            )
            breadth = round(len(pattern) / smallest, 4) if smallest else None
            over_fraction = (
                max_absence_source_fraction is not None
                and breadth is not None
                and breadth > max_absence_source_fraction
            )
            over_absolute = (
                max_absence_pattern_chars is not None
                and len(pattern) > max_absence_pattern_chars
            )
            if over_fraction or over_absolute:
                raise GateError(
                    "absence_pattern_too_broad",
                    index=index,
                    pattern_chars=len(pattern),
                    smallest_source_chars=smallest,
                    source_fraction=breadth,
                    max_absence_source_fraction=max_absence_source_fraction,
                    max_absence_pattern_chars=max_absence_pattern_chars,
                    # Which bound bit matters to the caller: the fraction says
                    # "too big for this file", the absolute says "you wrote a
                    # statement instead of naming a thing".
                    exceeded=sorted(
                        bound
                        for bound, hit in (
                            ("source_fraction", over_fraction),
                            ("pattern_chars", over_absolute),
                        )
                        if hit
                    ),
                    **_field("evidence", evidence),
                )
            rows.append(
                {
                    "index": index,
                    "quoted": False,
                    "verified_absence": True,
                    "absent_pattern_chars": len(pattern),
                    "absent_pattern_fraction": breadth,
                    "searched_sources": [source.name for source in scoped],
                    **_field("requirement", requirement),
                    **_field("evidence", evidence),
                    **_field("detail", finding["detail"]),
                    "match_count": 0,
                    "matches": [],
                }
            )
            continue

        # Quotation cannot express absence.  A requirement such as "every
        # withdrawal must be recorded" is violated by code that is missing,
        # and missing code has no quotable location.  Measured on
        # devstral:24b: of three findings, the one about present code carried
        # a correct 25-character quote and the two about missing code carried
        # an empty one.  Without a way to say "nothing to quote", a gate
        # scores that as model failure when it is a gate limitation.
        #
        # So absence is expressible, but only when the caller enables it, only
        # via an exact declared label, and it is counted separately in the
        # receipt.  An absence finding is a lead, never evidence.
        if absence_label is not None and evidence == absence_label:
            rows.append(
                {
                    "index": index,
                    "quoted": False,
                    "verified_absence": False,
                    **_field("requirement", requirement),
                    **_field("evidence", evidence),
                    **_field("detail", finding["detail"]),
                    "match_count": 0,
                    "matches": [],
                }
            )
            continue

        if len(evidence) < min_evidence_chars:
            raise GateError(
                "evidence_too_short",
                index=index,
                evidence_chars=len(evidence),
                min_evidence_chars=min_evidence_chars,
                **_field("evidence", evidence),
            )

        if len(evidence) > max_evidence_chars:
            raise GateError(
                "evidence_too_long",
                index=index,
                evidence_chars=len(evidence),
                max_evidence_chars=max_evidence_chars,
                **_field("evidence", evidence),
            )

        matches = [
            {
                "source": source.name,
                "kind": source.kind,
                "offset": offset,
                "source_fraction": round(len(evidence) / max(len(source.text), 1), 4),
            }
            for source in scoped
            for offset in _offsets(source.text, evidence)
        ]
        if matches and max_source_fraction is not None and all(
            match["source_fraction"] > max_source_fraction for match in matches
        ):
            # Observed on a real model: asked to quote the defective line, it
            # pasted the entire file.  Every character matched, so a pure
            # substring test admitted it, while the quote localised nothing.
            raise GateError(
                "evidence_covers_whole_source",
                index=index,
                max_source_fraction=max_source_fraction,
                source_fractions=sorted(
                    {match["source_fraction"] for match in matches}
                ),
                **_field("evidence", evidence),
            )
        if not matches:
            # Distinguish the two failures.  "You quoted something you were
            # shown, but not of the kind this question is about" is a
            # different defect from "you quoted something you were never
            # shown", and collapsing them hides which one is happening.
            elsewhere = [
                source.kind
                for source in out_of_scope
                if evidence in source.text
            ]
            if elsewhere:
                raise GateError(
                    "evidence_kind_not_allowed",
                    index=index,
                    matched_kinds=sorted(set(elsewhere)),
                    allowed_kinds=sorted(allowed_kinds or ()),
                    **_field("evidence", evidence),
                )
            # A third failure, separated for the same reason as the second.
            # The quote IS in the source apart from its line endings, so the
            # model did read the code and the mismatch came from the platform
            # the file was checked out on.  Still refused -- comparison stays
            # exact, and nothing here admits a quote it could not find -- but
            # a caller seeing this knows to normalise its sources rather than
            # to distrust its model.
            if any(
                _one_newline(evidence) in _one_newline(source.text)
                for source in scoped
            ):
                raise GateError(
                    "evidence_line_endings_differ",
                    index=index,
                    **_field("evidence", evidence),
                )
            raise GateError(
                "evidence_not_in_context",
                index=index,
                **_field("evidence", evidence),
            )

        rows.append(
            {
                "index": index,
                "quoted": True,
                "verified_absence": False,
                **_field("requirement", requirement),
                **_field("evidence", evidence),
                # Reported because the receipt was unusable without it.  Every
                # offset in ``matches`` is a character index -- it comes from
                # ``str.find`` -- while ``evidence_bytes`` is a byte count, so
                # a reader trying to turn a match into a span was mixing two
                # units and getting away with it only on ASCII input.  Found
                # by writing a consumer that needed exactly that arithmetic,
                # which is the sort of defect no amount of reading the code
                # reveals.
                "evidence_chars": len(evidence),
                **_field("detail", finding["detail"]),
                "match_count": len(matches),
                "matches": matches,
            }
        )

    return {
        "schema": RECEIPT_SCHEMA,
        "accepted": True,
        "finding_count": len(rows),
        # Kept apart on purpose: a reader must never be able to mistake the
        # number of located defects for the number of allegations.
        "quoted_finding_count": sum(1 for row in rows if row["quoted"]),
        "verified_absence_count": sum(1 for row in rows if row["verified_absence"]),
        # Counted because it cannot be refused.  Every bound in this library
        # constrains one quote in isolation, and nothing stops a model from
        # attaching the *same* admissible quote to every requirement it was
        # given.  That answer passes every check here: the span is real, it is
        # located, it is inside both ceilings and above both floors.  Judging
        # whether a quote is *about* its requirement needs a reader or a model,
        # and this library has neither, so refusing it would be a judgment the
        # gate is not entitled to make.
        #
        # What the gate can do is make the shape visible.  Four findings
        # carrying one distinct span is not proof of a lazy answer, and the
        # receipt does not say it is -- but a caller who sees
        # ``distinct_evidence_count == 1`` next to ``quoted_finding_count == 4``
        # knows to look, and a caller who never gets the number cannot.
        "distinct_evidence_count": len(
            {row["evidence_sha256"] for row in rows if row["quoted"]}
        ),
        "reused_evidence_count": sum(1 for row in rows if row["quoted"])
        - len({row["evidence_sha256"] for row in rows if row["quoted"]}),
        "unverified_absence_count": sum(
            1 for row in rows if not row["quoted"] and not row["verified_absence"]
        ),
        "absence_label": absence_label,
        "absence_prefix": absence_prefix,
        "require_whole_clause": require_whole_clause,
        "requirement_clause_count": len(clauses),
        "min_evidence_chars": min_evidence_chars,
        "max_evidence_chars": max_evidence_chars,
        "max_source_fraction": max_source_fraction,
        "min_absence_pattern_chars": min_absence_pattern_chars,
        "max_absence_source_fraction": max_absence_source_fraction,
        "max_absence_pattern_chars": max_absence_pattern_chars,
        "allowed_kinds": sorted(allowed_kinds) if allowed_kinds is not None else None,
        # Canonical digest over the clause list, so a reader can tell whether
        # two receipts were produced against the same requirements.
        **_field("requirements", "\n".join(clauses)),
        "sources": [
            {"name": source.name, "kind": source.kind, **_digest(source.text)}
            for source in sources
        ],
        "findings": rows,
    }
