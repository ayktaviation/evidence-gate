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
        "evidence_covers_whole_source",
        "evidence_kind_not_allowed",
        "evidence_not_in_context",
    }
)

#: A quote is a pointer into text, not a copy of it.  Both bounds exist
#: because a substring test has a degenerate escape at each end: a quote of
#: three characters matches by accident, and a quote of the entire file
#: matches by construction.  A model that pastes the whole source back has
#: shown nothing about where the defect is.
DEFAULT_MAX_EVIDENCE_CHARS = 2000
DEFAULT_MAX_SOURCE_FRACTION = 0.5

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
    if isinstance(requirements, str) is False:
        raise TypeError("requirements must be str")

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
            if requirement not in requirements:
                raise GateError(
                    "requirement_not_exact_quote",
                    index=index,
                    **_field("requirement", requirement),
                )
            rows.append(
                {
                    "index": index,
                    "quoted": False,
                    **_field("requirement", requirement),
                    **_field("evidence", evidence),
                    **_field("detail", finding["detail"]),
                    "match_count": 0,
                    "matches": [],
                }
            )
            continue

        # A model may only allege against a requirement the caller actually
        # wrote.  Paraphrasing a requirement is how a review drifts into
        # reviewing a rule nobody asked for.
        if requirement not in requirements:
            raise GateError(
                "requirement_not_exact_quote",
                index=index,
                **_field("requirement", requirement),
            )

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
            raise GateError(
                "evidence_not_in_context",
                index=index,
                **_field("evidence", evidence),
            )

        rows.append(
            {
                "index": index,
                "quoted": True,
                **_field("requirement", requirement),
                **_field("evidence", evidence),
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
        "unquoted_finding_count": sum(1 for row in rows if not row["quoted"]),
        "absence_label": absence_label,
        "min_evidence_chars": min_evidence_chars,
        "max_evidence_chars": max_evidence_chars,
        "max_source_fraction": max_source_fraction,
        "allowed_kinds": sorted(allowed_kinds) if allowed_kinds is not None else None,
        **_field("requirements", requirements),
        "sources": [
            {"name": source.name, "kind": source.kind, **_digest(source.text)}
            for source in sources
        ],
        "findings": rows,
    }
