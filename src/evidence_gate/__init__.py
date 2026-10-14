"""evidence-gate: refuse model findings whose quotes were never shown.

    from evidence_gate import EvidenceSource, GateError, check

    sources = [EvidenceSource(name="app.py", kind="source", text=open("app.py").read())]
    try:
        receipt = check(findings, requirements=brief, sources=sources)
    except GateError as refusal:
        print(refusal.reason, refusal.diagnostic)

No network, no model, no dependencies.  See ``evidence_gate.ollama`` for an
optional local reviewer used by the examples.
"""
from .gate import (
    DEFAULT_MAX_ABSENCE_PATTERN_CHARS,
    DEFAULT_MAX_ABSENCE_SOURCE_FRACTION,
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_SOURCE_FRACTION,
    DEFAULT_MIN_ABSENCE_PATTERN_CHARS,
    DEFAULT_MIN_EVIDENCE_CHARS,
    REASONS,
    RECEIPT_SCHEMA,
    REQUIRED_KEYS,
    EvidenceSource,
    Finding,
    GateError,
    check,
)

#: The bounds are re-exported because the versioning policy already treats
#: them as public: tightening a default is a breaking change, and both
#: tightenings in this library's history are recorded in the changelog with
#: the measurement that caused them.  A promise a caller can only reach by
#: importing a submodule is a promise they will not check, so a caller who
#: depends on a bound can now assert against the name rather than copying the
#: number.
__all__ = [
    "DEFAULT_MAX_ABSENCE_PATTERN_CHARS",
    "DEFAULT_MAX_ABSENCE_SOURCE_FRACTION",
    "DEFAULT_MAX_EVIDENCE_CHARS",
    "DEFAULT_MAX_SOURCE_FRACTION",
    "DEFAULT_MIN_ABSENCE_PATTERN_CHARS",
    "DEFAULT_MIN_EVIDENCE_CHARS",
    "EvidenceSource",
    "Finding",
    "GateError",
    "REASONS",
    "RECEIPT_SCHEMA",
    "REQUIRED_KEYS",
    "check",
    "__version__",
]

__version__ = "0.2.0"
