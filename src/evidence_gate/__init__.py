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
    REASONS,
    RECEIPT_SCHEMA,
    EvidenceSource,
    Finding,
    GateError,
    check,
)

__all__ = [
    "EvidenceSource",
    "Finding",
    "GateError",
    "REASONS",
    "RECEIPT_SCHEMA",
    "check",
    "__version__",
]

__version__ = "0.1.0"
