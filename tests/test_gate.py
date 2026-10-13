"""Controls for the gate.

A gate is worth exactly what it refuses.  So the suite is built around a
table of refusals, and one test asserts the table covers every declared
reason: if a release adds a way to refuse without adding a control for it,
the suite fails.
"""
from __future__ import annotations

import json
import unittest
import unicodedata

from evidence_gate import REASONS, EvidenceSource, Finding, GateError, check

REQUIREMENTS = "A balance must never be negative. The UI must show the balance."

SOURCE = EvidenceSource(
    name="account.py",
    kind="source",
    text=(
        "def withdraw(account, amount):\n"
        "    account.balance -= amount\n"
        "    return account.balance\n"
    ),
)

# Text the model was also shown, but of a different kind.  In the host system
# this was an observed HTTP response; here it stands for any non-source
# material a model might quote when it cannot quote the source.
OBSERVATION = EvidenceSource(
    name="response-1",
    kind="http_observation",
    text='{"status": 200, "body": {"balance": -40}}',
)

SOURCES = (SOURCE, OBSERVATION)


def finding(**overrides: object) -> dict:
    base = {
        "requirement": "A balance must never be negative",
        "evidence": "account.balance -= amount",
        "detail": "withdraw subtracts without a floor, so the balance can go below zero.",
    }
    base.update(overrides)
    return base


# (reason, findings, kwargs) -- every entry is a refusal the gate must make.
REFUSALS: tuple[tuple[str, list, dict], ...] = (
    (
        "finding_not_object",
        ["a bare string"],
        {},
    ),
    (
        "finding_key_set_mismatch",
        [{"requirement": "A balance must never be negative", "evidence": "x" * 12}],
        {},
    ),
    (
        "finding_text_invalid",
        [finding(detail="   ")],
        {},
    ),
    (
        # The model invented a plausible-sounding rule instead of quoting one.
        "requirement_not_exact_quote",
        [finding(requirement="Balances should stay positive")],
        {},
    ),
    (
        "evidence_too_short",
        [finding(evidence="amount")],
        {},
    ),
    (
        "evidence_too_long",
        [finding(evidence="x" * 40)],
        {"max_evidence_chars": 20},
    ),
    (
        # Measured against qwen3:8b on 2026-10-13: asked to quote the
        # defective line, it pasted all 87 bytes of the file, three times.
        # A pure substring test admits that; it localises nothing.
        "evidence_covers_whole_source",
        [finding(evidence=SOURCE.text)],
        {},
    ),
    (
        # The question was about source code; the model answered by quoting
        # an observation.  This is the single most common escape route.
        "evidence_kind_not_allowed",
        [finding(evidence='{"balance": -40}')],
        {"allowed_kinds": frozenset({"source"})},
    ),
    (
        # Nothing the model was shown contains this text.
        "evidence_not_in_context",
        [finding(evidence="if account.balance < 0: raise ValueError")],
        {},
    ),
)


class AdmissionTests(unittest.TestCase):
    def test_exact_quote_is_admitted_with_a_locatable_receipt(self):
        receipt = check([finding()], requirements=REQUIREMENTS, sources=SOURCES)
        self.assertIs(True, receipt["accepted"])
        self.assertEqual(1, receipt["finding_count"])
        row = receipt["findings"][0]
        self.assertEqual(1, row["match_count"])
        match = row["matches"][0]
        self.assertEqual("account.py", match["source"])
        self.assertEqual("source", match["kind"])
        # The offset is not decoration: the caller can re-extract the quote
        # from its own text and confirm the gate without trusting it.
        start = match["offset"]
        self.assertEqual(
            finding()["evidence"], SOURCE.text[start : start + len(finding()["evidence"])]
        )

    def test_dataclass_and_dict_findings_are_equivalent(self):
        typed = Finding(**finding())
        self.assertEqual(
            check([typed], requirements=REQUIREMENTS, sources=SOURCES),
            check([finding()], requirements=REQUIREMENTS, sources=SOURCES),
        )

    def test_repeated_quote_reports_every_location(self):
        source = EvidenceSource(name="d.py", kind="source", text="total = 0\ntotal = 0\n")
        receipt = check(
            [finding(evidence="total = 0")],
            requirements=REQUIREMENTS,
            sources=[source],
        )
        self.assertEqual(2, receipt["findings"][0]["match_count"])

    def test_empty_finding_list_is_admitted_and_not_a_pass_verdict(self):
        receipt = check([], requirements=REQUIREMENTS, sources=SOURCES)
        # Admitting nothing is not the same as approving the code.  The
        # receipt says a count, never a quality verdict.
        self.assertEqual(0, receipt["finding_count"])
        self.assertNotIn("critical", receipt)
        self.assertNotIn("approved", receipt)


class RefusalTests(unittest.TestCase):
    def test_every_tabled_refusal_is_made(self):
        for reason, findings, kwargs in REFUSALS:
            with self.subTest(reason=reason):
                with self.assertRaises(GateError) as caught:
                    check(
                        findings,
                        requirements=REQUIREMENTS,
                        sources=SOURCES,
                        **kwargs,
                    )
                self.assertEqual(reason, caught.exception.reason)
                self.assertEqual(reason, caught.exception.diagnostic["reason"])

    def test_the_table_covers_every_declared_reason(self):
        self.assertEqual(REASONS, {reason for reason, _, _ in REFUSALS})

    def test_no_refusal_echoes_the_rejected_text(self):
        # A rejected quote is often fabricated text.  A diagnostic that
        # repeats it would launder an unverified claim into the log.
        for reason, findings, kwargs in REFUSALS:
            with self.subTest(reason=reason):
                with self.assertRaises(GateError) as caught:
                    check(
                        findings,
                        requirements=REQUIREMENTS,
                        sources=SOURCES,
                        **kwargs,
                    )
                serialised = json.dumps(caught.exception.diagnostic, sort_keys=True)
                for item in findings:
                    if not isinstance(item, dict):
                        continue
                    for key in ("evidence", "requirement", "detail"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            self.assertNotIn(value, serialised)

    def test_refusal_is_closed_at_the_first_inadmissible_finding(self):
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence="never shown anywhere at all"), finding()],
                requirements=REQUIREMENTS,
                sources=SOURCES,
            )
        self.assertEqual(0, caught.exception.diagnostic["index"])


class NearMissTests(unittest.TestCase):
    """Comparison is exact.  These would all pass under "close enough"."""

    def _refused(self, evidence: str) -> str:
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence=evidence)],
                requirements=REQUIREMENTS,
                sources=SOURCES,
            )
        return caught.exception.reason

    def test_collapsed_whitespace_is_refused(self):
        self.assertEqual(
            "evidence_not_in_context", self._refused("account.balance-=amount")
        )

    def test_changed_case_is_refused(self):
        self.assertEqual(
            "evidence_not_in_context", self._refused("Account.Balance -= amount")
        )

    def test_reflowed_indentation_is_refused(self):
        self.assertEqual(
            "evidence_not_in_context",
            self._refused("def withdraw(account, amount):\naccount.balance -= amount"),
        )

    def test_different_unicode_normal_form_is_refused(self):
        source = EvidenceSource(
            name="u.py", kind="source", text=unicodedata.normalize("NFC", "tutar = ödenen")
        )
        decomposed = unicodedata.normalize("NFD", "tutar = ödenen")
        self.assertNotEqual(source.text, decomposed)
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence=decomposed)],
                requirements=REQUIREMENTS,
                sources=[source],
            )
        self.assertEqual("evidence_not_in_context", caught.exception.reason)


class ScopeTests(unittest.TestCase):
    def test_widening_the_scope_admits_what_narrowing_refused(self):
        quoted = finding(evidence='{"balance": -40}')
        with self.assertRaises(GateError):
            check(
                [quoted],
                requirements=REQUIREMENTS,
                sources=SOURCES,
                allowed_kinds=frozenset({"source"}),
            )
        receipt = check([quoted], requirements=REQUIREMENTS, sources=SOURCES)
        self.assertEqual("http_observation", receipt["findings"][0]["matches"][0]["kind"])

    def test_whole_file_quote_is_a_policy_not_a_law(self):
        # Disabling the bound admits the degenerate quote again.  Recording
        # the bound in the receipt is what lets a reader tell which policy
        # produced a given verdict.
        whole = finding(evidence=SOURCE.text)
        receipt = check(
            [whole],
            requirements=REQUIREMENTS,
            sources=SOURCES,
            max_source_fraction=None,
        )
        self.assertIsNone(receipt["max_source_fraction"])
        self.assertEqual(1.0, receipt["findings"][0]["matches"][0]["source_fraction"])

    def test_declared_scope_is_recorded_in_the_receipt(self):
        receipt = check(
            [finding()],
            requirements=REQUIREMENTS,
            sources=SOURCES,
            allowed_kinds=frozenset({"source"}),
        )
        self.assertEqual(["source"], receipt["allowed_kinds"])
        # Out-of-scope material stays listed, so a reader can see what the
        # model had access to and not merely what it was allowed to quote.
        self.assertEqual(
            ["account.py", "response-1"], [s["name"] for s in receipt["sources"]]
        )


class AbsenceTests(unittest.TestCase):
    """A requirement can be violated by code that is missing."""

    MISSING = "Every withdrawal must be recorded"
    LABEL = "absence:no-such-code"

    def _findings(self) -> list[dict]:
        return [
            finding(),
            finding(
                requirement=self.MISSING,
                evidence=self.LABEL,
                detail="No call records the withdrawal anywhere in the file.",
            ),
        ]

    def test_absence_is_inexpressible_unless_the_caller_enables_it(self):
        with self.assertRaises(GateError) as caught:
            check(
                self._findings(),
                requirements=REQUIREMENTS + " " + self.MISSING + ".",
                sources=SOURCES,
            )
        self.assertEqual("evidence_not_in_context", caught.exception.reason)

    def test_enabled_absence_is_admitted_but_counted_apart(self):
        receipt = check(
            self._findings(),
            requirements=REQUIREMENTS + " " + self.MISSING + ".",
            sources=SOURCES,
            absence_label=self.LABEL,
        )
        self.assertEqual(2, receipt["finding_count"])
        self.assertEqual(1, receipt["quoted_finding_count"])
        self.assertEqual(1, receipt["unquoted_finding_count"])
        self.assertEqual(self.LABEL, receipt["absence_label"])
        unquoted = receipt["findings"][1]
        self.assertIs(False, unquoted["quoted"])
        self.assertEqual(0, unquoted["match_count"])
        self.assertEqual([], unquoted["matches"])

    def test_absence_still_requires_a_real_requirement(self):
        with self.assertRaises(GateError) as caught:
            check(
                [finding(requirement="Invented rule", evidence=self.LABEL)],
                requirements=REQUIREMENTS,
                sources=SOURCES,
                absence_label=self.LABEL,
            )
        self.assertEqual("requirement_not_exact_quote", caught.exception.reason)

    def test_empty_evidence_is_not_a_declaration_of_absence(self):
        # Silence must not be read as a claim.  A model that wants to say
        # "nothing to quote" has to say it.
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence="")],
                requirements=REQUIREMENTS,
                sources=SOURCES,
                absence_label=self.LABEL,
            )
        self.assertEqual("finding_text_invalid", caught.exception.reason)

    def test_absence_label_must_match_exactly(self):
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence=self.LABEL.upper())],
                requirements=REQUIREMENTS,
                sources=SOURCES,
                absence_label=self.LABEL,
            )
        self.assertEqual("evidence_not_in_context", caught.exception.reason)


class ReceiptTests(unittest.TestCase):
    def test_receipt_carries_no_quoted_text(self):
        receipt = check([finding()], requirements=REQUIREMENTS, sources=SOURCES)
        serialised = json.dumps(receipt, sort_keys=True)
        for value in (SOURCE.text, REQUIREMENTS, finding()["evidence"], finding()["detail"]):
            self.assertNotIn(value, serialised)

    def test_receipt_is_json_serialisable_and_stable(self):
        first = json.dumps(
            check([finding()], requirements=REQUIREMENTS, sources=SOURCES), sort_keys=True
        )
        second = json.dumps(
            check([finding()], requirements=REQUIREMENTS, sources=SOURCES), sort_keys=True
        )
        self.assertEqual(first, second)

    def test_source_digests_detect_a_changed_source(self):
        before = check([finding()], requirements=REQUIREMENTS, sources=SOURCES)
        edited = EvidenceSource(
            name=SOURCE.name, kind=SOURCE.kind, text=SOURCE.text + "# touched\n"
        )
        after = check([finding()], requirements=REQUIREMENTS, sources=[edited, OBSERVATION])
        self.assertNotEqual(
            before["sources"][0]["sha256"], after["sources"][0]["sha256"]
        )


class ContractTests(unittest.TestCase):
    def test_undeclared_reason_cannot_be_raised(self):
        with self.assertRaises(AssertionError):
            raise GateError("made_up_reason")

    def test_invalid_minimum_is_rejected(self):
        with self.assertRaises(ValueError):
            check([], requirements=REQUIREMENTS, sources=SOURCES, min_evidence_chars=0)

    def test_non_string_requirements_are_rejected(self):
        with self.assertRaises(TypeError):
            check([], requirements=None, sources=SOURCES)


if __name__ == "__main__":
    unittest.main()
