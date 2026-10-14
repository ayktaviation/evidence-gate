"""Controls for the gate.

A gate is worth exactly what it refuses.  So the suite is built around a
table of refusals, and one test asserts the table covers every declared
reason: if a release adds a way to refuse without adding a control for it,
the suite fails.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest
import unicodedata

# So that `python -m unittest discover -s tests` works in a fresh clone with
# nothing installed.  The gate has no dependencies; requiring an install step
# to run its own controls would be a poor advertisement for that.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evidence_gate import REASONS, EvidenceSource, Finding, GateError, check  # noqa: E402

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
    (
        "absence_pattern_too_short",
        [finding(evidence="absent:x")],
        {"absence_prefix": "absent:"},
    ),
    (
        # The model claimed something is missing that is plainly there.  This
        # is the one absence claim a gate can settle outright.
        "absence_pattern_present",
        [finding(evidence="absent:account.balance -= amount")],
        {"absence_prefix": "absent:", "allowed_kinds": frozenset({"source"})},
    ),
    (
        # The mirror of evidence_covers_whole_source.  A whole invented
        # sentence is missing from a short file whether or not the requirement
        # is met, so its absence carries no information.
        "absence_pattern_too_broad",
        [
            finding(
                evidence=(
                    "absent:if amount > account.balance:\n"
                    "        raise ValueError('insufficient funds')\n"
                )
            )
        ],
        {"absence_prefix": "absent:", "allowed_kinds": frozenset({"source"})},
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

    def test_the_published_reason_set_is_pinned_by_name(self):
        """Lock the public interface, because the versioning policy is a claim.

        CHANGELOG.md states that adding or removing a refusal reason is a
        breaking change: callers are expected to branch on these names, and
        code that handled every reason yesterday may fall through today.
        Nothing enforced that.  The test above is self-referential -- add a
        reason and a control for it together and it still passes -- so the
        policy was the one rule in this repository with no control of its own.

        Pinned literally, so changing the set is impossible to do by accident.
        Editing this list is the moment you are meant to read the policy and
        decide on a version bump; a diff to these twelve strings is the signal
        a reviewer needs and could not previously get.
        """
        self.assertEqual(
            {
                "absence_pattern_present",
                "absence_pattern_too_broad",
                "absence_pattern_too_short",
                "evidence_covers_whole_source",
                "evidence_kind_not_allowed",
                "evidence_not_in_context",
                "evidence_too_long",
                "evidence_too_short",
                "finding_key_set_mismatch",
                "finding_not_object",
                "finding_text_invalid",
                "requirement_not_exact_quote",
            },
            set(REASONS),
        )

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

    def test_a_match_offset_and_an_evidence_length_share_one_unit(self):
        """Positions and sizes must be measurable against each other.

        The receipt reports match offsets from ``str.find``, which are
        character indices, and ``evidence_bytes`` from a UTF-8 encode, which
        is a byte count.  Anyone turning a match into a span needs a length in
        the same unit as the offset, and for a while the receipt published no
        such field -- so the first consumer that needed the arithmetic was
        mixing units and surviving only because its input was ASCII.

        The multi-byte text has to sit inside the quote, not merely in the
        file.  A first attempt put it in a preceding comment and the control
        passed for the wrong reason: the offset shifts, but an ASCII quote's
        byte and character lengths still agree, so the units never collide.
        With the non-ASCII characters in the evidence itself,
        ``offset + evidence_bytes`` overshoots the end of the span and
        ``offset + evidence_chars`` lands exactly on it.
        """
        source = EvidenceSource(
            name="unicode.py",
            kind="source",
            # Padded past twice the quote's length, because the whole-source
            # ceiling refused a shorter version of this very control.
            text=(
                "# byte ile karakter ayni sey degildir\n"
                "value = compute(ölçüm, total)\n"
                "return value\n"
            ),
        )
        evidence = "value = compute(ölçüm, total)"
        requirement = "A value must be computed."

        receipt = check(
            [{"requirement": requirement, "evidence": evidence, "detail": "."}],
            requirements=requirement,
            sources=[source],
        )
        row = receipt["findings"][0]
        offset = row["matches"][0]["offset"]

        self.assertNotEqual(
            len(source.text.encode("utf-8")),
            len(source.text),
            "the control source is no longer multi-byte, so it proves nothing",
        )
        self.assertEqual(len(evidence), row["evidence_chars"])
        self.assertEqual(
            evidence,
            source.text[offset : offset + row["evidence_chars"]],
            "offset plus evidence_chars must recover exactly the quote",
        )
        self.assertNotEqual(
            evidence,
            source.text[offset : offset + row["evidence_bytes"]],
            "if bytes and chars agree here the control has stopped working",
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
        self.assertEqual(1, receipt["unverified_absence_count"])
        self.assertEqual(0, receipt["verified_absence_count"])
        self.assertEqual(self.LABEL, receipt["absence_label"])
        unquoted = receipt["findings"][1]
        self.assertIs(False, unquoted["quoted"])
        self.assertIs(False, unquoted["verified_absence"])
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


class VerifiedAbsenceTests(unittest.TestCase):
    """The strong form: the model names what is missing and the gate looks."""

    MISSING = "Every withdrawal must be recorded"
    PREFIX = "absent:"

    def _check(self, pattern: str, **kwargs):
        return check(
            [
                finding(
                    requirement=self.MISSING,
                    evidence=self.PREFIX + pattern,
                    detail="Nothing in the file records the withdrawal.",
                )
            ],
            requirements=[self.MISSING, "A balance must never be negative"],
            sources=SOURCES,
            absence_prefix=self.PREFIX,
            **kwargs,
        )

    def test_absence_of_named_text_is_verified_not_trusted(self):
        receipt = self._check("record(")
        row = receipt["findings"][0]
        self.assertIs(True, row["verified_absence"])
        self.assertIs(False, row["quoted"])
        self.assertEqual(1, receipt["verified_absence_count"])
        self.assertEqual(0, receipt["unverified_absence_count"])
        # Which text was searched is recorded, so the search is repeatable.
        self.assertEqual(7, row["absent_pattern_chars"])
        self.assertEqual(["account.py", "response-1"], row["searched_sources"])

    def test_a_false_absence_claim_is_refused_with_its_locations(self):
        with self.assertRaises(GateError) as caught:
            self._check("account.balance")
        self.assertEqual("absence_pattern_present", caught.exception.reason)
        self.assertEqual(2, caught.exception.diagnostic["occurrences"])
        self.assertEqual(["account.py"], caught.exception.diagnostic["sources"])

    def test_search_respects_the_declared_scope(self):
        # A pattern absent from source but present in an out-of-scope
        # observation is still a verified absence *of the source*, and the
        # receipt says which sources were searched.
        receipt = self._check("balance\": -40", allowed_kinds=frozenset({"source"}))
        self.assertIs(True, receipt["findings"][0]["verified_absence"])
        self.assertEqual(["account.py"], receipt["findings"][0]["searched_sources"])

    def test_absence_floor_is_lower_than_the_quote_floor(self):
        # The two bounds exist for opposite reasons; a seven-character search
        # pattern is admissible while a seven-character quote is not.
        self.assertIs(True, self._check("record(")["findings"][0]["verified_absence"])
        with self.assertRaises(GateError) as caught:
            check(
                [finding(evidence="amount)")],
                requirements=REQUIREMENTS,
                sources=SOURCES,
            )
        self.assertEqual("evidence_too_short", caught.exception.reason)

    def test_a_pattern_as_long_as_the_file_is_refused_as_uninformative(self):
        # Found by measurement, not by design: three local models asked to
        # name the missing code produced patterns of 70, 57 and 35 characters
        # against an 88-byte file.  The first two are absent from it whatever
        # the code says, which is the same vacuous pass as quoting the whole
        # file, arrived at from the other direction.
        with self.assertRaises(GateError) as caught:
            self._check(
                "if amount > account.balance:\n        raise ValueError('no')\n",
                allowed_kinds=frozenset({"source"}),
            )
        self.assertEqual("absence_pattern_too_broad", caught.exception.reason)
        self.assertGreater(caught.exception.diagnostic["source_fraction"], 0.5)

    def test_breadth_is_recorded_so_a_weak_absence_is_visible(self):
        row = self._check("record(", allowed_kinds=frozenset({"source"}))["findings"][0]
        self.assertEqual(0.0795, row["absent_pattern_fraction"])

    def test_a_present_pattern_is_refuted_before_it_is_judged_broad(self):
        # Both bounds would fire.  The refutation is reported, because "that
        # text is in the file" tells the caller something true about the code,
        # while "your pattern was too broad" only grades the question.
        with self.assertRaises(GateError) as caught:
            self._check(
                "def withdraw(account, amount):\n    account.balance -= amount\n",
                allowed_kinds=frozenset({"source"}),
            )
        self.assertEqual("absence_pattern_present", caught.exception.reason)

    def test_the_fraction_ceiling_alone_fails_at_realistic_scale(self):
        # The regression this exists to prevent.  The fraction was fitted to
        # an 88-byte file; against a 9,000-character module it permits a
        # 4,500-character "absence pattern", so the bound that caught the
        # vacuous claim on the toy catches nothing on real code.  An invented
        # statement is absent by construction at every size, so the absolute
        # cap is the one that does the work here.
        big = EvidenceSource(
            name="big.py",
            kind="source",
            text="def handler(request):\n    return None\n" + ("# filler\n" * 1000),
        )
        invented = "if amount > account.balance:\n        raise ValueError('no')\n"
        self.assertGreater(len(big.text), 9000)

        findings = [
            finding(
                requirement=self.MISSING,
                evidence=self.PREFIX + invented,
                detail="Nothing records the withdrawal.",
            )
        ]
        arguments = {
            "requirements": [self.MISSING],
            "sources": [big],
            "absence_prefix": self.PREFIX,
        }

        # With the fraction alone -- the bounds as they were fitted to the toy.
        admitted = check(
            findings, max_absence_pattern_chars=None, **arguments
        )
        self.assertIs(True, admitted["findings"][0]["verified_absence"])
        self.assertLess(admitted["findings"][0]["absent_pattern_fraction"], 0.01)

        # With both bounds, which is the default.
        with self.assertRaises(GateError) as caught:
            check(findings, **arguments)
        self.assertEqual("absence_pattern_too_broad", caught.exception.reason)
        # And the diagnostic says which bound bit, because only one did.
        self.assertEqual(["pattern_chars"], caught.exception.diagnostic["exceeded"])

    def test_the_two_ceilings_cross_at_eighty_characters(self):
        # Worked out rather than assumed, because it decides whether the
        # fraction is doing anything.  The fraction permits half the source
        # and the cap permits 40, so the fraction binds only below 80
        # characters -- which is no real file, and exactly the degenerate case
        # this library exists for.  Below the crossover it is the only bound
        # between a 30-character pattern and a 48-character file.
        tiny = EvidenceSource(name="t.py", kind="source", text="def f():\n    return account.balance\n")
        self.assertLess(len(tiny.text), 80)
        pattern = "x" * 30
        self.assertLess(len(pattern), 40)  # under the absolute cap

        with self.assertRaises(GateError) as caught:
            check(
                [finding(requirement=self.MISSING, evidence=self.PREFIX + pattern,
                         detail="Nothing records it.")],
                requirements=[self.MISSING],
                sources=[tiny],
                absence_prefix=self.PREFIX,
            )
        self.assertEqual("absence_pattern_too_broad", caught.exception.reason)
        self.assertEqual(["source_fraction"], caught.exception.diagnostic["exceeded"])

    def test_the_ceiling_can_be_disabled_but_not_set_absurdly(self):
        long_pattern = "if amount > account.balance:\n        raise ValueError('no')\n"
        receipt = self._check(
            long_pattern,
            allowed_kinds=frozenset({"source"}),
            max_absence_source_fraction=None,
            max_absence_pattern_chars=None,
        )
        self.assertIs(True, receipt["findings"][0]["verified_absence"])
        self.assertIsNone(receipt["max_absence_source_fraction"])
        self.assertIsNone(receipt["max_absence_pattern_chars"])
        with self.assertRaises(ValueError):
            self._check("record(", max_absence_source_fraction=0)
        with self.assertRaises(ValueError):
            # A cap below the floor would refuse every possible pattern.
            self._check("record(", max_absence_pattern_chars=2)

    def test_empty_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            check([], requirements=REQUIREMENTS, sources=SOURCES, absence_prefix="")


class ClauseTests(unittest.TestCase):
    CLAUSES = ("A balance must never be negative", "The UI must show the balance")

    def test_a_requirement_fragment_is_admissible_by_default(self):
        receipt = check(
            [finding(requirement="A balance must never be")],
            requirements=self.CLAUSES,
            sources=SOURCES,
        )
        self.assertEqual(2, receipt["requirement_clause_count"])

    def test_whole_clause_mode_refuses_a_fragment(self):
        with self.assertRaises(GateError) as caught:
            check(
                [finding(requirement="A balance must never be")],
                requirements=self.CLAUSES,
                sources=SOURCES,
                require_whole_clause=True,
            )
        self.assertEqual("requirement_not_exact_quote", caught.exception.reason)

    def test_whole_clause_mode_admits_an_exact_clause(self):
        receipt = check(
            [finding()],
            requirements=self.CLAUSES,
            sources=SOURCES,
            require_whole_clause=True,
        )
        self.assertIs(True, receipt["require_whole_clause"])

    def test_clause_order_changes_the_requirements_digest(self):
        first = check([finding()], requirements=self.CLAUSES, sources=SOURCES)
        second = check(
            [finding()], requirements=tuple(reversed(self.CLAUSES)), sources=SOURCES
        )
        self.assertNotEqual(first["requirements_sha256"], second["requirements_sha256"])


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
