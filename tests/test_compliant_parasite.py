"""What a fully compliant model can still get away with.

Every other control in this suite asserts that the gate refuses something.
These assert the opposite, and they are the most important tests in the
repository for anyone deciding whether to depend on it: they pin the exact
shape of an answer that satisfies every bound and carries no information.

The name is the claim.  A parasite does not break its host; it meets every
entry requirement and contributes nothing.  A model that wants to look
productive without doing any work does not need to fabricate a quote -- that
gets caught.  It needs to copy one real line and attach it to everything.

These tests exist so that nobody -- including me -- can later describe this
library as checking that evidence supports a finding.  It checks that
evidence exists where the model said it does.  Those are different claims,
and the gap between them is measured here rather than disclaimed in prose.
If a future release closes any of this, these tests fail, and that failure is
the signal to rewrite the README's limitations section.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evidence_gate import EvidenceSource, check  # noqa: E402

# Four unrelated requirements about four unrelated concerns.  No single line
# of any program could honestly be evidence for all of them.
REQUIREMENTS = (
    "A balance must never be negative.",
    "Every withdrawal must be written to an audit log.",
    "A password must never be stored in plain text.",
    "The session must expire after inactivity.",
)

SOURCE = EvidenceSource(
    name="account.py",
    kind="source",
    text=(
        '"""Account operations for the teller terminal."""\n'
        "import datetime\n"
        "\n"
        "def withdraw(account, amount):\n"
        "    account.balance -= amount\n"
        "    return account.balance\n"
    ),
)

# A real line, copied exactly, that has nothing to do with any requirement
# above.  It is 15 characters: comfortably over min_evidence_chars (8) and
# nowhere near max_evidence_chars (2000) or half the file.
IRRELEVANT_BUT_REAL = "import datetime"


class TheGateAdmitsTrueButIrrelevantEvidence(unittest.TestCase):
    def test_one_real_line_answers_every_requirement(self):
        """The cheapest possible attack, and it works.

        One exact quote, reused four times, against four requirements it has
        no bearing on.  The gate admits all four because all four claims it
        is able to evaluate are true: the span is real, it is located, and it
        is within every bound.
        """
        findings = [
            {
                "requirement": requirement,
                "evidence": IRRELEVANT_BUT_REAL,
                "detail": "This is not an analysis and the gate cannot tell.",
            }
            for requirement in REQUIREMENTS
        ]

        receipt = check(findings, requirements=REQUIREMENTS, sources=[SOURCE])

        self.assertTrue(receipt["accepted"])
        self.assertEqual(4, receipt["quoted_finding_count"])

    def test_the_receipt_makes_the_reuse_countable(self):
        """The one defence available: say so in the numbers.

        The gate may not refuse this, but it must not hide it either.  Four
        quoted findings resting on a single distinct span is the signature of
        the attack above, and it is arithmetic rather than judgment, so the
        receipt is entitled to report it.
        """
        findings = [
            {
                "requirement": requirement,
                "evidence": IRRELEVANT_BUT_REAL,
                "detail": "Reused on purpose.",
            }
            for requirement in REQUIREMENTS
        ]

        receipt = check(findings, requirements=REQUIREMENTS, sources=[SOURCE])

        self.assertEqual(1, receipt["distinct_evidence_count"])
        self.assertEqual(3, receipt["reused_evidence_count"])
        self.assertEqual(
            receipt["quoted_finding_count"],
            receipt["distinct_evidence_count"] + receipt["reused_evidence_count"],
            "the two counts must partition the quoted findings, or a reader "
            "cannot use either",
        )

    def test_an_honest_answer_shows_no_reuse(self):
        """The counter-case, so the signal is not constant.

        A number that is high for every input measures nothing.  Distinct
        quotes must produce zero reuse, or the count above is decoration.
        """
        findings = [
            {
                "requirement": REQUIREMENTS[0],
                "evidence": "account.balance -= amount",
                "detail": "Subtracts without a floor.",
            },
            {
                "requirement": REQUIREMENTS[1],
                "evidence": "    return account.balance",
                "detail": "Returns without recording.",
            },
        ]

        receipt = check(findings, requirements=REQUIREMENTS, sources=[SOURCE])

        self.assertEqual(2, receipt["distinct_evidence_count"])
        self.assertEqual(0, receipt["reused_evidence_count"])

    def test_absence_reuse_is_not_counted_as_quoted_reuse(self):
        """Absence patterns are deliberately outside the count.

        An absence finding carries a search pattern, not a located span, and
        it is already reported apart from quotes everywhere else in the
        receipt.  Folding it into the reuse count would let a reader add
        verified absences to located quotes, which is the one confusion this
        receipt is built to prevent.
        """
        findings = [
            {
                "requirement": REQUIREMENTS[1],
                "evidence": "absent:audit_log",
                "detail": "No audit call.",
            },
            {
                "requirement": REQUIREMENTS[2],
                "evidence": "absent:audit_log",
                "detail": "Same pattern, different requirement.",
            },
        ]

        receipt = check(
            findings,
            requirements=REQUIREMENTS,
            sources=[SOURCE],
            absence_prefix="absent:",
        )

        self.assertEqual(2, receipt["verified_absence_count"])
        self.assertEqual(0, receipt["quoted_finding_count"])
        self.assertEqual(0, receipt["distinct_evidence_count"])
        self.assertEqual(0, receipt["reused_evidence_count"])


class TheGateAdmitsATriviallyAbsentPattern(unittest.TestCase):
    def test_a_short_nonsense_pattern_is_a_verified_absence(self):
        """The floor bounds length, and length is not informativeness.

        ``min_absence_pattern_chars`` is 3 because ``record(`` is seven
        characters and must stay admissible.  But ``zqx`` is three characters
        too, it is genuinely absent, and the host genuinely verified that.
        The observation is sound and the finding is worthless.

        This is the one cell of the 2x2 that cannot be closed by a bound.  A
        pattern's value depends on whether a competent implementation would
        have contained it, which is a question about plausibility, and
        plausibility is not a property of the text's length.  Anyone who reads
        ``verified_absence_count`` as a count of real defects is reading it
        wrong, and this test is where that is written down.
        """
        findings = [
            {
                "requirement": REQUIREMENTS[1],
                "evidence": "absent:zqx",
                "detail": "Technically true, entirely uninformative.",
            }
        ]

        receipt = check(
            findings,
            requirements=REQUIREMENTS,
            sources=[SOURCE],
            absence_prefix="absent:",
        )

        self.assertTrue(receipt["accepted"])
        self.assertEqual(1, receipt["verified_absence_count"])
        self.assertEqual(3, receipt["findings"][0]["absent_pattern_chars"])


if __name__ == "__main__":
    unittest.main()
