"""Controls for the dataset harness, which is where a wrong number would come from.

The measurement in ``examples/against_a_dataset.py`` scores a model against a
third party's ground truth.  That score is only worth reading if the harness
derived the ground truth correctly, and a harness bug here would not look like
a bug -- it would look like a result.  This repository has already published one
number that turned out to be an artefact of its own harness, so the extraction
gets controls of its own.

The dataset is not vendored.  Point ``EVIDENCE_GATE_DATASET`` at a QuixBugs
checkout to run these; without it they skip, and the rest of the suite is
unaffected.  No network, no model: these tests only check arithmetic on files.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "against_a_dataset", ROOT / "examples" / "against_a_dataset.py"
)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

DATASET = os.environ.get("EVIDENCE_GATE_DATASET")


class GroundTruthExtraction(unittest.TestCase):
    """Pure-function controls that need no dataset at all."""

    def test_a_trailing_docstring_is_not_mistaken_for_a_code_change(self):
        """The defect this harness was born with.

        In QuixBugs the broken file carries the specification as a trailing
        docstring and the fixed file does not.  A plain line diff therefore
        reports the whole docstring as changed -- eighteen lines for
        levenshtein, where the real defect is three characters.  Comparing
        code only is what makes the ground truth a single line.
        """
        broken = 'def f(a):\n    return 1 + a\n\n"""\nSpec\nover\nlines\n"""\n'
        fixed = 'def f(a):\n    return a\n'

        self.assertEqual(
            ["def f(a):", "    return 1 + a"], harness._code_lines(broken)
        )
        self.assertEqual(["def f(a):", "    return a"], harness._code_lines(fixed))

    def test_comments_and_blank_lines_are_not_ground_truth(self):
        source = "def f():\n\n    # a comment\n    return 1\n"
        self.assertEqual(["def f():", "    return 1"], harness._code_lines(source))

    def test_unparseable_source_is_reported_rather_than_guessed(self):
        self.assertIsNone(harness._code_lines("def ("))

    def test_the_requirement_comes_from_the_dataset_not_from_me(self):
        source = 'def f():\n    return 1\n\n"""\nGreatest Common Divisor\n\nOutput:\n    An int\n"""\n'
        requirement = harness._requirement(source)
        self.assertIsNotNone(requirement)
        self.assertIn("Greatest Common Divisor", requirement)
        self.assertNotIn("\n", requirement, "the requirement must be one clause")

    def test_a_trivially_short_docstring_is_not_treated_as_a_specification(self):
        """Otherwise a one-word docstring becomes the requirement and the
        model is being asked to review against nothing."""
        self.assertIsNone(harness._requirement('def f():\n    return 1\n\n"""x"""\n'))

    def test_renaming_removes_the_names_and_keeps_the_algorithm(self):
        source = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        renamed = harness._rename(source)
        self.assertIsNotNone(renamed)
        for name in ("gcd", " a", " b"):
            self.assertNotIn(name, renamed, f"{name!r} survived renaming")
        self.assertIn("%", renamed, "the operation itself must be untouched")
        self.assertIn("return", renamed)


@unittest.skipUnless(DATASET, "set EVIDENCE_GATE_DATASET to a QuixBugs checkout")
class AgainstTheRealDataset(unittest.TestCase):
    """The control that matters: every offset must address its own line.

    ``_localised`` decides the headline number by intersecting receipt match
    offsets with the defect's character span.  If a single span is off by one
    the score is quietly wrong, so every case is checked rather than a sample.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = pathlib.Path(DATASET)

    def test_every_defect_span_addresses_the_line_it_claims(self):
        for renamed in (False, True):
            cases = harness.cases(self.dataset, renamed)
            self.assertGreater(len(cases), 0, f"no cases with renamed={renamed}")
            for case in cases:
                with self.subTest(case=case["name"], renamed=renamed):
                    self.assertEqual(
                        case["defect_line"],
                        case["source"][case["defect_start"] : case["defect_end"]],
                        "the recorded span does not contain the defect line",
                    )

    def test_the_defect_line_actually_differs_from_the_fix(self):
        """A case where the two versions agree would score as free success."""
        for case in harness.cases(self.dataset, False):
            with self.subTest(case=case["name"]):
                self.assertNotEqual(case["defect_line"], case["fixed_line"])

    def test_the_model_is_never_shown_the_specification_twice(self):
        """The requirement is passed separately, so it must be out of the source.

        If the docstring stayed in the program, the model could quote the
        specification and the gate would admit it: a real span, exactly
        located, and not code at all.
        """
        for case in harness.cases(self.dataset, False):
            with self.subTest(case=case["name"]):
                self.assertNotIn(case["requirement"][:40], case["source"])

    def test_localisation_is_true_for_the_defect_line_and_false_elsewhere(self):
        """A scorer that cannot fail is not a scorer.

        Two synthetic receipts per case: one whose match sits on the defect
        span, one whose match sits outside it.  The first must score, the
        second must not.
        """
        for case in harness.cases(self.dataset, False)[:10]:
            with self.subTest(case=case["name"]):
                on_target = {
                    "findings": [
                        {
                            "quoted": True,
                            "evidence_chars": len(case["defect_line"]),
                            "matches": [{"offset": case["defect_start"]}],
                        }
                    ]
                }
                self.assertTrue(harness._localised(on_target, case))

                elsewhere = {
                    "findings": [
                        {
                            "quoted": True,
                            "evidence_chars": 1,
                            "matches": [{"offset": case["defect_end"] + 1}],
                        }
                    ]
                }
                self.assertFalse(harness._localised(elsewhere, case))

    def test_a_miss_distance_of_zero_means_the_same_line(self):
        """The diagnostic must agree with the headline it explains.

        If a quote on the defect span could report a non-zero line distance,
        the two numbers would disagree and a reader would have no way to tell
        which one to trust.
        """
        for case in harness.cases(self.dataset, False)[:10]:
            with self.subTest(case=case["name"]):
                on_target = {
                    "findings": [
                        {
                            "quoted": True,
                            "evidence_chars": len(case["defect_line"]),
                            "matches": [{"offset": case["defect_start"]}],
                        }
                    ]
                }
                self.assertTrue(harness._localised(on_target, case))
                self.assertEqual(0, harness._miss_distance(on_target, case))

    def test_a_miss_distance_counts_lines_not_characters(self):
        """Measured in the unit the model was asked to answer in.

        A quote placed exactly one newline earlier than the defect must
        report 1, whatever the length of the lines involved.
        """
        case = next(
            c
            for c in harness.cases(self.dataset, False)
            if c["source"][: c["defect_start"]].count("\n") >= 2
        )
        previous_newline = case["source"].rindex("\n", 0, case["defect_start"] - 1)
        receipt = {
            "findings": [
                {
                    "quoted": True,
                    "evidence_chars": 3,
                    "matches": [{"offset": previous_newline + 1}],
                }
            ]
        }
        self.assertEqual(1, harness._miss_distance(receipt, case))
        self.assertFalse(harness._localised(receipt, case))

    def test_a_miss_distance_is_none_when_nothing_was_quoted(self):
        """Absence has no position, so it must not report a distance of zero.

        Zero would read as a perfect hit in the aggregate, turning a finding
        that pointed nowhere into the best possible result.
        """
        case = harness.cases(self.dataset, False)[0]
        receipt = {"findings": [{"quoted": False, "matches": []}]}
        self.assertIsNone(harness._miss_distance(receipt, case))

    def test_an_absence_finding_never_counts_as_localisation(self):
        """Absence carries no position, so it cannot overlap anything.

        Letting it score would mean a model that names a missing fragment gets
        credit for pointing at a line it never pointed at.
        """
        case = harness.cases(self.dataset, False)[0]
        receipt = {
            "findings": [
                {"quoted": False, "evidence_chars": 0, "matches": []}
            ]
        }
        self.assertFalse(harness._localised(receipt, case))


if __name__ == "__main__":
    unittest.main()
