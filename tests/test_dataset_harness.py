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

    def test_the_builtin_allowlist_does_not_depend_on_how_this_file_was_loaded(self):
        """The defect that made the tests and the measurement disagree.

        The allowlist was built from ``dir(__builtins__)``.  Inside
        ``__main__`` that is the builtins *module*, so ``dir`` returns the
        builtin names; inside an imported module it is that module's
        ``__dict__``, so ``dir`` returns a dict's own methods.  Every real
        run of this harness is a script and every test of it is an import, so
        the suite was exercising a different allowlist from the one that
        produced the published numbers -- and passing, because nothing
        checked a name that fell out.

        Asserted through the public function rather than by inspecting the
        set, so the test keeps meaning something if the allowlist moves.
        """
        for name in ("any", "all", "max", "min", "sum", "len", "range"):
            with self.subTest(builtin=name):
                source = f"def g(xs):\n    return {name}(xs)\n"
                renamed = harness._rename(source)
                self.assertIsNotNone(renamed)
                self.assertIn(
                    name, renamed, f"{name!r} is a builtin and must not be renamed"
                )

    def test_renaming_cannot_erase_the_defect_it_is_controlling_for(self):
        """sieve, which is how the allowlist bug surfaced.

        Its defect is ``any`` against ``all``.  With both missing from the
        allowlist they were rewritten to the same opaque name, the broken and
        fixed programs came out byte-identical, and the case was dropped for
        having no single-line diff -- a control silently deleting the very
        thing it was meant to measure.  A rename that makes the two versions
        identical is always a harness failure, never a result.
        """
        broken = "def f(ns, ps):\n    if any(n % p > 0 for n in ns for p in ps):\n        return 1\n    return 0\n"
        fixed = broken.replace("any(", "all(")
        self.assertNotEqual(
            harness._rename(broken),
            harness._rename(fixed),
            "the rename erased the difference between broken and fixed",
        )

    def test_normalising_reformats_without_touching_the_names(self):
        """The middle arm, which is what makes the contamination claim honest.

        ``--rename`` changes two things at once: the identifiers and, because
        it reparses, the whitespace.  A model that simply reads reformatted
        code less well would produce the same localisation drop with
        contamination playing no part at all.  This arm performs the identical
        round-trip and leaves the names alone, so the two variables can be
        separated instead of disclaimed.
        """
        source = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        normalised = harness._rename(source, rename=False)
        self.assertIsNotNone(normalised)
        for name in ("gcd", "a", "b"):
            self.assertIn(name, normalised, f"{name!r} must survive normalising")
        self.assertIn("%", normalised)

        renamed = harness._rename(source)
        self.assertNotIn("gcd", renamed, "the rename arm must still rename")
        self.assertNotEqual(normalised, renamed)

    def test_the_two_reformatted_arms_differ_only_in_their_names(self):
        """Otherwise the middle arm is not controlling what it claims to.

        Both arms go through ``ast.unparse``, so their structure must match
        line for line.  If normalising produced a different shape from
        renaming, the comparison between them would be measuring the shape
        difference as well, which is the exact confound this arm exists to
        remove.
        """
        source = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        normalised = harness._rename(source, rename=False).splitlines()
        renamed = harness._rename(source).splitlines()
        self.assertEqual(len(normalised), len(renamed))
        for left, right in zip(normalised, renamed):
            self.assertEqual(
                len(left) - len(left.lstrip()),
                len(right) - len(right.lstrip()),
                "indentation must match between the two reformatted arms",
            )

    def test_asking_for_both_reformatted_arms_is_refused(self):
        """--rename already reformats, so --rename --normalise is one arm
        recorded under two names, and the control would quietly vanish."""
        import subprocess

        finished = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "against_a_dataset.py"),
                "--dataset", "/nonexistent",
                "--rename",
                "--normalise",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, finished.returncode)
        self.assertIn("would record one arm under two names", finished.stderr)

    def test_renaming_removes_the_names_and_keeps_the_algorithm(self):
        source = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        renamed = harness._rename(source)
        self.assertIsNotNone(renamed)
        for name in ("gcd", " a", " b"):
            self.assertNotIn(name, renamed, f"{name!r} survived renaming")
        self.assertIn("%", renamed, "the operation itself must be untouched")
        self.assertIn("return", renamed)

    def test_a_partial_run_is_written_to_disk_and_marked_incomplete(self):
        """Answers already paid for must survive a crash on a later case.

        A run is thirty-six sequential model calls and roughly twenty-five
        minutes.  The first version wrote the file only after the last call,
        so a reviewer dying on case thirty-five discarded every answer before
        it and left no record the attempt had happened -- in a repository
        whose rule is that failed runs are preserved.
        """
        import argparse
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "partial.json"
            options = argparse.Namespace(
                out=out,
                dataset=pathlib.Path("/nowhere"),
                rename=False,
                normalise=False,
                model="m",
                seed=1,
            )
            rows = [{"outcome": "admitted", "localised": True, "case": "a"}]

            harness._record(options, rows, complete=False)
            self.assertTrue(out.exists(), "a partial run left nothing on disk")
            partial = json.loads(out.read_text(encoding="utf-8"))
            self.assertIs(False, partial["complete"])
            self.assertEqual(1, partial["cases"])

            rows.append({"outcome": "refused", "reason": "r", "case": "b"})
            harness._record(options, rows, complete=True)
            final = json.loads(out.read_text(encoding="utf-8"))
            self.assertIs(True, final["complete"])
            self.assertEqual(2, final["cases"])
            self.assertEqual(1, final["admitted"])

    def test_recording_without_an_out_path_is_not_an_error(self):
        """The run file is optional, and the incremental write must respect that."""
        import argparse

        harness._record(
            argparse.Namespace(out=None), [{"outcome": "admitted", "localised": True}],
            complete=False,
        )


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
