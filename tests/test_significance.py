"""Controls for the statistics, checked against values computed elsewhere.

A p-value is a claim like any other, and this repository's whole argument is
that an unchecked claim is worth less than an absent one.  These tests pin the
exact tests used in the README against textbook cases with known answers, so a
reader who distrusts a hand-rolled Fisher implementation can see it reproduce
numbers they can verify independently.

No dataset and no model needed: this is arithmetic.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "significance", ROOT / "tools" / "significance.py"
)
assert _spec is not None and _spec.loader is not None
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)


class FisherExact(unittest.TestCase):
    def test_the_tea_tasting_table(self):
        """Fisher's own example, whose answer is 0.4857 in every textbook.

        3/4 correct against 1/4 correct in a 2x2 of eight cups.  If the
        implementation cannot reproduce the case the test was invented for, it
        cannot be trusted on this project's data either.
        """
        self.assertAlmostEqual(0.4857, stats.fisher_two_tailed(3, 1, 1, 3), places=4)

    def test_a_perfect_separation_is_not_certain_at_small_n(self):
        """2x2 with no overlap still gives p = 0.0286 at n = 8, not 0.

        Worth pinning because the intuition that a clean split must be
        significant is exactly the intuition that produces overclaims.
        """
        self.assertAlmostEqual(0.0286, stats.fisher_two_tailed(4, 0, 0, 4), places=4)

    def test_identical_proportions_are_not_evidence_of_anything(self):
        self.assertAlmostEqual(1.0, stats.fisher_two_tailed(5, 5, 5, 5), places=6)

    def test_an_empty_table_does_not_divide_by_zero(self):
        self.assertEqual(1.0, stats.fisher_two_tailed(0, 0, 0, 0))

    def test_the_test_is_symmetric_under_swapping_the_rows(self):
        """Otherwise the reported p would depend on argument order, and the
        README's number would be an artefact of how the call was written."""
        self.assertAlmostEqual(
            stats.fisher_two_tailed(8, 26, 4, 32),
            stats.fisher_two_tailed(4, 32, 8, 26),
            places=10,
        )

    def test_the_measured_comparison_is_not_significant(self):
        """The result this repository must not overclaim.

        8/34 localised with real identifiers against 4/36 with meaningless
        ones looks like the contamination control landing hard.  At this
        sample size it is not significant, and the README says so.  If a
        future change to the statistics makes this pass the 0.05 threshold,
        this test fails and the claim must be re-examined rather than
        quietly upgraded.
        """
        self.assertGreater(stats.fisher_two_tailed(8, 26, 4, 32), 0.05)

    def test_the_near_miss_collapse_is_significant(self):
        """The result that does survive: 9/26 adjacent misses against 1/32."""
        self.assertLess(stats.fisher_two_tailed(9, 17, 1, 31), 0.01)


class MantelHaenszel(unittest.TestCase):
    """Controls for the stratified test, which exists to pool models safely.

    Pinned against an identity rather than a remembered constant.  For a
    single stratum the CMH statistic equals ``(n-1)/n`` times the Pearson
    chi-squared statistic, and Pearson is four terms of ``(o-e)^2/e`` that a
    reader can compute on paper.  Checking against a number I recalled from a
    textbook would be checking my memory, not the code.
    """

    @staticmethod
    def pearson(a, b, c, d):
        total = a + b + c + d
        rows, cols = (a + b, c + d), (a + c, b + d)
        expected = [rows[i] * cols[j] / total for i in (0, 1) for j in (0, 1)]
        return sum((o - e) ** 2 / e for o, e in zip((a, b, c, d), expected))

    def test_one_stratum_reduces_to_the_corrected_chi_squared(self):
        for table in [(8, 26, 4, 32), (9, 17, 1, 31), (3, 1, 1, 3), (12, 3, 10, 4)]:
            with self.subTest(table=table):
                total = sum(table)
                statistic, _ = stats.mantel_haenszel([table])
                self.assertAlmostEqual(
                    (total - 1) / total * self.pearson(*table),
                    statistic,
                    places=9,
                )

    def test_pooling_can_reverse_what_every_stratum_says(self):
        """Simpson's paradox, with counts small enough to verify by hand.

        In stratum one 1/2 = 0.500 against 4/7 = 0.571, and in stratum two
        11/13 = 0.846 against 6/7 = 0.857: row one is lower in both.  Add the
        tables and row one becomes 12/15 = 0.800 against 10/14 = 0.714 --
        higher.  This is why the models are stratified rather than summed,
        and the test asserts the arithmetic rather than describing it.
        """
        first, second = (1, 1, 4, 3), (11, 2, 6, 1)
        self.assertLess(1 / 2, 4 / 7)
        self.assertLess(11 / 13, 6 / 7)

        pooled = tuple(x + y for x, y in zip(first, second))
        self.assertEqual((12, 3, 10, 4), pooled)
        self.assertGreater(12 / 15, 10 / 14)

        # The sign of the CMH numerator is the direction it reports.  Summing
        # the tables first flips it; stratifying keeps the direction every
        # stratum actually shows.
        self.assertLess(stats.mantel_haenszel([first, second])[0], float("inf"))
        stratified = sum(
            a - (a + b) * (a + c) / (a + b + c + d) for a, b, c, d in (first, second)
        )
        naive = pooled[0] - (pooled[0] + pooled[1]) * (pooled[0] + pooled[2]) / sum(
            pooled
        )
        self.assertLess(stratified, 0, "stratified must follow the strata: row 1 lower")
        self.assertGreater(naive, 0, "the pooled table points the other way")

    def test_repeating_a_stratum_strengthens_the_same_finding(self):
        """Two models that agree must be more persuasive than one.

        Not a tautology: a stratified test that mishandled the variance sum
        could easily make agreement look weaker.
        """
        table = (8, 26, 4, 32)
        _, one = stats.mantel_haenszel([table])
        _, three = stats.mantel_haenszel([table] * 3)
        self.assertLess(three, one)

    def test_strata_with_no_difference_report_no_difference(self):
        _, p = stats.mantel_haenszel([(5, 5, 5, 5), (7, 7, 7, 7)])
        self.assertAlmostEqual(1.0, p, places=9)

    def test_a_degenerate_stratum_does_not_divide_by_zero(self):
        """One case, or none, must contribute nothing rather than crash."""
        self.assertEqual((0.0, 1.0), stats.mantel_haenszel([]))
        self.assertEqual((0.0, 1.0), stats.mantel_haenszel([(1, 0, 0, 0)]))
        _, with_junk = stats.mantel_haenszel([(8, 26, 4, 32), (0, 0, 0, 0)])
        _, without = stats.mantel_haenszel([(8, 26, 4, 32)])
        self.assertAlmostEqual(without, with_junk, places=12)

    def test_the_approximation_is_anti_conservative_here(self):
        """Pinned because it is the reason this test is not the headline.

        At n = 70 the asymptotic test reports a smaller p than the exact one
        on the same table. If that ever reverses the README's choice of which
        number to quote should be revisited, so the direction is asserted
        rather than left as a remark in a docstring.
        """
        table = (8, 26, 4, 32)
        _, approximate = stats.mantel_haenszel([table])
        self.assertLess(approximate, stats.fisher_two_tailed(*table))
        self.assertGreater(approximate, 0.05, "still not significant either way")


class ArmIdentification(unittest.TestCase):
    """The arm comes from the run file, never from argument order.

    The first version of this tool took (published, renamed) positionally and
    checked the recorded flags afterwards to catch a swap.  A third arm makes
    that check inexpressible, and a filename was never evidence of anything,
    so the arm is read from what the run wrote about itself.
    """

    def run_file(self, **flags):
        import json
        import tempfile

        body = {
            "model": flags.pop("model", "m"),
            "renamed": flags.pop("renamed", False),
            "cases": 1,
            "results": [
                {
                    # Named, because --paired identifies cases by this field
                    # and a fixture without it stopped resembling a run file.
                    "case": "only",
                    "outcome": "admitted",
                    "localised": True,
                    "source_lines": 10,
                    "miss_lines": None,
                }
            ],
            **flags,
        }
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(body, handle)
        handle.close()
        return pathlib.Path(handle.name)

    def test_the_three_arms_are_told_apart_by_their_recorded_flags(self):
        for flags, expected in (
            ({}, "published"),
            ({"normalised": False}, "published"),
            ({"normalised": True}, "normalised"),
            ({"renamed": True}, "renamed"),
        ):
            with self.subTest(flags=flags):
                self.assertEqual(expected, stats.summarise(self.run_file(**flags))["arm"])

    def test_a_run_predating_the_middle_arm_is_still_the_published_arm(self):
        """Older run files have no 'normalised' key at all.

        They must keep meaning exactly what they meant when they were
        written, rather than becoming unreadable or, worse, being relabelled.
        """
        path = self.run_file()
        import json

        self.assertNotIn("normalised", json.loads(path.read_text()))
        self.assertEqual("published", stats.summarise(path)["arm"])

    def test_restricting_to_shared_cases_changes_the_counts_it_should(self):
        """``--paired`` exists because the arms are not the same programs.

        Reformatting through ast.unparse moves where the lines fall, so the
        single-line-defect filter keeps ``powerset`` in the reformatted arms
        and drops it from the published one, and does the reverse with
        ``shortest_path_length_test``.  Each arm has 36 cases and 35 are
        common to all three, so an unpaired comparison is between slightly
        different sets of programs.
        """
        import json
        import tempfile

        body = {
            "model": "m",
            "renamed": False,
            "cases": 3,
            "results": [
                {"case": name, "outcome": "admitted", "localised": hit,
                 "source_lines": 10, "miss_lines": None}
                for name, hit in (("a", True), ("b", True), ("c", False))
            ],
        }
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(body, handle)
        handle.close()
        path = pathlib.Path(handle.name)

        everything = stats.summarise(path)
        self.assertEqual(3, everything["cases"])
        self.assertEqual(2, everything["localised"])
        self.assertEqual({"a", "b", "c"}, everything["names"])

        restricted = stats.summarise(path, {"a", "c"})
        self.assertEqual(2, restricted["cases"])
        self.assertEqual(1, restricted["localised"])
        self.assertEqual({"a", "c"}, restricted["names"])

        # A name that is not in the file must not invent a case.
        self.assertEqual(1, stats.summarise(path, {"a", "absent"})["cases"])

    def test_renamed_wins_if_a_run_somehow_claims_to_be_both(self):
        """The CLI refuses to produce such a run, so this is about reading one.

        Reporting the stronger transformation is the conservative choice: a
        file carrying both flags has had its identifiers stripped whatever
        else was done to it, and calling it the middle arm would understate
        what the model was shown.
        """
        summary = stats.summarise(self.run_file(renamed=True, normalised=True))
        self.assertEqual("renamed", summary["arm"])


class Binomial(unittest.TestCase):
    def test_a_fair_coin_is_symmetric(self):
        """P(X >= k) must equal P(X <= n - k) when p = 0.5.

        The first version of this test was an incoherent expression that
        happened to assert a number, and it failed for the right reason. Left
        as a clean identity instead: if the two tails disagree on a fair coin,
        one of the two functions is wrong and every p-value in the README is
        suspect.
        """
        for k in range(0, 11):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    stats.binomial_at_least(k, 10, 0.5),
                    stats.binomial_at_most(10 - k, 10, 0.5),
                    places=9,
                )

    def test_the_tails_partition_the_distribution(self):
        """P(>=k) + P(<=k-1) must be exactly 1, or one of them is wrong."""
        for k in range(0, 13):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    1.0,
                    stats.binomial_at_least(k, 12, 0.3)
                    + stats.binomial_at_most(k - 1, 12, 0.3),
                    places=9,
                )

    def test_every_outcome_is_at_least_as_likely_as_certainty(self):
        self.assertAlmostEqual(1.0, stats.binomial_at_least(0, 20, 0.11), places=9)

    def test_the_published_arm_beats_the_chance_floor(self):
        """8 of 34 against a per-case floor of 0.110."""
        self.assertLess(stats.binomial_at_least(8, 34, 0.110), 0.05)

    def test_the_renamed_arm_does_not_beat_the_chance_floor(self):
        """4 of 36 against a floor of 0.136: no evidence of skill left.

        Stated as a failure to reject, not as proof of guessing. The
        distinction is the difference between a finding and a slogan.
        """
        self.assertGreater(stats.binomial_at_least(4, 36, 0.136), 0.05)


if __name__ == "__main__":
    unittest.main()
