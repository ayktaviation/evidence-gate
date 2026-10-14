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
