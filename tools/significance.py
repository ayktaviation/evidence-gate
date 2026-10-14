#!/usr/bin/env python3
"""Test the dataset runs for significance, so the README cannot overclaim.

    python tools/significance.py run-published.json run-renamed.json

The dataset harness produces ratios, and a ratio from 36 cases invites a
conclusion it cannot support.  The first run of this experiment localised 8 of
34 with real identifiers and 4 of 36 with identifiers stripped of meaning,
which looks like the contamination control working.  It is not significant:
Fisher's exact test gives p = 0.21.  Writing "localisation halved" would have
been true arithmetic and a false claim.

So the arithmetic lives here rather than in a scratch file, and the README
quotes what this prints.  Standard library only, like the rest of the
repository -- no SciPy, which means the tests are readable and the numbers are
checkable by hand if you distrust them.

Three questions get asked, and they are not the same question:

1. Does localisation differ between the two arms?  (Fisher, two-tailed.)
2. Is each arm better than picking a line at random?  (Binomial, one-tailed.)
3. Do near misses cluster differently?  (Fisher, two-tailed.)

The chance baseline is computed per case rather than assumed, because the
programs differ in length and a short program is easier to hit.  It is the
mean of 1/lines over the admitted cases: the localisation a model would reach
by quoting one uniformly random line.  That is a floor, not a model of how
models behave -- they avoid blank lines and signatures -- so read "no better
than chance" as "indistinguishable from the floor", not as a claim that the
model is guessing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from math import comb, erfc, sqrt


def fisher_two_tailed(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher exact test on a 2x2 table.

    Exact rather than chi-squared because the counts here are small enough
    that the approximation misbehaves, and because an exact test needs no
    justification about when it applies.
    """
    total = a + b + c + d
    if total == 0:
        return 1.0

    def probability(w: int, x: int, y: int, z: int) -> float:
        return comb(w + x, w) * comb(y + z, y) / comb(total, w + y)

    observed = probability(a, b, c, d)
    accumulated = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k, l = a + b - i, a + c - i, d - (a - i)
        if j < 0 or k < 0 or l < 0:
            continue
        candidate = probability(i, j, k, l)
        # The tolerance matters: tables with identical probability to the
        # observed one belong in the tail, and float equality would drop them.
        if candidate <= observed + 1e-12:
            accumulated += candidate
    return min(accumulated, 1.0)


def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel across strata.  Returns ``(statistic, p)``.

    One model's 36 cases cannot settle a question at these effect sizes, so
    the obvious move is to pool several models and gain sample size.  Pooling
    the counts directly is wrong, and not subtly wrong: if the models differ
    in baseline localisation -- they do -- the pooled table can point the
    opposite way from every model in it.  That is Simpson's paradox, and
    ``test_pooling_can_reverse_what_every_stratum_says`` constructs a case
    where it happens with counts small enough to check by hand.

    Stratifying by model is the fix.  Each model contributes its own
    expectation and variance, and the sum is compared against a chi-squared
    with one degree of freedom.

    Two honest limits, because this function is weaker than the other two
    here and the difference matters:

    1. It is **asymptotic**, unlike ``fisher_two_tailed``.  On this
       repository's own headline table the exact test gives p = 0.2129 and
       this gives 0.1713.  It is anti-conservative at these counts -- it
       flatters the claim -- so it must never be quoted as the headline.  It
       is here to answer "does the drop survive when several models are
       considered together", not to replace the exact test on one of them.
    2. It tests a *common* direction across strata.  If two models genuinely
       move opposite ways it will report no effect, which is the correct
       answer to the question asked and the wrong answer to "did anything
       happen".  Read the per-model rows as well; they are printed first for
       that reason.
    """
    numerator = 0.0
    variance = 0.0
    for a, b, c, d in tables:
        total = a + b + c + d
        # A stratum of one observation has no variance to contribute and
        # would divide by zero below.  Skipped rather than special-cased into
        # a fake contribution.
        if total < 2:
            continue
        numerator += a - (a + b) * (a + c) / total
        variance += (
            (a + b) * (c + d) * (a + c) * (b + d) / (total * total * (total - 1))
        )
    if variance == 0:
        return 0.0, 1.0
    statistic = numerator * numerator / variance
    # The exact upper tail of chi-squared with one degree of freedom is
    # erfc(sqrt(x/2)).  No approximation and no table lookup: the only
    # approximation in this function is the asymptotic one above.
    return statistic, erfc(sqrt(statistic / 2))


def binomial_at_least(successes: int, trials: int, rate: float) -> float:
    return sum(
        comb(trials, i) * rate**i * (1 - rate) ** (trials - i)
        for i in range(successes, trials + 1)
    )


def binomial_at_most(successes: int, trials: int, rate: float) -> float:
    return sum(
        comb(trials, i) * rate**i * (1 - rate) ** (trials - i)
        for i in range(0, successes + 1)
    )


def summarise(path: pathlib.Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    admitted = [r for r in run["results"] if r["outcome"] == "admitted"]
    if not admitted:
        raise SystemExit(f"{path}: no admitted findings to test")
    localised = [r for r in admitted if r["localised"]]
    missed = [r for r in admitted if not r["localised"]]
    near = [r for r in missed if r.get("miss_lines") is not None]
    lines = [r["source_lines"] for r in admitted]
    # Read from the recorded flags, never from argument order.  The previous
    # version of this tool took (published, renamed) positionally and checked
    # the flags afterwards to catch a swap.  With a third arm that check
    # stops being expressible, and a filename is not evidence of anything --
    # so the arm is derived from what the run wrote down about itself.
    if run["renamed"]:
        arm = "renamed"
    elif run.get("normalised"):
        arm = "normalised"
    else:
        arm = "published"
    return {
        "path": path.name,
        "model": run["model"],
        "renamed": run["renamed"],
        "arm": arm,
        "cases": run["cases"],
        "admitted": len(admitted),
        "localised": len(localised),
        "adjacent": sum(1 for r in near if r["miss_lines"] <= 1),
        "scattered": sum(1 for r in near if r["miss_lines"] > 1),
        # Per case, not assumed: a 5-line program is far easier to hit than a
        # 35-line one, and averaging the ratio is not the same as the ratio of
        # the averages.
        "chance": sum(1 / n for n in lines if n) / len(lines),
        "mean_lines": sum(lines) / len(lines),
    }


ORDER = ("published", "normalised", "renamed")


def compare(first: dict, second: dict) -> dict:
    """Print one arm-to-arm comparison and return its 2x2 tables."""
    left, right = first["arm"], second["arm"]
    localisation = (
        first["localised"],
        first["admitted"] - first["localised"],
        second["localised"],
        second["admitted"] - second["localised"],
    )
    clustering = (
        first["adjacent"],
        first["scattered"],
        second["adjacent"],
        second["scattered"],
    )
    drop = fisher_two_tailed(*localisation)
    print(
        f"  localisation, {left} vs {right}: "
        f"{first['localised']}/{first['admitted']} vs "
        f"{second['localised']}/{second['admitted']}  Fisher p = {drop:.4f}"
        f"{'   significant' if drop < 0.05 else '   NOT SIGNIFICANT'}"
    )
    near = fisher_two_tailed(*clustering)
    print(
        f"  misses within one line, {left} vs {right}: "
        f"{first['adjacent']}/{first['adjacent'] + first['scattered']} vs "
        f"{second['adjacent']}/{second['adjacent'] + second['scattered']}  "
        f"Fisher p = {near:.5f}"
        f"{'   significant' if near < 0.05 else '   NOT SIGNIFICANT'}"
    )
    return {"pair": (left, right), "localisation": localisation, "clustering": clustering}


def report(arms: dict[str, dict]) -> dict:
    """Print one model's arms and every adjacent comparison between them."""
    present = [name for name in ORDER if name in arms]
    for name in present:
        arm = arms[name]
        rate = arm["localised"] / arm["admitted"]
        print(
            f"{name:11s} admitted {arm['admitted']}/{arm['cases']}  "
            f"localised {arm['localised']}/{arm['admitted']} = {rate:.3f}  "
            f"chance {arm['chance']:.3f}  "
            f"ratio {rate / arm['chance']:.2f}x"
        )

    print()
    for name in present:
        arm = arms[name]
        above = binomial_at_least(arm["localised"], arm["admitted"], arm["chance"])
        below = binomial_at_most(arm["localised"], arm["admitted"], arm["chance"])
        print(
            f"  {name} vs chance: P(>={arm['localised']}) = {above:.4f}, "
            f"P(<={arm['localised']}) = {below:.4f}"
            f"{'   above chance' if above < 0.05 else ''}"
            f"{'   indistinguishable from chance' if above >= 0.05 and below >= 0.05 else ''}"
        )

    print()
    # Adjacent arms first, because those are the comparisons that isolate one
    # variable each: published to normalised is the formatting, normalised to
    # renamed is the names.  published to renamed is printed last and moves
    # both at once, which is the number that needed decomposing.
    comparisons = {}
    for left, right in zip(present, present[1:]):
        comparisons[(left, right)] = compare(arms[left], arms[right])
    if len(present) == 3:
        print()
        print("  both variables at once, for comparison with the two above:")
        comparisons[(present[0], present[-1])] = compare(
            arms[present[0]], arms[present[-1]]
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test dataset runs for significance.  Any number of run "
        "files; the arm and model of each is read from the file itself."
    )
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    options = parser.parse_args()

    by_model: dict[str, dict[str, dict]] = {}
    for path in options.runs:
        summary = summarise(path)
        arms = by_model.setdefault(summary["model"], {})
        if summary["arm"] in arms:
            raise SystemExit(
                f"{path.name} and {arms[summary['arm']]['path']} are both the "
                f"{summary['arm']} arm of {summary['model']}: one of them is "
                "not the run it was taken for, and averaging two runs of the "
                "same arm is not what any test here does"
            )
        arms[summary["arm"]] = summary

    strata = []
    for index, (model, arms) in enumerate(by_model.items()):
        if len(arms) < 2:
            raise SystemExit(
                f"{model} has only the {next(iter(arms))} arm: a single arm "
                "has nothing to be compared against, and reporting it alone "
                "would be the bare ratio this tool exists to qualify"
            )
        if index:
            print()
            print("-" * 72)
        print(f"model: {model}")
        strata.append((model, report(arms)))

    if len(strata) < 2:
        return 0

    # Only pairs every model actually ran.  Stratifying over a comparison one
    # model is missing would silently change which models the pooled number
    # describes, and the label would still say all of them.
    shared = set.intersection(*(set(c) for _, c in strata))
    if not shared:
        return 0

    print()
    print("=" * 72)
    models = [model for model, _ in strata]
    print(f"stratified across {len(strata)} models: {', '.join(models)}")
    print(
        "asymptotic, and anti-conservative at these counts -- read the exact "
        "per-model tests above as the headline"
    )
    for pair in [p for p in [(a, b) for a in ORDER for b in ORDER] if p in shared]:
        for label, key in (("localisation", "localisation"), ("near misses", "clustering")):
            tables = [comparisons[pair][key] for _, comparisons in strata]
            statistic, pooled = mantel_haenszel(tables)
            print(
                f"{label}, {pair[0]} vs {pair[1]}, stratified by model: "
                f"CMH chi2 = {statistic:.4f}  p = {pooled:.4f}"
                f"{'   significant' if pooled < 0.05 else '   NOT SIGNIFICANT'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
