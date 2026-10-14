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
    return {
        "path": path.name,
        "model": run["model"],
        "renamed": run["renamed"],
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


def report(first: dict, second: dict) -> dict:
    """Print one model's comparison and return its 2x2 tables for pooling."""
    for arm in (first, second):
        rate = arm["localised"] / arm["admitted"]
        print(
            f"{'renamed ' if arm['renamed'] else 'published'}  "
            f"admitted {arm['admitted']}/{arm['cases']}  "
            f"localised {arm['localised']}/{arm['admitted']} = {rate:.3f}  "
            f"chance {arm['chance']:.3f}  "
            f"ratio {rate / arm['chance']:.2f}x"
        )

    print()
    drop = fisher_two_tailed(
        first["localised"],
        first["admitted"] - first["localised"],
        second["localised"],
        second["admitted"] - second["localised"],
    )
    print(
        f"localisation, published vs renamed: "
        f"{first['localised']}/{first['admitted']} vs "
        f"{second['localised']}/{second['admitted']}  Fisher p = {drop:.4f}"
        f"{'' if drop < 0.05 else '   NOT SIGNIFICANT'}"
    )

    for arm in (first, second):
        above = binomial_at_least(arm["localised"], arm["admitted"], arm["chance"])
        below = binomial_at_most(arm["localised"], arm["admitted"], arm["chance"])
        label = "renamed " if arm["renamed"] else "published"
        print(
            f"{label} vs chance: P(>={arm['localised']}) = {above:.4f}, "
            f"P(<={arm['localised']}) = {below:.4f}"
            f"{'   above chance' if above < 0.05 else ''}"
            f"{'   indistinguishable from chance' if above >= 0.05 and below >= 0.05 else ''}"
        )

    print()
    clustering = fisher_two_tailed(
        first["adjacent"],
        first["scattered"],
        second["adjacent"],
        second["scattered"],
    )
    print(
        f"misses within one line of the defect: "
        f"{first['adjacent']}/{first['adjacent'] + first['scattered']} vs "
        f"{second['adjacent']}/{second['adjacent'] + second['scattered']}  "
        f"Fisher p = {clustering:.5f}"
        f"{'   significant' if clustering < 0.05 else '   NOT SIGNIFICANT'}"
    )
    return {
        "model": first["model"],
        "localisation": (
            first["localised"],
            first["admitted"] - first["localised"],
            second["localised"],
            second["admitted"] - second["localised"],
        ),
        "clustering": (
            first["adjacent"],
            first["scattered"],
            second["adjacent"],
            second["scattered"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test one or more (published, renamed) run pairs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=pathlib.Path,
        help="published and renamed run JSON, in that order, one pair per model",
    )
    options = parser.parse_args()

    if len(options.runs) % 2:
        raise SystemExit(
            f"{len(options.runs)} files given: runs come in (published, renamed) "
            "pairs, and an odd count means one arm is missing -- which pair is "
            "incomplete is not something this tool should guess at"
        )

    strata = []
    for published, renamed in zip(options.runs[::2], options.runs[1::2]):
        first, second = summarise(published), summarise(renamed)
        if first["renamed"] or not second["renamed"]:
            raise SystemExit(
                f"{published.name} and {renamed.name}: arguments are "
                "(published-names run, renamed run); the recorded 'renamed' "
                "flags say otherwise, and comparing the wrong pair would "
                "produce a confident number about nothing"
            )
        if first["model"] != second["model"]:
            raise SystemExit(
                f"different models ({first['model']} vs {second['model']}): the "
                "contamination control only means anything within one model"
            )
        if strata:
            print()
            print("-" * 72)
        print(f"model: {first['model']}")
        strata.append(report(first, second))

    if len(strata) < 2:
        return 0

    models = [s["model"] for s in strata]
    if len(set(models)) != len(models):
        raise SystemExit(
            "the same model appears in more than one pair: stratifying by "
            "model only controls for the model if each stratum is a different "
            "one, and repeating a stratum would count the same cases twice"
        )

    print()
    print("=" * 72)
    print(f"stratified across {len(strata)} models: {', '.join(models)}")
    print(
        "asymptotic, and anti-conservative at these counts -- read the exact "
        "per-model tests above as the headline"
    )
    for label, key in (("localisation", "localisation"), ("near misses", "clustering")):
        statistic, pooled = mantel_haenszel([s[key] for s in strata])
        print(
            f"{label}, published vs renamed, stratified by model: "
            f"CMH chi2 = {statistic:.4f}  p = {pooled:.4f}"
            f"{'   significant' if pooled < 0.05 else '   NOT SIGNIFICANT'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
