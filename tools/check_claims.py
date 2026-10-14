#!/usr/bin/env python3
"""Fail if the README states something the code does not.

This project's argument is that an unchecked claim is worth less than an
absent one.  A README asserting "39 tests" and "eleven declared reasons" is
exactly such a claim, and documentation drifts silently while code does not.
So the prose is checked against the code on every push, the same way the
suite checks that every declared refusal has a control.

Run it directly; it prints what it verified and exits non-zero on drift.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_gate import REASONS, EvidenceSource, __version__, check  # noqa: E402

# Read off a real receipt rather than grepped from the source.  A field name
# that appears in a dict literal is not proof the field is emitted; running
# the gate once is.
#
# The probe source is padded deliberately.  The first attempt quoted twelve
# characters of a seventeen-character file and the gate refused it with
# evidence_covers_whole_source -- correctly, since that quote localises
# nothing.  Worth recording that the tooling had to satisfy the same bound as
# a model does.
RECEIPT_FIELDS = frozenset(
    check(
        [{"requirement": "A probe.", "evidence": "a probe line", "detail": "."}],
        requirements=("A probe.",),
        sources=[
            EvidenceSource(
                name="probe",
                kind="source",
                text="padding before\na probe line\npadding after it all\n",
            )
        ],
    )
)

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
}


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def check_test_count(report: list[str]) -> list[str]:
    """The README advertises a test count; count the tests."""
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    actual = suite.countTestCases()
    if not actual:
        return ["discovered zero tests, which means discovery is broken"]
    claimed = {int(found) for found in re.findall(r"(\d+) tests", _readme())}
    if not claimed:
        return ["README no longer states a test count; the claim check is dead"]
    report.append(f"test count: README says {sorted(claimed)}, suite has {actual}")
    if claimed != {actual}:
        return [f"README claims {sorted(claimed)} tests, suite has {actual}"]
    return []


def check_reason_count(report: list[str]) -> list[str]:
    """The README says how many ways the gate can refuse, in words."""
    readme = _readme()
    expected = NUMBER_WORDS.get(len(REASONS))
    if expected is None:
        return [f"no word for {len(REASONS)} reasons; extend NUMBER_WORDS"]
    report.append(f"reason count: {len(REASONS)} ({expected})")
    if f"{expected} declared reasons" not in readme:
        found = re.search(r"(\w+) declared reasons", readme)
        return [
            f"README says {found.group(1)!r} declared reasons, code has "
            f"{len(REASONS)} ({expected})"
            if found
            else "README no longer states a reason count"
        ]
    return []


def check_every_reason_is_documented(report: list[str]) -> list[str]:
    """Every refusal the code can raise must appear in the README's table.

    The suite already guarantees each reason has a control.  This guarantees
    each reason has an explanation, so a caller branching on a reason name can
    find out what it means.
    """
    readme = _readme()
    undocumented = sorted(
        reason for reason in REASONS if f"`{reason}`" not in readme
    )
    report.append(f"documented reasons: {len(REASONS) - len(undocumented)}/{len(REASONS)}")
    if undocumented:
        return [f"reasons raised but not in the README table: {undocumented}"]
    return []


def check_no_placeholder_urls(report: list[str]) -> list[str]:
    """A placeholder link shipped once is a dead link forever."""
    problems = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git/" in str(path.relative_to(ROOT)):
            continue
        if path.suffix not in {".md", ".toml", ".py", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Assembled rather than written out, so this file does not match its
        # own patterns.  A checker that reports itself trains you to ignore it.
        for marker in ("github.com/" + "OWNER", "github.com/" + "USER", "exam" + "ple.com"):
            if marker in text:
                problems.append(f"{path.relative_to(ROOT)} contains {marker!r}")
    report.append("placeholder URLs: none" if not problems else "placeholder URLs: found")
    return problems


def check_version_is_recorded(report: list[str]) -> list[str]:
    """The version in code, the packaging metadata and the changelog agree."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    report.append(f"version: {__version__}")
    problems = []
    if f'version = "{__version__}"' not in pyproject:
        problems.append(f"pyproject.toml does not declare version {__version__}")
    if f"## {__version__}" not in changelog:
        problems.append(f"CHANGELOG.md has no entry for {__version__}")
    return problems


def check_the_blind_spot_is_documented(report: list[str]) -> list[str]:
    """The gate's unclosable hole stays described, and stays demonstrated.

    This check guards against a specific, tempting kind of drift.  The known
    weakness here -- that a real quote reused across unrelated requirements is
    admitted -- is embarrassing to advertise, and the quiet way to stop
    advertising it is to delete a README section while leaving every test
    green.  The other checks in this file would not notice.

    So the section, the two receipt fields that make the weakness countable,
    and the adversarial test file that proves it are all required to coexist.
    Removing any one of them fails the build, which is the point: a limitation
    that can be deleted without consequence is a limitation that will be.
    """
    readme = _readme()
    problems = []
    if "## The attack this gate does not stop" not in readme:
        problems.append("README no longer documents the gate's known blind spot")
    for field in ("distinct_evidence_count", "reused_evidence_count"):
        if field not in readme:
            problems.append(f"README does not mention the {field} receipt field")
        if field not in RECEIPT_FIELDS:
            problems.append(f"the receipt no longer reports {field}")
    if not (ROOT / "tests" / "test_compliant_parasite.py").exists():
        problems.append("the adversarial control for the blind spot is gone")
    report.append(
        f"blind spot: documented, {len(RECEIPT_FIELDS & {'distinct_evidence_count', 'reused_evidence_count'})}/2 fields reported"
    )
    return problems


def check_the_skipped_control_count(report: list[str]) -> list[str]:
    """The README says how many controls a dataset checkout unlocks.

    Added because that number was wrong.  The README claimed five while the
    suite had grown to eight, and nothing here noticed -- an unverified count
    in a repository whose argument is that an unverified claim is worth less
    than an absent one.  The number is small, which is exactly why it rotted:
    nobody recounts a number they can nearly remember.

    Counted in a subprocess with the variable removed from the environment,
    which is the same thing a reader does when they run the command the README
    prints.  Doing it in this process does not work, and the first version of
    this check was wrong in exactly that way: the gated module reads the
    variable once at import time, discovery has already imported it by the
    time this runs, and unsetting the variable afterwards changes nothing.
    The check passed with the variable unset and failed with it set -- a
    checker whose answer depended on the environment it happened to be
    invoked in.
    """
    readme = _readme()
    claimed = {
        int(found)
        for found in re.findall(r"(\d+) harness controls stop skipping", readme)
    }
    if not claimed:
        return ["README no longer states how many controls the dataset unlocks"]

    environment = {k: v for k, v in os.environ.items() if k != "EVIDENCE_GATE_DATASET"}
    finished = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
    )
    actual = finished.stderr.count("... skipped")
    report.append(
        f"dataset controls: README says {sorted(claimed)}, suite skips {actual}"
    )
    if finished.returncode != 0:
        return [
            "the suite does not pass without a dataset, so the skip count "
            "cannot be trusted"
        ]
    if claimed != {actual}:
        return [
            f"README claims {sorted(claimed)} controls unlock with a dataset, "
            f"suite skips {actual} without one"
        ]
    return []


def check_the_measurement_arithmetic(report: list[str]) -> list[str]:
    """Recompute the README's statistics from the committed counts.

    Every other number in this README is checked against the code, and the
    statistics were the exception: they were computed once from run files that
    are deliberately not in this repository and then typed into prose.  A
    reader could not check them and neither could I, which is the position
    this project argues against.

    So the counts the tests consumed are committed (no case names -- the
    holdout is not publishable) and the arithmetic is redone here on every
    push.  This checks the claims, not just the digits: that formatting shows
    no effect, that renaming does, that every renamed arm sits at the chance
    floor, and that the floor is *higher* in the reformatted arms -- the
    asymmetry that runs in favour of the conclusion and would be the first
    thing to quietly rot.
    """
    aggregate = ROOT / "measurements" / "quixbugs-2026-10-13.json"
    if not aggregate.exists():
        return ["the committed measurement counts are gone"]

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "significance", ROOT / "tools" / "significance.py"
    )
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)

    import json

    data = json.loads(aggregate.read_text(encoding="utf-8"))
    arms = data["arms"]
    readme = _readme()
    problems = []

    def table(first, second):
        return (
            first["localised"],
            first["admitted"] - first["localised"],
            second["localised"],
            second["admitted"] - second["localised"],
        )

    for model, by_arm in arms.items():
        for arm, counts in by_arm.items():
            fraction = f"{counts['localised']}/{counts['admitted']}"
            if fraction not in readme:
                problems.append(
                    f"README does not state {fraction} for {model} {arm}"
                )

        formatting = stats.fisher_two_tailed(
            *table(by_arm["published"], by_arm["normalised"])
        )
        if formatting <= 0.05:
            problems.append(
                f"{model}: reformatting now shows an effect (p = {formatting:.4f}); "
                "the README says it does not"
            )
        # The renamed arm must stay indistinguishable from quoting at random.
        renamed = by_arm["renamed"]
        floor = stats.binomial_at_least(
            renamed["localised"], renamed["admitted"], renamed["chance"]
        )
        if floor <= 0.05:
            problems.append(
                f"{model}: the renamed arm now beats chance (p = {floor:.4f}); "
                "the README says it does not"
            )
        if renamed["chance"] <= by_arm["published"]["chance"]:
            problems.append(
                f"{model}: the renamed chance floor is no longer the higher "
                "one, so the README's asymmetry argument no longer holds"
            )

    pooled_formatting = stats.mantel_haenszel(
        [table(a["published"], a["normalised"]) for a in arms.values()]
    )[1]
    pooled_names = stats.mantel_haenszel(
        [table(a["normalised"], a["renamed"]) for a in arms.values()]
    )[1]
    if pooled_formatting <= 0.05:
        problems.append(
            f"stratified formatting effect is now significant (p = {pooled_formatting:.4f})"
        )
    if pooled_names >= 0.001:
        problems.append(
            f"stratified naming effect is no longer significant (p = {pooled_names:.4f})"
        )

    # The scope of the claim, not just its direction.  A measurement quoted
    # without its sample size is the overclaim this section exists to avoid.
    if len(arms) < 3:
        problems.append(f"the README describes three models, the counts have {len(arms)}")
    paired = {counts["cases"] for by_arm in arms.values() for counts in by_arm.values()}
    if paired != {35} or "35 paired cases" not in readme:
        problems.append(f"README and counts disagree on the paired case count: {paired}")

    report.append(
        f"measurement: {len(arms)} models, formatting p = {pooled_formatting:.4f} NS, "
        f"names p = {pooled_names:.6f}, all renamed arms at chance"
    )
    return problems


CHECKS = (
    check_test_count,
    check_reason_count,
    check_every_reason_is_documented,
    check_no_placeholder_urls,
    check_version_is_recorded,
    check_the_blind_spot_is_documented,
    check_the_skipped_control_count,
    check_the_measurement_arithmetic,
)


def main() -> int:
    report: list[str] = []
    problems: list[str] = []
    for check in CHECKS:
        problems.extend(check(report))
    for line in report:
        print(f"  {line}")
    sys.stdout.flush()
    if problems:
        print("\nthe README claims something the code does not:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\n{len(CHECKS)} claims verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
