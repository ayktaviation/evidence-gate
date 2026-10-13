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

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_gate import REASONS, __version__  # noqa: E402

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


CHECKS = (
    check_test_count,
    check_reason_count,
    check_every_reason_is_documented,
    check_no_placeholder_urls,
    check_version_is_recorded,
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
