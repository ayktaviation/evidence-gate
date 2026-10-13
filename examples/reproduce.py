#!/usr/bin/env python3
"""Ask a local model to review a file, then check whether its quotes are real.

    python examples/reproduce.py --model qwen3:8b

Requires a running Ollama on 127.0.0.1:11434.  Nothing leaves the machine.
The script reports what happened and prints a content-free receipt; it makes
no claim about whether the model's opinion was correct, only about whether
its quotes exist in the file it was shown.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evidence_gate import EvidenceSource, GateError, check  # noqa: E402
from evidence_gate.ollama import ReviewerUnavailable, ask  # noqa: E402

REQUIREMENTS = (
    "A balance must never be negative. "
    "Every withdrawal must be recorded. "
    "The caller must be told when a withdrawal is refused."
)

SUBJECT = """def withdraw(account, amount):
    account.balance -= amount
    return account.balance
"""

ABSENCE_PREFIX = "absent:"

SYSTEM = (
    "You review source code against a list of requirements. "
    "Reply with JSON only: {\"findings\": [{\"requirement\": ..., "
    "\"evidence\": ..., \"detail\": ...}]}. "
    "'requirement' must be copied exactly from the requirements. "
    "For a defect in code that IS present, 'evidence' must be copied exactly "
    "from the source code, character for character, as a short span -- never "
    "the whole file. "
    f"For a defect caused by MISSING code, write {ABSENCE_PREFIX!r} followed by "
    "a short literal fragment that would have to appear in the file if the "
    "requirement were met, and that does not appear in it. The reviewer will "
    "search the file for that fragment; if it is found, your finding is "
    "rejected. Keep that fragment under 20 characters -- a name or a call, "
    "such as 'log(' or 'raise ', not a whole statement you invented, because a "
    "long invented line is missing from the file no matter what the code does. "
    "Never write a quote you did not copy."
)

EXPLANATIONS = {
    "evidence_not_in_context": (
        "The model attached a quote that appears nowhere in the file it was "
        "shown. It did not misread the code; it wrote code that does not "
        "exist and attributed it to the file."
    ),
    "evidence_covers_whole_source": (
        "The model pasted (almost) the entire file as its quote. Every "
        "character matched, so a plain substring test would admit it, but the "
        "quote points at nothing in particular."
    ),
    "evidence_too_short": (
        "The quote is too short to identify anything; a few characters match "
        "some source by accident."
    ),
    "requirement_not_exact_quote": (
        "The model reviewed against a requirement it paraphrased rather than "
        "one that was given to it."
    ),
    "absence_pattern_present": (
        "The model declared that some code is missing and named the text it "
        "claimed was absent -- and that text is in the file. This is the one "
        "kind of absence claim the gate can settle on its own, and it is "
        "refuted, not merely unsupported."
    ),
    "absence_pattern_too_short": (
        "The absence claim named too little text to be meaningful. A fragment "
        "of one or two characters is absent from almost nothing."
    ),
    "absence_pattern_too_broad": (
        "The absence claim named so much text that its absence carries no "
        "information: a whole invented statement is missing from a short file "
        "whether or not the requirement is met. This is the same vacuous pass "
        "as quoting the entire file, reached from the other direction."
    ),
    "_other": "The findings were refused before any judgement of their merit.",
}

PROMPT = f"""Requirements:
{REQUIREMENTS}

Source code of account.py:
{SUBJECT}
Report defects. JSON only."""


def _unfenced(text: str) -> str:
    """Strip markdown fences and surrounding prose before parsing.

    Deliberately lenient: wrapping JSON in `````json`` is a chat convention,
    not a claim about code, and counting it as an evidence failure would make
    the measurement unfair. Leniency stops here -- the quotes inside are
    compared exactly.
    """
    stripped = text.strip()
    if "```" in stripped:
        block = stripped.split("```")[1]
        if block.lower().startswith("json"):
            block = block[4:]
        stripped = block.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else stripped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--think",
        choices=("on", "off", "default"),
        default="off",
        help="reasoning trace; 'off' keeps the token budget for the answer",
    )
    options = parser.parse_args()

    try:
        answer = ask(
            PROMPT,
            model=options.model,
            system=SYSTEM,
            endpoint=options.endpoint,
            seed=options.seed,
            think={"on": True, "off": False, "default": None}[options.think],
        )
    except ReviewerUnavailable as failure:
        print(f"reviewer unavailable: {failure}", file=sys.stderr)
        return 2

    print("model receipt:", json.dumps(answer.receipt(), indent=2))

    try:
        findings = json.loads(_unfenced(answer.text))["findings"]
    except Exception:
        print("\nresult: the model did not return the requested JSON shape.")
        print("That is a format failure, not an evidence failure.")
        return 1

    sources = [EvidenceSource(name="account.py", kind="source", text=SUBJECT)]
    try:
        receipt = check(
            findings,
            requirements=REQUIREMENTS,
            sources=sources,
            absence_prefix=ABSENCE_PREFIX,
        )
    except GateError as refusal:
        print(f"\nREFUSED: {refusal.reason}")
        print(json.dumps(refusal.diagnostic, indent=2, sort_keys=True))
        print("\n" + EXPLANATIONS.get(refusal.reason, EXPLANATIONS["_other"]))
        print(
            "The reason above carries hashes and byte counts only. A rejected "
            "quote is often fabricated text, so it is never echoed."
        )
        return 1

    print(
        f"\nADMITTED: {receipt['finding_count']} finding(s) -- "
        f"{receipt['quoted_finding_count']} located in the source, "
        f"{receipt['verified_absence_count']} named missing code that the gate "
        f"searched for and did not find"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(
        "\nEvery quote was located in the source, and every absence claim was "
        "searched for and not found. That makes the findings admissible, not "
        "correct: an absence verified in this file is still only a lead about "
        "the requirement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
