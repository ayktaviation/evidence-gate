#!/usr/bin/env python3
"""Run the gate at a realistic size: the gate reviewing its own source.

    python examples/at_scale.py --model devstral:24b

The published measurement uses one 88-byte file, which is nine data points
against a toy.  Thresholds chosen against a toy are suspect, so this script
runs the same procedure against two real modules (~18 KB) and reports what the
thresholds do when the subject is a thousand times larger.  Nothing leaves the
machine; requires a local Ollama.

Use ``--bounds-from-the-toy`` to run with the defaults as they were fitted to
the 88-byte file, which is how the scale defect was found.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_gate import EvidenceSource, GateError, check  # noqa: E402
from evidence_gate.ollama import ReviewerUnavailable, ask  # noqa: E402

SUBJECTS = ("src/evidence_gate/gate.py", "src/evidence_gate/ollama.py")

REQUIREMENTS = (
    "Every refusal must name a reason that is declared in advance.",
    "A diagnostic must never contain the text that was rejected.",
    "A quote must be compared without normalising whitespace or case.",
    "The gate must stop at the first inadmissible finding.",
)

ABSENCE_PREFIX = "absent:"

SYSTEM = (
    "You review source code against a list of requirements. "
    "Reply with JSON only: {\"findings\": [{\"requirement\": ..., "
    "\"evidence\": ..., \"detail\": ...}]}. "
    "'requirement' must be copied exactly from the requirements, as one whole "
    "requirement. "
    "For a defect in code that IS present, 'evidence' must be copied exactly "
    "from the source, character for character, as a short span of one or two "
    "lines -- never a whole function. "
    f"For a defect caused by MISSING code, write {ABSENCE_PREFIX!r} followed by "
    "a short literal fragment that would have to appear if the requirement "
    "were met, and that does not appear. The reviewer searches for it; if it "
    "is found, your finding is rejected. Keep that fragment under 20 "
    "characters -- a name or a call, not a statement you invented. "
    "Report at most four findings. Never write a quote you did not copy."
)


def _unfenced(text: str) -> str:
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
    parser.add_argument("--model", default="devstral:24b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--think", choices=("on", "off", "default"), default="off")
    parser.add_argument(
        "--bounds-from-the-toy",
        action="store_true",
        help="use the fraction-only absence ceiling fitted to the 88-byte file",
    )
    options = parser.parse_args()

    sources = [
        EvidenceSource(name=name, kind="source", text=(ROOT / name).read_text())
        for name in SUBJECTS
    ]
    total = sum(len(source.text) for source in sources)
    smallest = min(len(source.text) for source in sources)
    print(f"subject: {len(sources)} files, {total} chars, smallest {smallest}")
    print(
        "the toy was 88 chars, so a fraction-based ceiling of 0.5 now permits "
        f"an absence pattern of up to {int(smallest * 0.5)} characters"
    )

    prompt = "Requirements:\n" + "\n".join(f"- {line}" for line in REQUIREMENTS)
    for source in sources:
        prompt += f"\n\nSource of {source.name}:\n{source.text}"
    prompt += "\n\nReport defects. JSON only."

    try:
        answer = ask(
            prompt,
            model=options.model,
            system=SYSTEM,
            endpoint=options.endpoint,
            seed=options.seed,
            num_predict=1536,
            think={"on": True, "off": False, "default": None}[options.think],
        )
    except ReviewerUnavailable as failure:
        print(f"reviewer unavailable: {failure}", file=sys.stderr)
        return 2

    print("\nmodel receipt:", json.dumps(answer.receipt(), indent=2))

    try:
        findings = json.loads(_unfenced(answer.text))["findings"]
    except Exception:
        print("\nresult: the model did not return the requested JSON shape.")
        return 1

    bounds: dict[str, object] = {}
    if options.bounds_from_the_toy:
        bounds["max_absence_pattern_chars"] = None
    try:
        receipt = check(
            findings,
            requirements=REQUIREMENTS,
            sources=sources,
            absence_prefix=ABSENCE_PREFIX,
            require_whole_clause=True,
            **bounds,
        )
    except GateError as refusal:
        print(f"\nREFUSED: {refusal.reason}")
        print(json.dumps(refusal.diagnostic, indent=2, sort_keys=True))
        return 1

    print(
        f"\nADMITTED: {receipt['finding_count']} finding(s) -- "
        f"{receipt['quoted_finding_count']} quoted, "
        f"{receipt['verified_absence_count']} verified absences"
    )
    for row in receipt["findings"]:
        if row["quoted"]:
            spread = sorted({match["source"] for match in row["matches"]})
            print(
                f"  [{row['index']}] quote of {row['evidence_bytes']} bytes, "
                f"{row['match_count']} match(es) in {spread}"
            )
        else:
            print(
                f"  [{row['index']}] absence pattern of "
                f"{row['absent_pattern_chars']} chars, "
                f"fraction {row['absent_pattern_fraction']}"
            )
    print(
        "\nRead the absence fractions. Against an 88-byte file a 57-character "
        "pattern scored 0.65 and was refused; the same pattern here scores "
        "about 0.007. A fraction-based ceiling dissolves exactly where real "
        "code lives, which is why the absolute cap exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
