#!/usr/bin/env python3
"""Measure the gap between *admissible* and *correct*, against ground truth.

    python examples/against_a_dataset.py --dataset /path/to/QuixBugs --model devstral:24b

Every number this repository has published so far describes whether a model's
evidence was admissible: did the quote exist where the model said it did.  The
README is explicit that admissible does not mean correct, and until now that
was a disclaimer rather than a measurement, because nothing here knew where
the defects actually were.

A defect dataset knows.  QuixBugs ships each program twice, once broken and
once fixed, so the difference between the two versions is a third party's
statement of where the bug is.  That turns the disclaimer into a number:

    localisation = (admitted findings whose quote overlaps the real defect)
                 / (admitted findings)

Three things in this experiment are authored by someone else, which is the
whole point.  The code is theirs.  The defect is theirs.  And the requirement
handed to the model is their own module docstring, so the specification is not
something I wrote to suit my own gate.

The dataset is NOT vendored into this repository and must be pointed at from
outside it.  Same discipline as the other external subject here: a result you
can check without me republishing someone else's work, and no confound from
the subject drifting whenever this library changes.

CONTAMINATION, STATED UP FRONT: QuixBugs is public, dates from 2017, and is
almost certainly in the training data of every model measured here.  A high
localisation score is therefore NOT evidence that a model can find bugs.  It
is an upper bound, measured under the most favourable conditions available.
Use --rename to re-run with identifiers stripped of meaning; the drop between
the two runs is the part of the score that memorised names were carrying.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_gate import EvidenceSource, GateError, check  # noqa: E402
from evidence_gate.ollama import ReviewerUnavailable, ask  # noqa: E402

ABSENCE_PREFIX = "absent:"

SYSTEM = (
    "You review one short program against one requirement. "
    'Reply with JSON only: {"findings": [{"requirement": ..., '
    '"evidence": ..., "detail": ...}]}. '
    "'requirement' must be copied exactly from the requirement given. "
    "'evidence' must be copied exactly from the program, character for "
    "character, as ONE line or a short span -- the line you believe is wrong. "
    f"If the defect is missing code, write {ABSENCE_PREFIX!r} followed by a "
    "short literal fragment that would have to appear if the requirement were "
    "met and that does not appear; keep it under 20 characters. "
    "Report at most two findings. Never write a quote you did not copy."
)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
# A naive diff of the two versions is not ground truth.  Measured on this
# dataset: the broken files carry their specification as a trailing module
# docstring and the fixed files do not, so a plain `diff` reports eighteen
# changed lines for levenshtein when the defect is one character.  Comparing
# code only, with docstrings and comments removed, is what isolates the bug.


def _code_lines(source: str) -> list[str] | None:
    """Significant code lines, with docstrings, comments and blanks removed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end = node.end_lineno or node.lineno
            docstring_lines.update(range(node.lineno, end + 1))
    kept = []
    for number, line in enumerate(source.splitlines(), 1):
        if number in docstring_lines:
            continue
        if not line.strip() or line.strip().startswith("#"):
            continue
        kept.append(line)
    return kept


def _requirement(source: str) -> str | None:
    """The dataset author's own specification, flattened to one clause."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            text = " ".join(node.value.value.split())
            if len(text) > 24:
                return text
    return None


def _without_docstrings(source: str) -> str:
    lines = _code_lines(source)
    return "\n".join(lines) + "\n" if lines else ""


def _rename(source: str) -> str | None:
    """Rewrite identifiers to opaque names, preserving behaviour.

    The control for contamination.  ``gcd`` and ``levenshtein`` are names a
    model has seen thousands of times attached to their correct
    implementations, so a quote landing on the right line proves nothing about
    reading the code in front of it.  Here every function, argument and local
    becomes ``f0``, ``p0``, ``v0``; the algorithm is untouched and the names
    carry no information.

    Reparsing through ``ast.unparse`` also normalises formatting, which is why
    ground truth is re-derived from the renamed pair rather than mapped across
    from the original.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    builtins = set(dir(__builtins__)) | {
        "self", "True", "False", "None", "range", "len", "list", "dict", "set",
        "tuple", "int", "str", "float", "bool", "sorted", "enumerate", "abs",
        "min", "max", "sum", "zip", "reversed", "map", "filter", "print",
        "append", "pop", "keys", "values", "items", "add", "extend", "join",
        "heappush", "heappop", "deepcopy", "defaultdict", "Counter",
    }
    mapping: dict[str, str] = {}

    def opaque(name: str, prefix: str) -> str:
        if name in builtins or name.startswith("__"):
            return name
        if name not in mapping:
            mapping[name] = f"{prefix}{len(mapping)}"
        return mapping[name]

    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self, node):  # noqa: N802
            node.name = opaque(node.name, "f")
            for argument in node.args.args:
                argument.arg = opaque(argument.arg, "p")
            self.generic_visit(node)
            return node

        def visit_Name(self, node):  # noqa: N802
            node.id = opaque(node.id, "v")
            return node

        def visit_arg(self, node):  # noqa: N802
            node.arg = opaque(node.arg, "p")
            return node

    stripped = ast.parse(_without_docstrings(source) or source)
    try:
        return ast.unparse(ast.fix_missing_locations(Rewrite().visit(stripped)))
    except Exception:  # noqa: BLE001 - an unrenameable case is skipped, not fixed
        return None


def cases(dataset: pathlib.Path, rename: bool) -> list[dict]:
    """Programs with exactly one changed code line, which is the ground truth.

    Only single-line defects are kept.  A multi-line defect has no single
    correct answer to point at, so scoring a quote against it would measure
    the scorer's generosity rather than the model.
    """
    broken_root = dataset / "python_programs"
    fixed_root = dataset / "correct_python_programs"
    if not broken_root.is_dir() or not fixed_root.is_dir():
        raise SystemExit(f"not a QuixBugs checkout: {dataset}")

    selected = []
    for broken_path in sorted(broken_root.glob("*.py")):
        fixed_path = fixed_root / broken_path.name
        if not fixed_path.is_file():
            continue
        broken_source = broken_path.read_text(encoding="utf-8")
        fixed_source = fixed_path.read_text(encoding="utf-8")
        requirement = _requirement(broken_source)
        if requirement is None:
            continue

        if rename:
            broken_shown = _rename(broken_source)
            fixed_shown = _rename(fixed_source)
            if broken_shown is None or fixed_shown is None:
                continue
        else:
            broken_shown = _without_docstrings(broken_source)
            fixed_shown = _without_docstrings(fixed_source)

        left, right = broken_shown.splitlines(), fixed_shown.splitlines()
        operations = [
            op for op in difflib.SequenceMatcher(None, left, right).get_opcodes()
            if op[0] != "equal"
        ]
        if len(operations) != 1:
            continue
        tag, i1, i2, j1, j2 = operations[0]
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue

        # Character span of the defective line inside what the model is shown.
        offset = sum(len(line) + 1 for line in left[:i1])
        selected.append(
            {
                "name": broken_path.stem,
                "requirement": requirement,
                "source": broken_shown,
                "defect_line": left[i1],
                "defect_start": offset,
                "defect_end": offset + len(left[i1]),
                "fixed_line": right[j1],
            }
        )
    return selected


def schema_for(requirement: str) -> dict:
    """Decoding constrained to the shape the gate requires.

    ``requirement`` is an enum of one.  Measured earlier in this repository:
    with a free string there, every local model paraphrased and every one was
    refused for an inexact requirement -- a harness defect wearing a model
    defect's clothes.  ``evidence`` is left unconstrained because it is the
    claim under test.
    """
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {"type": "string", "enum": [requirement]},
                        "evidence": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["requirement", "evidence", "detail"],
                },
            }
        },
        "required": ["findings"],
    }


def _unfenced(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        block = stripped.split("```")[1]
        if block.lower().startswith("json"):
            block = block[4:]
        stripped = block.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else stripped


def _localised(receipt: dict, case: dict) -> bool:
    """Did any admitted quote overlap the line the dataset says is wrong?

    Computed from the receipt's own match offsets, not from the model's text.
    That is deliberate: if the receipt cannot support this question, the
    receipt is underspecified, and finding that out is worth more than a
    convenient shortcut through the raw finding.
    """
    for row in receipt["findings"]:
        if not row["quoted"]:
            continue
        for match in row["matches"]:
            start = match["offset"]
            end = start + row["evidence_bytes"]
            if start < case["defect_end"] and case["defect_start"] < end:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="devstral:24b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--think", choices=("on", "off", "default"), default="off")
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="first N cases only")
    parser.add_argument(
        "--rename",
        action="store_true",
        help="strip identifiers of meaning; the contamination control",
    )
    parser.add_argument("--out", type=pathlib.Path, help="write the run as JSON")
    options = parser.parse_args()

    selected = cases(options.dataset, options.rename)
    if options.limit:
        selected = selected[: options.limit]
    if not selected:
        raise SystemExit("no single-line cases found")

    print(f"dataset: {options.dataset}")
    print(f"cases: {len(selected)} single-line defects, identifiers "
          f"{'renamed' if options.rename else 'as published'}")
    print(f"model: {options.model}")
    print(
        "contamination: this dataset is public and predates these models; "
        "read the score as an upper bound"
    )
    print()

    results = []
    for number, case in enumerate(selected, 1):
        source = EvidenceSource(
            name=f"{case['name']}.py", kind="source", text=case["source"]
        )
        prompt = (
            f"Requirement:\n{case['requirement']}\n\n"
            f"Program {source.name}:\n{case['source']}\n\n"
            "Exactly one line of this program is wrong. Report it. JSON only."
        )
        try:
            answer = ask(
                prompt,
                model=options.model,
                system=SYSTEM,
                endpoint=options.endpoint,
                seed=options.seed,
                num_predict=options.num_predict,
                think={"on": True, "off": False, "default": None}[options.think],
                schema=schema_for(case["requirement"]),
            )
        except ReviewerUnavailable as failure:
            print(f"reviewer unavailable: {failure}", file=sys.stderr)
            return 2

        row = {
            "case": case["name"],
            "defect_line_sha256": hashlib.sha256(
                case["defect_line"].encode("utf-8")
            ).hexdigest(),
            "output_bytes": answer.output_bytes,
            "done_reason": answer.done_reason,
        }

        if answer.truncated:
            # Not a model failure and not an evidence failure: the budget ended
            # the answer.  Conflating the three is how a harness produces a
            # confident number that means nothing.
            row.update(outcome="truncated")
        else:
            try:
                findings = json.loads(_unfenced(answer.text))["findings"]
            except Exception:  # noqa: BLE001
                row.update(outcome="unparseable")
            else:
                if not findings:
                    row.update(outcome="no_finding")
                else:
                    try:
                        receipt = check(
                            findings,
                            requirements=(case["requirement"],),
                            sources=[source],
                            absence_prefix=ABSENCE_PREFIX,
                        )
                    except GateError as refusal:
                        row.update(outcome="refused", reason=refusal.reason)
                    else:
                        row.update(
                            outcome="admitted",
                            quoted=receipt["quoted_finding_count"],
                            verified_absence=receipt["verified_absence_count"],
                            distinct_evidence=receipt["distinct_evidence_count"],
                            reused_evidence=receipt["reused_evidence_count"],
                            localised=_localised(receipt, case),
                        )

        mark = {
            "admitted": "LOCALISED" if row.get("localised") else "admitted",
            "refused": f"refused {row.get('reason', '')}",
        }.get(row["outcome"], row["outcome"])
        print(f"  [{number:2d}/{len(selected)}] {case['name']:28s} {mark}")
        results.append(row)

    admitted = [r for r in results if r["outcome"] == "admitted"]
    localised = [r for r in admitted if r["localised"]]
    refused = [r for r in results if r["outcome"] == "refused"]
    quoted_admitted = [r for r in admitted if r["quoted"]]

    print(f"\ncases: {len(results)}")
    print(f"admitted: {len(admitted)}")
    print(f"refused: {len(refused)}")
    for reason in sorted({r["reason"] for r in refused}):
        print(f"    {reason}: {sum(1 for r in refused if r['reason'] == reason)}")
    for outcome in ("truncated", "unparseable", "no_finding"):
        count = sum(1 for r in results if r["outcome"] == outcome)
        if count:
            print(f"{outcome}: {count}")

    if admitted:
        rate = round(len(localised) / len(admitted), 4)
        print(
            f"\nlocalisation: {len(localised)}/{len(admitted)} = {rate} "
            "of ADMITTED findings pointed at the line the dataset calls wrong"
        )
        print(
            "the complement is the number this repository could not previously "
            "report: evidence that exists exactly where the model said it "
            "does, attached to the wrong line."
        )
    if quoted_admitted:
        reuse = sum(r["reused_evidence"] for r in quoted_admitted)
        print(f"evidence reuse across admitted findings: {reuse}")

    if options.out:
        options.out.write_text(
            json.dumps(
                {
                    "dataset": str(options.dataset),
                    "renamed": options.rename,
                    "model": options.model,
                    "seed": options.seed,
                    "cases": len(results),
                    "admitted": len(admitted),
                    "localised": len(localised),
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
