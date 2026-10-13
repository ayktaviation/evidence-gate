# evidence-gate

**When an LLM reviews your code and tells you what's wrong, check whether the
code it quoted actually exists.**

`evidence-gate` takes the findings a model produced and the text you showed it,
and refuses any finding whose quote is not in that text, byte for byte. It has
no model, no network and no dependencies. It decides *admissibility*, never
truth: it cannot tell you whether a finding is correct, only whether the model
was looking at your code when it made it.

```python
from evidence_gate import EvidenceSource, GateError, check

sources = [EvidenceSource(name="account.py", kind="source", text=open("account.py").read())]

try:
    receipt = check(findings, requirements=brief, sources=sources)
except GateError as refusal:
    print(refusal.reason)        # e.g. evidence_not_in_context
    print(refusal.diagnostic)    # hashes and byte counts, never the text
```

## The thing I actually found

I built this expecting to measure how often local models fabricate evidence.
That is not what dominates. **Most of what looks like fabrication is the gate
being unable to express the finding.**

Three local models, one 87-byte file, three requirements, identical prompt
(`sha256:43199ee4…`), temperature 0, seed 20261013, reasoning traces off.
Every run deterministic — repeated runs returned byte-identical output.

| gate version | qwen3:8b | qwen2.5-coder:14b | devstral:24b |
|---|---|---|---|
| **A.** quote must appear in source | admitted 3/3 — **vacuously** | refused | refused |
| **B.** + a quote may not be the whole file | refused | refused | refused |
| **C.** + a way to say "the required code is missing" | admitted 3/3 | admitted 3/3 | admitted 3/3 |

Read the rows in order, because each one is a different lesson.

**Row A is a false pass.** qwen3:8b satisfied a plain substring test by pasting
the entire file as its quote — three times, once per finding. Every character
matched. The quote localised nothing. A substring gate has a degenerate escape
at both ends: a three-character quote matches by accident, and a whole-file
quote matches by construction. Only the lower bound is obvious, so only the
lower bound tends to get written.

**Row B is the honest floor of a quote-only gate: 0 of 3.** But look at *why*
the two code models were refused. Of three findings each, the one about code
that was present carried a correctly located quote — devstral's was 25
characters, exactly the defective line. The two about code that was *missing*
carried an empty quote. The requirements were "every withdrawal must be
recorded" and "the caller must be told when a withdrawal is refused". Neither
is violated by any line you can point at. They are violated by absent lines,
and absent lines have no location.

So the models were not fabricating. They were being asked for a kind of
evidence that cannot exist, and they declined to invent it. The gate scored
that as their failure.

**Row C adds one concept** — an explicit, caller-enabled label a model may use
to say *there is nothing to quote* — and all three models become admissible.
Across them: 4 located quotes and 5 declared absences, zero fabrications.

An absence finding is a lead, not evidence, so the receipt counts the two
separately and never merges them:

```json
{ "finding_count": 3, "quoted_finding_count": 1, "unquoted_finding_count": 2 }
```

The uncomfortable conclusion: a verification gate reports the model's
limitations and its own in the same number, and has no way to tell you which
one you are reading. I only separated them here by making the gate stricter in
one direction and more expressive in another, and re-measuring.

One more trap worth naming. On the first pass, qwen3:8b returned zero bytes and
I nearly recorded it as "failed to produce evidence". It was a token-budget
artefact: the model is a reasoning model, the whole 768-token budget went to
the trace, and nothing was left for the answer. A measurement harness that
cannot tell an empty answer from a dishonest one will produce confident
nonsense.

## Try it in 60 seconds

```bash
git clone https://github.com/OWNER/evidence-gate && cd evidence-gate
python -m unittest discover -s tests          # 26 tests, no dependencies

# with a local Ollama running:
python examples/reproduce.py --model qwen3:8b
```

The example prints the model's receipt (digest, prompt hash, output hash) and
then either the refusal or the admission receipt. Use `--model` to run it
against anything you have. If you try a frontier model through your own
provider, the numbers are directly comparable to the table above — that is the
point of recording digests and seeds.

## How it decides

Comparison is exact. No normalisation of whitespace, case, quotes or Unicode
form. A model that cannot reproduce a span exactly has not read it, and every
tolerance you add is a hole someone's output will find. The suite includes
near-miss controls for collapsed whitespace, changed case, reflowed
indentation and a different Unicode normal form; all four are refused.

The gate stops at the first inadmissible finding rather than collecting every
problem, because one inadmissible quote already makes the answer inadmissible.

Every refusal is one of seven declared reasons, and raising an undeclared one
is a programming error:

| reason | meaning |
|---|---|
| `finding_not_object` | not a JSON object |
| `finding_key_set_mismatch` | wrong keys; diagnostic lists which |
| `finding_text_invalid` | a field is blank or not a string |
| `requirement_not_exact_quote` | reviewed against a paraphrased requirement |
| `evidence_too_short` | too short to identify anything |
| `evidence_too_long` | a quote is a pointer, not a copy |
| `evidence_covers_whole_source` | quoted (nearly) the entire file |
| `evidence_kind_not_allowed` | quoted something real, of the wrong kind |
| `evidence_not_in_context` | quoted something never shown |

`evidence_kind_not_allowed` is the one that is easy to miss. Each source
carries a caller-chosen `kind`, and `allowed_kinds` restricts what may be
quoted for a given question. Without it, a model asked about source code can
satisfy the gate by quoting an HTTP response it was also shown — a real quote,
a real refusal to answer the question. Collapsing that into
`evidence_not_in_context` hides which of the two is happening.

### Diagnostics never echo the rejected text

A rejected quote is frequently text the model invented. A log that repeats it
has turned itself into a store of unverified claims that later reads like
evidence. So `GateError.diagnostic` and the receipt carry reasons, indices,
SHA-256 digests and byte counts only. A test asserts this over every refusal in
the suite, and another asserts the admission receipt contains none of the
source, the requirements or the quotes.

The receipt does carry byte offsets of each match, which is what makes it
checkable: you can re-extract the span from your own text and confirm the gate
without trusting it.

## What this is not

- **Not novel.** Exact-span verification of model citations is deployed
  elsewhere — Anthropic's Citations API returns exact character spans, and
  groundedness checking is standard practice in retrieval systems. The
  mechanism here is a substring search. What this repository contributes is the
  measurement above, the negative controls, and the two refusals that a naive
  implementation omits.
- **Not a code reviewer.** It has no opinion about your code.
- **Not a correctness proof.** An admitted finding can be completely wrong.
  Admissible means the model was looking at your file.
- **Not a security boundary.** Hashes detect stale or careless evidence, not an
  adversary who can rewrite both the evidence and the code that checks it.
- **Not sufficient for absence.** The absence label records a claim the gate
  cannot verify at all. If you enable it, treat those findings as leads.

## Provenance

This was extracted from the review layer of a larger offline experiment: a
local, networkless agent pipeline whose entire design rests on separating what
a model *claims* from what the host *observed*. The gate exists because, in
that system, a reviewer model produced a 90-byte quote that appeared nowhere in
the serialised prompt. The pipeline has delivered zero applications across
eight recorded runs; every failure is preserved. The gate is the part that
turned out to be independently useful, and the measurement above is the part I
could not have guessed without running it.

Model digests in the table (`500a1f06…`, `9ec8897f…`, `9bd74193…`) are the
Ollama content digests, read from the server at call time rather than assumed
from the tag, because a tag moves and a digest does not.

## License

Apache-2.0.
