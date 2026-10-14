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
On a small file that is not what dominates: **most of what looked like
fabrication was the gate being unable to express the finding.**

Then I ran it on real source and the conclusion reversed. Across two unrelated
subjects — this library, and TypeScript from a codebase I did not write — every
model produced at least one finding whose evidence does not survive checking.
All of it is below in the order I found it, because the gap between the two is
the only part worth your time: **the toy measurement I trusted was the one that
lied, and it lied in the flattering direction.**

Three local models, one 88-byte file, three requirements, identical question
(`prompt sha256:43199ee4…`), temperature 0, seed 20261013, reasoning traces
off. Every run deterministic — repeated runs returned byte-identical output.

| gate version | qwen3:8b | qwen2.5-coder:14b | devstral:24b |
|---|---|---|---|
| **A.** quote must appear in source | admitted 3/3 — **vacuously** | refused | refused |
| **B.** + a quote may not be the whole file | refused | refused | refused |
| **C.** + a blind label for "the required code is missing" | admitted 3/3 — **partly vacuously** | admitted 3/3 | admitted 3/3 |
| **D.** + absence must name the missing text, and the gate searches for it | admitted 3/3 | refused | refused |
| **E.** + the prompt states the pattern must be short | admitted 3/3 | admitted 3/3 | admitted 3/3 |

Read the rows in order, because each one is a different lesson, and two of the
five are lessons about the gate deceiving *me*.

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

That looked like the answer, and it was the second false pass. The label is a
claim the gate accepts without checking anything. I had replaced a quote I
verify with a sentence I believe.

**Row D makes absence checkable.** Instead of asserting that something is
missing, the model names the literal text that would have to be there —
`record(`, `raise `, `log(` — and the gate searches the file for it. Zero
occurrences is an observation the host made. One occurrence refutes the model
outright. This is the only claim in the whole system the gate can settle on its
own, and it is settled against the same bytes as a quote.

Two of three models were then refused — and this is the part worth the detour.
Their patterns were 70 and 57 characters against an 88-byte file: whole
invented statements like `if amount > account.balance: raise ValueError(...)`.
Such a line is absent from that file whether or not the requirement is met.
**That is Row A's vacuous pass arriving from the opposite direction.** A
substring test has a degenerate escape at each end, and so does an absence
test, with the ends swapped:

| | too short | too long |
|---|---|---|
| **quote** | matches by accident | matches by construction |
| **absence pattern** | absent from almost nothing | absent by construction |

I had written the floor for quotes and the floor for patterns, and the ceiling
for quotes. The fourth cell was missing, which is why Row C and the first
measurement of Row D both read better than they were. Note the floors differ on
purpose: `record(` is seven characters, inadmissible as a quote and exactly
right as a search pattern.

**Row E changes the prompt, not the gate** — it tells the model to keep the
pattern under twenty characters — and all three pass with patterns of 4, 12, 13
and 28 characters. Real searches, really performed.

An absence finding is still a lead, not evidence, so the receipt counts the
three kinds separately and never merges them:

```json
{ "finding_count": 3, "quoted_finding_count": 1,
  "verified_absence_count": 2, "unverified_absence_count": 0 }
```

The uncomfortable conclusion: a verification gate reports the model's
limitations and its own in the same number, and has no way to tell you which
one you are reading. Every separation above came from making the gate stricter
in one direction, more expressive in another, and re-measuring — never from
reasoning about it. Twice the number moved the wrong way and I had believed it.

One hole stays open, and the gate cannot close it. In Row E, qwen3:8b never used
the absence form at all: it answered all three requirements with real quotes,
including the two about missing code, by pointing at the line that *fails* to
record. Those quotes are exactly located and perfectly admissible. Whether they
are a fair answer to the requirement is a judgement about meaning, and this
gate makes no judgements about meaning.

One more trap worth naming. On the first pass, qwen3:8b returned zero bytes and
I nearly recorded it as "failed to produce evidence". It was a token-budget
artefact: the model is a reasoning model, the whole 768-token budget went to
the trace, and nothing was left for the answer. A measurement harness that
cannot tell an empty answer from a dishonest one will produce confident
nonsense.

## What happened at realistic scale

Everything above is nine data points against an 88-byte file. Thresholds fitted
to a toy are suspect, so I ran the same procedure against two real modules —
`gate.py` and `ollama.py`, four requirements about this code's own stated rules
(`examples/at_scale.py`).

Two numbers, because the subject is this library and the library kept changing
while I measured it: 26,538 characters for the free-form run below, 28,156 for
the constrained one. A subject that drifts between runs is a poor subject, which
is why the third measurement uses an unrelated codebase.

**Asking for JSON in the prompt, no model produced an admissible review, and
not one of the failures was about evidence:**

| model | output | outcome |
|---|---|---|
| devstral:24b | 3,647 bytes | prose essay, no `findings` key — format failure |
| qwen2.5-coder:14b | 2,950 bytes | prose, no `findings` key — format failure |
| qwen3:8b | 1,750 bytes | JSON, refused `finding_key_set_mismatch` |

A gate that reports *format* when you asked it about *evidence* is measuring
the harness. So I removed the harness defect instead of recording it as a
finding: Ollama can constrain decoding to a JSON schema, and the requirement
field becomes an enum of the exact requirement strings, which makes a
paraphrased requirement unrepresentable rather than punished after the fact.
Nothing about `evidence` is constrained — that is the claim under test, and
constraining it would rig the experiment.

**With the shape no longer in question, the result inverts:**

| model | done_reason | outcome |
|---|---|---|
| devstral:24b | `stop` | **refused `evidence_not_in_context`** — 57-byte quote |
| qwen3:8b | `stop` | **refused `evidence_not_in_context`** — 54-byte quote |
| qwen2.5-coder:14b | `length` | truncated at 18,301 bytes; not measurable |

Both models that answered cleanly attached a quote that appears **nowhere in
the 28,156 characters they were shown.** Not a near-miss, not a whitespace
difference, not an inexpressible absence — invented code attributed to a file.

**This contradicts the headline of this README on anything larger than a toy.**
At 88 bytes, apparent fabrication was mostly the gate failing to express a
finding. At 26 KB, with every expressiveness excuse removed and the format
failure engineered away, fabrication is what is left. The optimistic reading
was an artefact of a subject small enough to hold in one glance.

So the gate is worth having, and I could not have learned that from the
measurement I was most proud of.

The third model is recorded as **unmeasured, not as a failure.** `done_reason`
is read back from the server rather than guessed from output size: at a 4,096
token budget it was cut off mid-answer, and at 10,000 it ran past the 300-second
timeout, which means it degenerates into a loop. Truncated JSON under
constrained decoding looks exactly like a model that cannot comply, and
counting it as fabrication would be the same error as the qwen3 empty answer
further down.

**One of those three failures has a confound worth naming.** qwen3's rejected
keys were `index`, `match_count` and `matches` — fields of the receipt schema
defined in the very file it was reviewing. It imitated the data structure it was
reading instead of answering about it. Reviewing the gate with the gate is
convenient and reproducible, and it contaminates the output format, which is
what the next section removes.

### Without the confound

Both runs above review this library with this library, which lets a model
imitate the receipt schema it is reading and lets the subject drift as the
library changes. So I repeated it on an unrelated subject: two TypeScript
modules from a private aviation-inspection codebase, 14,025 characters, four
requirements taken from the code's own stated disclaimers. Different language,
different domain, nothing for a reviewer to mimic, and a subject that does not
move.

The subject is not in this repository and will not be. `--subject` takes paths
from outside it and the script prints only names, byte counts and digests —
the same discipline the gate applies to evidence, so a result stays checkable
without republishing someone else's code.

| model | bytes | done_reason | refusal |
|---|---|---|---|
| devstral:24b | 7,779 + 6,275 | `stop` | **`absence_pattern_present`** — claimed text is missing; it is in the file |
| qwen3:8b | same | `stop` | **`evidence_not_in_context`** — 206-byte quote, shown nowhere |
| qwen2.5-coder:14b | same | `stop` | **`absence_pattern_too_broad`** — invented statement as the missing text |

No format failures, no truncation, three clean answers, and **three different
evidence refusals.** Read precisely: the gate stops at the first inadmissible
finding, so this says each model produced *at least one* finding that does not
hold, not that everything each said was wrong.

devstral's is the interesting one. `absence_pattern_present` is the only
refusal that *contradicts* rather than declines: the model named text it said
was absent, the gate searched, and the text was there — one occurrence, located.
That is a model claim settled against the file, which is the strongest thing
this library can do and the first time it has done it on code I did not write.

So the reversal holds with the confound removed. On a toy the gate was mostly
reporting its own inexpressiveness; on real code, at two independent subjects
and three models, the evidence does not survive checking.

### The threshold that dissolved

The scale run also killed a bound I had just written. `max_absence_source_fraction`
permits half the smallest source: 44 characters against the toy, but **2,504
characters** against these modules. The bound that caught the vacuous absence
claim on an 88-byte file catches nothing whatsoever on real code, because an
invented statement is absent by construction at any size and a fraction of a
large file is enormous.

So absence now carries an absolute cap as well, and the binding bound is
whichever is smaller. They cross at 80 characters: above that — all real code —
the absolute cap works alone, and the fraction survives only to stand between a
39-character pattern and a 40-character file. Written out because a bound whose
scope you cannot state is a bound you are trusting, not using.

## Try it in 60 seconds

```bash
cd evidence-gate
python -m unittest discover -s tests          # 41 tests, no dependencies, no install
python tools/check_claims.py                  # the README's own numbers, verified

# with a local Ollama running:
python examples/reproduce.py --model qwen3:8b   # the 88-byte measurement
python examples/at_scale.py  --model qwen3:8b   # the 26 KB one, which fails
```

The example prints the model's receipt (model digest, and separate hashes for
the question, the instructions, the sampling options and the output — changing
the instructions while leaving the question alone must not produce the previous
run's receipt, which is a mistake I made and only caught because the numbers
moved while the hash did not) and
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

Every refusal is one of twelve declared reasons, and raising an undeclared one
is a programming error. A test asserts the suite contains a control for every
one, so adding a way to refuse without a control for it fails the build:

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
| `absence_pattern_too_short` | absent from almost any text |
| `absence_pattern_too_broad` | absent by construction, so absence proves nothing |
| `absence_pattern_present` | claimed text is missing; it is in the file |

`absence_pattern_present` is the only refusal that contradicts the model rather
than declining to accept it. If both it and `absence_pattern_too_broad` apply,
the gate reports the refutation: *that text is in your file* says something
true about the code, while *your pattern was too broad* only grades the
question.

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
- **Not sufficient for absence.** A verified absence says one literal string
  does not occur in the text you supplied. It does not say the requirement is
  unmet: the same behaviour can be written a hundred ways, and the model named
  one. Treat these as leads with a receipt, not findings. The unverified label
  (`absence_label`) is weaker still — it records a claim the gate checks
  nothing about — and the receipt counts the two apart so you can never read
  one as the other.

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
