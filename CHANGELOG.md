# Changelog

## Versioning policy

`REASONS` is part of the public interface, not an implementation detail. A
caller is expected to branch on those names, so:

- **Adding a refusal reason is a breaking change** and takes a minor version
  before 1.0, a major version after. Code that handled every reason yesterday
  may fall through today, and silently admitting what it meant to inspect is
  the worst outcome this library can produce.
- **Removing or renaming a reason** is breaking for the same reason.
- **Tightening a default bound** is breaking: input that was admitted becomes
  refused. Both bound changes so far were tightenings, and both are recorded
  below with the measurement that caused them.
- `RECEIPT_SCHEMA` carries a version string. Adding a receipt field is not
  breaking; changing or removing one bumps the schema name.
- Loosening a default bound will not happen quietly. Every widening in this
  library's history was paid for by counting the widened case separately in the
  receipt, so no reader can mistake it for a verified quote.

## 0.1.0

First release. Exact-substring verification of model-supplied evidence, with
twelve declared refusals, content-free diagnostics, and verifiable absence.

Both bounds on absence, and the ceiling on quotes, exist because measurement
forced them rather than because they were designed:

- `evidence_covers_whole_source` — a local model satisfied a plain substring
  test by pasting an entire 88-byte file as its quote, three times. Every
  character matched and the quote localised nothing.
- `absence_pattern_too_broad` — with absence made checkable, two local models
  named 70- and 57-character invented statements as the missing text. Such a
  line is absent from an 88-byte file whatever the code does. The same vacuous
  pass as the whole-file quote, reached from the opposite direction.
- `max_absence_pattern_chars` — the fraction above, measured on real modules,
  permitted a 2,504-character absence pattern. A bound fitted to a toy file
  dissolves at scale, so an absolute cap was added alongside it. The two cross
  at 80 characters; above that the cap binds alone.

The absence floor (`min_absence_pattern_chars`, 3) is deliberately lower than
the quote floor (`min_evidence_chars`, 8). A short quote matches by accident; a
short absence pattern is harder to be absent, so its absence is the stronger
claim. `record(` is seven characters: inadmissible as a quote, exactly right as
a search pattern.

`distinct_evidence_count` and `reused_evidence_count` are receipt fields, not
refusals, and adding them is therefore not breaking. They exist because of the
one weakness no bound in this library can close: a real, exactly located quote
reused across unrelated requirements satisfies every check here. Judging
whether a quote is *about* its requirement needs a reader or a model, and this
library is neither, so the gate counts what it is not entitled to refuse. The
same hole exists on the absence side and is worse, because there it cannot even
be counted — `absent:zqx` is three characters, genuinely missing, and
worthless. `tests/test_compliant_parasite.py` asserts all of this passes, and
`tools/check_claims.py` fails the build if the README stops saying so.

Known limits, stated because they bound what the numbers in the README mean:

- The published measurement is nine data points — three local models, three
  requirements, one 88-byte file. Thresholds were chosen against that file, and
  one of them turned out to be meaningless at the scale of a real module.
- At 28 KB of real source, with decoding constrained to the required schema so
  that format failure is engineered away rather than counted, two of three
  models attached quotes that appear nowhere in what they were shown. The
  optimistic finding above does not survive leaving the toy. The third model
  looped and is recorded as unmeasured.
- That scale run reviewed this library with this library, so the model could
  imitate the receipt schema it was reading — and one of the three failures
  looks like exactly that. The experiment needs an unrelated subject.
- Against a third party's ground truth (QuixBugs, 36 single-line Python
  defects, three local models, paired on 35 cases), the gate admits 94–100% of
  findings while 11–49% of admitted findings name the right line. Admissibility
  is not correctness, and nothing in this library claims to measure the latter.
- In that run, all three models fall to chance when identifiers are rewritten
  and are indistinguishable from quoting a line at random. Reformatting through
  the same code path changes nothing (two models return identical counts), so
  the drop tracks the names, not the layout. Read as failure to reject chance,
  not as proof of guessing: n = 35 and one seed cannot support more. The
  published arm is an upper bound, since the dataset is public and
  contamination is the expected condition.
- No frontier model has been measured, so the central finding (that apparent
  fabrication was mostly the gate failing to express the finding) is not
  separated from the limits of 8–24B models.
- A model may answer a "missing code" requirement with a real, exactly located
  quote of the line that fails to do the thing. That is admissible and the gate
  has no opinion on whether it is a fair answer.
