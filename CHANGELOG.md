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

Known limits, stated because they bound what the numbers in the README mean:

- The published measurement is nine data points — three local models, three
  requirements, one 88-byte file. Thresholds were chosen against that file, and
  one of them turned out to be meaningless at the scale of a real module.
- At 26 KB of real source, none of the three models produced an admissible
  review, and no failure was about evidence: two wrote prose instead of JSON
  and one returned the wrong key set. Whether a quote can be verified is simply
  not the binding constraint at that size. The library is not shown to be
  useful on a real codebase, and the one attempt says it is not yet.
- That scale run reviewed this library with this library, so the model could
  imitate the receipt schema it was reading — and one of the three failures
  looks like exactly that. The experiment needs an unrelated subject.
- No frontier model has been measured, so the central finding (that apparent
  fabrication was mostly the gate failing to express the finding) is not
  separated from the limits of 8–24B models.
- A model may answer a "missing code" requirement with a real, exactly located
  quote of the line that fails to do the thing. That is admissible and the gate
  has no opinion on whether it is a fair answer.
