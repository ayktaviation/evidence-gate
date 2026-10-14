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

## 0.2.0

**Not breaking, additive:** the receipt gains
`ambiguous_localisation_count`, the number of admitted quotes that occur more
than once in the text they were checked against.

Third instance of the same principle as `distinct_evidence_count` and
`reused_evidence_count`: a quote appearing twice is present, exact and inside
every bound, so the gate may not refuse it -- picking which occurrence was
meant would be inventing a location, the failure this library exists to catch.
The per-finding `match_count` always carried the fact, so nothing was hidden;
it was absent from the summary, and the summary is what a caller reads. A
receipt announcing one located finding about a quote pointing at two places
reported less than it knew. Admissibility is unchanged and a control asserts
that, since a receipt field that quietly starts refusing things is a
tightening in disguise.

**Not breaking, additive:** `ollama.py` has controls for the first time, 29 of
them, and `check_claims.py` fails if the README's count of them drifts.

That module is importable from the installed package, so it was public surface
with no tests at all while the gate beside it had a control for every refusal.
The imbalance was not a judgment that it mattered less; it was never counted.
Its job is to record *what was asked of which model*, and a receipt that is
wrong about that is worse than no receipt, because the numbers in this README
are offered as reproducible on someone else's machine.

The controls pin the distinctions that a measurement here has already got
wrong once: an unreachable server raises rather than returning an empty answer
(an unmeasured arm must not become a model that produced nothing), a truncated
answer is recorded from `done_reason` rather than inferred from its size, every
decoding setting reaches `options_sha256` so two different experiments cannot
share a receipt, and `receipt()` carries no prompt and no output text.

They run against a replaced `urlopen` and open no socket, which is asserted
rather than assumed. That also bounds them: a field the real server renames
would pass the suite and fail on your machine, and the README says so where it
states the count.

**Not breaking, in the same release:** the committed measurement counts now
carry the SHA-256 of the run file each arm was computed from
(`evidence-gate-localisation-aggregate-v2`), and `significance.py --verify`
recomputes a committed aggregate from run files and exits non-zero on any gap.

Committing counts in 0.1.0 made the README's statistics checkable against the
counts and left the counts describing no particular input, so the two could be
edited together and still agree. A digest does not make a number true — whoever
writes both can write a consistent pair — but it names which bytes are being
claimed, so a holder of the runs can refute the pair, and it catches the one
failure counts alone cannot: a substituted run with identical totals. Both
failure modes are tested, and `check_claims.py` fails if a digest goes missing,
is malformed, or is shared by two arms.

The counts themselves were regenerated from the same run files and are
unchanged; only the schema name, the note and the digests differ.

**Breaking: adds a refusal reason**, `evidence_line_endings_differ`, which is a
minor version under the policy above. Code that exhaustively handled every
reason may now fall through, so the bump is required rather than courteous.

Found by probing the library rather than by reading it. A quote that matches
the source apart from its line endings was refused `evidence_not_in_context` --
the same reason as a quote of code that was never shown. A repository checked
out on Windows has CRLF endings and a model that reflows its answer emits LF,
so every multi-line quote came back as apparent fabrication, caused entirely by
the checkout.

That is this project's own central argument turned against it: the README
spends several paragraphs on why a gate reporting *format* when you asked about
*evidence* is measuring the harness, and this gate did exactly that on any CRLF
source. The suite had four near-miss controls -- collapsed whitespace, changed
case, reflowed indentation, a different Unicode normal form -- and none for
line endings, which is why it survived to be found by probing.

**Comparison is unchanged and nothing new is admitted.** The quote is still
refused; only the name of the refusal changed, so a caller can tell "normalise
your sources" from "distrust your model". Three controls guard the separation:
invented code on a CRLF source must still report
`evidence_not_in_context`, an exactly matching CRLF quote must still be
admitted, and the mismatch is tested in both directions.

**The first version of this fix had the precedence wrong**, found by probing
the fix rather than by reading it. The new branch sat inside the "no match
anywhere" arm, so on a CRLF source a whole-file LF quote never reached
`evidence_covers_whole_source` and was reported as a line-ending mismatch --
the mild answer that sends a caller to normalise its sources, when the real
finding was the vacuous whole-file quote that produced this library's first
measured bound. Nothing was admitted either way, which is not a defence: the
name is the product. Vacuity now outranks the platform artefact, asserted in
both line-ending directions, and the fixtures that provoked the wrong reason
were padded rather than the bound relaxed.

Unicode normal form remains a refusal and is deliberately *not* given the same
treatment. A different normal form is a different string with no single
obviously-intended reading, whereas a line ending is a platform convention for
the same text.

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
- The per-arm counts from that run are committed as
  `measurements/quixbugs-2026-10-13.json` and the README's p-values are
  recomputed from them by `tools/check_claims.py`. The run files themselves are
  not published: they are content-free but name the holdout cases. Counts
  suffice to re-run every test and do not suffice to ask a new question, so
  re-analysis still requires the runs.
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
