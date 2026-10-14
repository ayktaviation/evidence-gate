"""Controls for the optional local reviewer, with no server and no model.

``ollama.py`` ships inside the package, so it is part of the published
surface whether or not the gate needs it.  It had no tests at all, which is
worse here than elsewhere: this module's whole job is to record *what was
asked of which model*, and a receipt that is wrong about that is more
dangerous than no receipt, because the numbers in the README are supposed to
be comparable on someone else's machine.

Nothing below starts a server or loads a model.  ``request.urlopen`` is
replaced, so these controls run on a machine with no Ollama installed and
cannot pass for the wrong reason -- a real server would make the digest
assertions depend on whatever happens to be pulled locally.

What is deliberately *not* tested: whether the model answers well.  That is
a measurement, it lives in ``measurements/``, and it cannot be an assertion.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import unittest
from urllib import error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from evidence_gate import ollama  # noqa: E402


class FakeReply(io.BytesIO):
    """Minimum of ``urlopen``'s contract that the module actually uses."""

    def __enter__(self) -> "FakeReply":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


@contextlib.contextmanager
def serving(*, tags=None, generate=None, raising=None):
    """Replace ``urlopen`` for the duration of a block.

    ``tags`` and ``generate`` are the decoded bodies the two endpoints
    return.  Calls are recorded so a test can assert what was *sent*, which
    is the half of this module that the receipt is a claim about.
    """
    sent: list[dict[str, object]] = []
    original = ollama.request.urlopen

    def fake(call, timeout=None):  # noqa: ANN001
        if raising is not None:
            raise raising
        url = call if isinstance(call, str) else call.full_url
        if url.endswith("/api/tags"):
            sent.append({"url": url, "timeout": timeout, "body": None})
            return FakeReply(json.dumps(tags or {}).encode("utf-8"))
        sent.append(
            {
                "url": url,
                "timeout": timeout,
                "body": json.loads(call.data.decode("utf-8")),
            }
        )
        return FakeReply(json.dumps(generate or {}).encode("utf-8"))

    ollama.request.urlopen = fake
    try:
        yield sent
    finally:
        ollama.request.urlopen = original


TAGS = {
    "models": [
        {"name": "qwen3:8b", "model": "qwen3:8b", "digest": "sha256:" + "ab" * 32},
        {"name": "other:latest", "model": "other:latest", "digest": "cd" * 32},
    ]
}


class AnAbsentServerIsAnAbsentServer(unittest.TestCase):
    """The one thing this module must never do: look like a model answer.

    A connection failure, a refused socket and a truncated body are three
    different events in the logs and the same event to a caller: no
    measurement happened.  They are converted to one exception type so that
    ``except ReviewerUnavailable`` is sufficient, and so an unmeasured arm
    cannot be silently recorded as a model that produced nothing.  The
    README's QuixBugs run has an arm recorded as unmeasured for exactly this
    reason, and that distinction depends on this conversion.
    """

    FAILURES = (
        error.URLError("connection refused"),
        OSError("socket gone"),
        ValueError("not json"),
        TimeoutError("timed out"),  # an OSError subclass; stated, not assumed
    )

    def test_ask_reports_unavailable_rather_than_an_empty_answer(self):
        for failure in self.FAILURES:
            with self.subTest(failure=type(failure).__name__):
                with serving(raising=failure):
                    with self.assertRaises(ollama.ReviewerUnavailable):
                        ollama.ask("prompt", model="qwen3:8b")

    def test_digest_reports_unavailable_rather_than_none(self):
        """``None`` already means "server answered, model not listed".

        If a dead server also returned ``None`` the two would be
        indistinguishable, and a run could be recorded with no digest
        because nothing was reachable.
        """
        for failure in self.FAILURES:
            with self.subTest(failure=type(failure).__name__):
                with serving(raising=failure):
                    with self.assertRaises(ollama.ReviewerUnavailable):
                        ollama.digest("qwen3:8b")

    def test_the_original_failure_is_not_swallowed(self):
        with serving(raising=error.URLError("connection refused")):
            with self.assertRaises(ollama.ReviewerUnavailable) as caught:
                ollama.digest("qwen3:8b")
        self.assertIn("connection refused", str(caught.exception))


class TheDigestIdentifiesTheModel(unittest.TestCase):
    """A name is not an identity.  ``latest`` moves; a local tag is rebuilt."""

    def test_the_sha256_prefix_is_stripped(self):
        with serving(tags=TAGS):
            self.assertEqual("ab" * 32, ollama.digest("qwen3:8b"))

    def test_a_bare_digest_is_accepted_too(self):
        """The server has shipped both forms; neither may crash the reader."""
        with serving(tags=TAGS):
            self.assertEqual("cd" * 32, ollama.digest("other:latest"))

    def test_either_key_matches(self):
        """``name`` and ``model`` have both been the populated one."""
        for key in ("name", "model"):
            with self.subTest(key=key):
                with serving(tags={"models": [{key: "solo:8b", "digest": "ef" * 32}]}):
                    self.assertEqual("ef" * 32, ollama.digest("solo:8b"))

    def test_an_unlisted_model_is_none_not_an_error(self):
        with serving(tags=TAGS):
            self.assertIsNone(ollama.digest("never-pulled:8b"))

    def test_an_empty_digest_is_none_rather_than_an_empty_string(self):
        """An empty string in a receipt reads as a digest; ``None`` does not."""
        with serving(tags={"models": [{"name": "blank:8b", "digest": ""}]}):
            self.assertIsNone(ollama.digest("blank:8b"))

    def test_a_missing_models_key_is_not_a_crash(self):
        with serving(tags={}):
            self.assertIsNone(ollama.digest("qwen3:8b"))

    def test_an_unpinned_run_still_answers_but_says_so(self):
        """Stated rather than defended.

        ``ask`` does not refuse when the server cannot name the model it is
        about to run.  The answer comes back with ``model_digest`` of
        ``None``, which is the honest record, and any measurement that
        depends on comparing models across machines must check that field.
        A reader of ``measurements/`` should know this is possible.
        """
        with serving(tags={"models": []}, generate={"response": "hello"}):
            answer = ollama.ask("prompt", model="never-pulled:8b")
        self.assertIsNone(answer.model_digest)
        self.assertEqual("hello", answer.text)


class TruncationIsRecordedNotInferred(unittest.TestCase):
    """An answer cut off by a budget is not a model declining to answer.

    Measured once and documented in ``ask``: on a reasoning model the whole
    token budget went to the thinking trace and the answer came back empty.
    Reading that as "the model could not quote its evidence" would have been
    a harness defect scored as a model defect, which is the one error this
    repository exists to avoid making.
    """

    def test_only_length_counts_as_truncated(self):
        for reason, expected in (
            ("length", True),
            ("stop", False),
            (None, False),
            ("load", False),
        ):
            with self.subTest(done_reason=reason):
                with serving(tags=TAGS, generate={"response": "x", "done_reason": reason}):
                    answer = ollama.ask("prompt", model="qwen3:8b")
                self.assertEqual(expected, answer.truncated)
                self.assertEqual(reason, answer.done_reason)

    def test_an_empty_answer_is_not_mistaken_for_truncation(self):
        """The two must stay separable, since they look identical by size."""
        with serving(tags=TAGS, generate={"response": "", "done_reason": "stop"}):
            answer = ollama.ask("prompt", model="qwen3:8b")
        self.assertEqual("", answer.text)
        self.assertEqual(0, answer.output_bytes)
        self.assertFalse(answer.truncated)

    def test_an_absent_done_reason_is_none_rather_than_empty(self):
        with serving(tags=TAGS, generate={"response": "x"}):
            self.assertIsNone(ollama.ask("prompt", model="qwen3:8b").done_reason)


class TwoExperimentsMayNotShareAReceipt(unittest.TestCase):
    """The mistake this hashing scheme was built after making.

    The absence-form instruction changed, results went from 1 of 3 to 3 of
    3, and ``prompt_sha256`` did not move -- so the two runs were, on paper,
    the same experiment with different outcomes.  Every input that can
    change an answer is hashed into some field, and the fields are separate
    so a reader can see *which* input moved.
    """

    def base(self):
        with serving(tags=TAGS, generate={"response": "answer"}):
            return ollama.ask("prompt", model="qwen3:8b", system="instructions")

    def test_changing_the_system_prompt_moves_only_its_own_hash(self):
        first = self.base()
        with serving(tags=TAGS, generate={"response": "answer"}):
            second = ollama.ask("prompt", model="qwen3:8b", system="different")
        self.assertEqual(first.prompt_sha256, second.prompt_sha256)
        self.assertNotEqual(first.system_sha256, second.system_sha256)
        self.assertEqual(first.options_sha256, second.options_sha256)

    def test_every_decoding_setting_moves_the_options_hash(self):
        """Seed, budget, thinking and schema are each an experiment change."""
        first = self.base()
        variants = {
            "seed": {"seed": 7},
            "num_predict": {"num_predict": 64},
            "think": {"think": False},
            "schema": {"schema": {"type": "object"}},
        }
        for label, kwargs in variants.items():
            with self.subTest(changed=label):
                with serving(tags=TAGS, generate={"response": "answer"}):
                    other = ollama.ask(
                        "prompt", model="qwen3:8b", system="instructions", **kwargs
                    )
                self.assertNotEqual(
                    first.options_sha256,
                    other.options_sha256,
                    f"{label} changes the experiment and must change the receipt",
                )
                self.assertEqual(first.prompt_sha256, other.prompt_sha256)
                self.assertEqual(first.system_sha256, other.system_sha256)

    def test_two_schemas_are_two_experiments(self):
        """A constrained run is not comparable to a differently constrained
        one, so the schema's *contents* must reach the hash, not merely the
        fact that a schema was supplied."""
        with serving(tags=TAGS, generate={"response": "answer"}):
            one = ollama.ask("p", model="qwen3:8b", schema={"required": ["a"]})
        with serving(tags=TAGS, generate={"response": "answer"}):
            two = ollama.ask("p", model="qwen3:8b", schema={"required": ["b"]})
        self.assertNotEqual(one.options_sha256, two.options_sha256)

    def test_the_same_request_gives_the_same_receipt(self):
        """The counter-case.  A hash that always differs identifies nothing."""
        self.assertEqual(self.base().options_sha256, self.base().options_sha256)
        self.assertEqual(self.base().prompt_sha256, self.base().prompt_sha256)

    def test_a_different_answer_moves_the_output_hash(self):
        with serving(tags=TAGS, generate={"response": "one"}):
            first = ollama.ask("prompt", model="qwen3:8b")
        with serving(tags=TAGS, generate={"response": "two"}):
            second = ollama.ask("prompt", model="qwen3:8b")
        self.assertNotEqual(first.output_sha256, second.output_sha256)
        self.assertEqual(first.prompt_sha256, second.prompt_sha256)

    def test_output_bytes_counts_bytes_not_characters(self):
        """Receipts are byte counts throughout; a mixed unit would make the
        published numbers unaddable."""
        with serving(tags=TAGS, generate={"response": "é"}):
            self.assertEqual(2, ollama.ask("p", model="qwen3:8b").output_bytes)


class TheRequestIsWhatTheReceiptClaims(unittest.TestCase):
    """A receipt describing a request that was not sent is a fabrication.

    These assert on the body that reached the fake server, because every
    hash above is otherwise a hash of whatever the function felt like
    hashing.
    """

    def test_determinism_is_actually_requested(self):
        with serving(tags=TAGS, generate={"response": "x"}) as sent:
            ollama.ask("prompt", model="qwen3:8b", seed=3, num_predict=11)
        body = [call for call in sent if call["body"] is not None][0]["body"]
        self.assertEqual(0, body["options"]["temperature"])
        self.assertEqual(3, body["options"]["seed"])
        self.assertEqual(11, body["options"]["num_predict"])
        self.assertFalse(body["stream"])
        self.assertEqual("prompt", body["prompt"])

    def test_unset_knobs_are_omitted_rather_than_sent_as_null(self):
        """``think=None`` means "server default", and sending ``null``
        instead would silently change the experiment on models where the
        default is on."""
        with serving(tags=TAGS, generate={"response": "x"}) as sent:
            ollama.ask("prompt", model="qwen3:8b")
        body = [call for call in sent if call["body"] is not None][0]["body"]
        self.assertNotIn("think", body)
        self.assertNotIn("format", body)

    def test_a_schema_is_sent_as_the_format_constraint(self):
        schema = {"type": "object", "required": ["findings"]}
        with serving(tags=TAGS, generate={"response": "x"}) as sent:
            ollama.ask("prompt", model="qwen3:8b", schema=schema)
        body = [call for call in sent if call["body"] is not None][0]["body"]
        self.assertEqual(schema, body["format"])

    def test_the_endpoint_is_respected_and_not_doubled(self):
        with serving(tags=TAGS, generate={"response": "x"}) as sent:
            ollama.ask("prompt", model="qwen3:8b", endpoint="http://127.0.0.1:9999/")
        for call in sent:
            self.assertNotIn("//api", call["url"].replace("http://", ""))
            self.assertTrue(call["url"].startswith("http://127.0.0.1:9999/api/"))

    def test_the_default_endpoint_is_loopback(self):
        """Stated as a control because this library may not reach a network.

        The measurement is meant to be reproducible on a machine with no
        internet access, and a default pointing anywhere else would make
        that false by accident.
        """
        self.assertTrue(ollama.DEFAULT_ENDPOINT.startswith("http://127.0.0.1"))


class TheReceiptCarriesNoContent(unittest.TestCase):
    """The rule the whole repository is built on, applied to this module.

    Diagnostics carry hashes, byte counts and key names -- never prompts,
    never model output.  A receipt is meant to be publishable next to a
    measurement, and one that quotes the prompt cannot be.
    """

    SECRETS = ("prompt-text", "system-text", "model-answer")

    def receipt(self):
        with serving(tags=TAGS, generate={"response": "model-answer"}):
            answer = ollama.ask(
                "prompt-text", model="qwen3:8b", system="system-text"
            )
        return answer.receipt()

    def test_no_field_contains_the_prompt_or_the_answer(self):
        serialised = json.dumps(self.receipt())
        for secret in self.SECRETS:
            with self.subTest(secret=secret):
                self.assertNotIn(secret, serialised)

    def test_the_receipt_is_json_serialisable(self):
        """It is written to ``measurements/`` files, so this is not academic."""
        json.dumps(self.receipt(), sort_keys=True)

    def test_the_fields_are_exactly_these(self):
        """Pinned, because a new field is how content leaks in later.

        If this fails, the question to answer is not "update the test" but
        "can the new field contain model output".
        """
        self.assertEqual(
            {
                "model",
                "model_digest",
                "prompt_sha256",
                "system_sha256",
                "options_sha256",
                "output_sha256",
                "output_bytes",
                "done_reason",
            },
            set(self.receipt()),
        )

    def test_the_answer_itself_still_carries_the_text(self):
        """The counter-case.  ``Answer.text`` is the caller's business; it is
        ``receipt()`` that must be safe to publish.  If the text were dropped
        everywhere the tests above would pass and the module would be
        useless."""
        with serving(tags=TAGS, generate={"response": "model-answer"}):
            answer = ollama.ask("p", model="qwen3:8b")
        self.assertEqual("model-answer", answer.text)

    def test_an_answer_is_immutable(self):
        """A receipt that can be edited after the fact is not a record."""
        with serving(tags=TAGS, generate={"response": "x"}):
            answer = ollama.ask("p", model="qwen3:8b")
        with self.assertRaises(Exception):
            answer.text = "something else"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
