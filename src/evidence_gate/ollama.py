"""Optional local reviewer used by the examples.  Standard library only.

The gate itself needs no model.  This module exists so the measurement in the
README can be reproduced on another machine against another model, and so the
numbers can be compared rather than believed.

It pins what it can: the model digest is read back from the server and
recorded, temperature is zero and the seed is fixed, so a run is identified by
something stronger than a model name.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from urllib import error, request

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    model_digest: str | None
    prompt_sha256: str
    system_sha256: str
    options_sha256: str
    output_sha256: str
    output_bytes: int
    #: Why the server stopped.  ``"length"`` means the answer was cut off by
    #: the token budget, which looks exactly like a model that cannot produce
    #: the requested shape and is not the same event at all.  Recorded rather
    #: than inferred, because inferring it from output size is guesswork.
    done_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"

    def receipt(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_digest": self.model_digest,
            "prompt_sha256": self.prompt_sha256,
            # The system prompt is hashed separately because changing the
            # instructions while leaving the question alone is the most common
            # thing a measurement does, and it must not produce a receipt
            # identical to the previous run's.  Observed exactly that: the
            # absence-form instruction changed, results went from 1 of 3 to
            # 3 of 3, and prompt_sha256 did not move.
            "system_sha256": self.system_sha256,
            # Likewise temperature, seed and token budget: an empty answer
            # caused by a budget is not the same event as a model declining.
            "options_sha256": self.options_sha256,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "done_reason": self.done_reason,
        }


class ReviewerUnavailable(RuntimeError):
    pass


def _post(endpoint: str, path: str, payload: dict, seconds: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    call = request.Request(
        endpoint.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(call, timeout=seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.URLError, OSError, ValueError) as failure:
        raise ReviewerUnavailable(str(failure)) from None


def digest(model: str, *, endpoint: str = DEFAULT_ENDPOINT, seconds: float = 30) -> str | None:
    """Return the server's content digest for ``model``, or ``None``.

    A model name is not an identity: ``latest`` moves and a local tag can be
    rebuilt.  The digest is what makes a recorded run comparable to another
    machine's run, so it is read from the server rather than assumed.
    """
    try:
        with request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=seconds) as reply:
            listed = json.loads(reply.read().decode("utf-8"))
    except (error.URLError, OSError, ValueError) as failure:
        raise ReviewerUnavailable(str(failure)) from None
    for entry in listed.get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            found = str(entry.get("digest") or "")
            return found.split(":")[-1] or None
    return None


def ask(
    prompt: str,
    *,
    model: str,
    system: str = "",
    endpoint: str = DEFAULT_ENDPOINT,
    seed: int = 0,
    seconds: float = 300,
    num_predict: int = 768,
    think: bool | None = None,
    schema: dict | None = None,
) -> Answer:
    """One deterministic completion.  Raises :class:`ReviewerUnavailable`.

    ``think`` is worth setting explicitly on reasoning models.  Measured on
    qwen3:8b with ``num_predict=768`` and thinking left on: the entire budget
    went to the reasoning trace and the answer came back empty.  An empty
    answer is a budget artefact, and reading it as a model failing to quote
    its evidence would be a measurement error.

    ``schema`` constrains decoding to a JSON schema.  It exists because of a
    measurement, not for convenience: against 26 KB of real source, two of
    three local models answered in prose and a third invented its own keys, so
    the gate never reached a single quote.  Constraining the shape removes that
    failure mode and removes *only* that one -- the first constrained answer
    observed still carried a fabricated quote.  That is the point.  A harness
    defect and a fabrication must not be able to produce the same number, and
    fixing the shape is how you stop crediting the gate for catching bad
    formatting.
    """
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
    }
    if think is not None:
        payload["think"] = think
    if schema is not None:
        payload["format"] = schema
    pinned = digest(model, endpoint=endpoint)
    reply = _post(endpoint, "/api/generate", payload, seconds)
    text = str(reply.get("response", ""))
    # The schema belongs in this hash.  A constrained run and an unconstrained
    # run of the same question are different experiments, and two experiments
    # that share a receipt cannot be compared -- the mistake already made once
    # with the system prompt.
    settings = json.dumps(
        {
            **payload["options"],  # type: ignore[dict-item]
            "think": payload.get("think"),
            "format": payload.get("format"),
        },
        sort_keys=True,
    )
    return Answer(
        text=text,
        model=model,
        model_digest=pinned,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        system_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest(),
        options_sha256=hashlib.sha256(settings.encode("utf-8")).hexdigest(),
        output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        output_bytes=len(text.encode("utf-8")),
        done_reason=str(reply.get("done_reason") or "") or None,
    )
