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
    output_sha256: str
    output_bytes: int

    def receipt(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_digest": self.model_digest,
            "prompt_sha256": self.prompt_sha256,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
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
) -> Answer:
    """One deterministic completion.  Raises :class:`ReviewerUnavailable`.

    ``think`` is worth setting explicitly on reasoning models.  Measured on
    qwen3:8b with ``num_predict=768`` and thinking left on: the entire budget
    went to the reasoning trace and the answer came back empty.  An empty
    answer is a budget artefact, and reading it as a model failing to quote
    its evidence would be a measurement error.
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
    pinned = digest(model, endpoint=endpoint)
    reply = _post(endpoint, "/api/generate", payload, seconds)
    text = str(reply.get("response", ""))
    return Answer(
        text=text,
        model=model,
        model_digest=pinned,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        output_bytes=len(text.encode("utf-8")),
    )
