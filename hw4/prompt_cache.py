"""Mark a stable prompt prefix and simulate cache reuse without a provider."""

import hashlib
import json


TOKEN_ESTIMATE = "utf8_bytes_div_4"


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _estimated_tokens(encoded):
    return (len(encoded) + 3) // 4


def mark_system_cache(messages):
    if not messages or messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str):
        raise ValueError("Expected a leading system message with text content")
    system = {**messages[0], "content": [{
        "type": "text", "text": messages[0]["content"], "cache_control": {"type": "ephemeral"},
    }]}
    return [system, *messages[1:]]


def _prefix_bytes(messages, tools):
    return _canonical_bytes({"tools": tools, "system": messages[0]})


def prefix_sha256(prepared_messages, tools):
    return hashlib.sha256(_prefix_bytes(prepared_messages, tools)).hexdigest()


class SimulatedPromptCache:
    """Remember prefixes within a single ask run; counts are synthetic."""

    def __init__(self):
        self._prefixes = set()
        self._steps = []

    def prepare(self, messages, tools):
        prepared = mark_system_cache(messages)
        prefix = _prefix_bytes(prepared, tools)
        digest = hashlib.sha256(prefix).hexdigest()
        prefix_tokens = _estimated_tokens(prefix)
        input_tokens = prefix_tokens + _estimated_tokens(_canonical_bytes(prepared[1:]))
        cached_tokens = prefix_tokens if digest in self._prefixes else 0
        self._prefixes.add(digest)

        metrics = {
            "prefix_sha256": digest,
            "prefix_tokens": prefix_tokens,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "cached_input_fraction": cached_tokens / input_tokens,
        }
        self._steps.append({"step": len(self._steps) + 1, **metrics})
        return prepared, metrics

    def report(self):
        input_tokens = sum(step["input_tokens"] for step in self._steps)
        cached_tokens = sum(step["cached_tokens"] for step in self._steps)
        return {
            "mode": "simulated",
            "token_estimate": TOKEN_ESTIMATE,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "cached_input_fraction": cached_tokens / input_tokens if input_tokens else 0.0,
            "steps": list(self._steps),
        }
