"""Sequential OpenRouter transport for the five HW2 configurations."""
import json
import time
from threading import Lock

import requests

from freshqa.client import BudgetExceeded, OpenRouterClient, OpenRouterError


# USD per million prompt/completion tokens, constrained on the route itself.
# https://openrouter.ai/docs/guides/routing/provider-selection#max-price
PRICE_CAPS = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-haiku-4.5": (1, 5),
    "anthropic/claude-sonnet-4.6": (3, 15),
}
MIN_INTERVAL = 2.5


class BenchmarkClient(OpenRouterClient):
    """Reserve headroom before each text request; account only the provider's cost."""

    def __init__(self, api_key, budget_usd, spent_usd=0):
        super().__init__(api_key, budget_usd, spent_usd)
        self.session = requests.Session()
        self.session.trust_env = False
        self._last_start = None
        self._lock = Lock()

    def close(self):
        self.session.close()

    def complete(self, *, model, messages, tools=None):
        # Hold the lock until actual usage has been booked by the parent.
        with self._lock:
            result = super().complete(model=model, messages=messages, tools=tools)
            if not isinstance(result["request_id"], str) or not result["request_id"].strip():
                raise OpenRouterError("The provider reported a cost without a request ID.")
            return result

    def _post(self, url, **kwargs):
        body = kwargs["json"]
        if body["model"] not in PRICE_CAPS:
            raise ValueError("No approved price cap for this model.")
        messages = body["messages"]
        for message in messages:
            if (not isinstance(message.get("content"), (str, type(None)))
                    or any(message.get(key) for key in ("audio", "images", "attachments"))):
                raise ValueError("The benchmark budget supports text-only messages.")
        if any(tool.get("type") != "function" for tool in body.get("tools", [])):
            raise ValueError("The benchmark supports only local function tools.")

        prompt_price, completion_price = PRICE_CAPS[body["model"]]
        serialized = json.dumps(body, ensure_ascii=True, allow_nan=False)
        # Estimate one token per escaped JSON byte, plus provider templates.
        # The 2x input rate also covers Anthropic's longest cache-write premium.
        prompt_allowance = len(serialized) + 4096 + 64 * len(messages)
        reserve = (2 * prompt_price * prompt_allowance + completion_price * body["max_tokens"]) / 1_000_000
        if self.spent_usd + reserve > self.budget_usd:
            raise BudgetExceeded("Insufficient remaining budget to admit this request; no request was sent.")
        body["provider"] = {"max_price": {"prompt": prompt_price, "completion": completion_price, "request": 0}}
        if self._last_start is not None:
            delay = self._last_start + MIN_INTERVAL - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._last_start = time.monotonic()
        return self.session.post(url, **kwargs)
