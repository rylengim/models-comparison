"""Groq diagnostics with separate list-price estimates, never claimed bills."""

import math

import requests

from .client import BudgetExceeded, CostUnknownError, OpenRouterError, _nonnegative_number


PRICING_SOURCE = "https://console.groq.com/docs/models"
PRICING_VERIFIED_AT = "2026-10-29"
PRICING = {
    "openai/gpt-oss-20b": {"input_per_million_usd": .075, "output_per_million_usd": .30},
    "openai/gpt-oss-120b": {"input_per_million_usd": .15, "output_per_million_usd": .60},
}
REQUEST_SETTINGS = {
    "max_completion_tokens": 1024, "stream": False, "temperature": 0,
    "reasoning_effort": "low", "include_reasoning": False, "parallel_tool_calls": False,
}


class GroqError(OpenRouterError):
    """Sanitized provider error accepted by the existing agent error boundary."""


class GroqClient:
    """A diagnostic client whose budget guard counts estimated usage.

    Groq does not report a billed amount in chat-completion usage. ``spent_usd``
    therefore remains zero (the known billed portion), while ``cost_usd=None``
    marks the actual charge as unreported. ``budget_used_usd`` is a list-price
    estimate without cache discounts, credits or account-specific adjustments.
    It is checked BETWEEN requests; the last response can cross the threshold.
    Ambiguous failures block further requests. POSTs are never retried.
    """

    provider = "groq"
    cost_basis = "list_price_estimate"

    def __init__(self, api_key: str, budget_usd: float = .1, budget_used_usd: float = 0.0):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set GROQ_API_KEY before running Groq diagnostics.")
        if not _nonnegative_number(budget_usd) or budget_usd == 0:
            raise ValueError("budget_usd must be a positive finite number")
        if not _nonnegative_number(budget_used_usd):
            raise ValueError("budget_used_usd must be a nonnegative finite number")
        self._api_key = api_key.strip()
        self.budget_usd = float(budget_usd)
        self.budget_used_usd = float(budget_used_usd)
        self.spent_usd = 0.0
        self.accounting_uncertain = False

    def _unknown_usage(self, message):
        self.accounting_uncertain = True
        raise CostUnknownError(message) from None

    def complete(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        if self.accounting_uncertain:
            raise CostUnknownError("Previous Groq request usage is unknown; reconcile before continuing.")
        if self.budget_used_usd >= self.budget_usd:
            raise BudgetExceeded("Groq estimated usage limit reached; no request was sent.")
        if model not in PRICING:
            raise ValueError("Unsupported Groq model; use openai/gpt-oss-20b or openai/gpt-oss-120b.")
        body = {"model": model, "messages": messages, **REQUEST_SETTINGS}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body, timeout=(10, 90), allow_redirects=False,
            )
        except requests.RequestException:
            self._unknown_usage("Groq network failure; request not retried because usage is uncertain.")
        if response.status_code != 200:
            if response.status_code in {401, 402, 403, 404, 422, 429}:
                raise GroqError(f"Groq rejected request (HTTP {response.status_code}); no retry.")
            # A 400 can follow generation of invalid tool calls. Do not infer a
            # zero charge, or expose failed_generation / the raw response body.
            self._unknown_usage(f"Groq HTTP {response.status_code}; usage is uncertain, no retry.")
        try:
            data = response.json()
        except ValueError:
            self._unknown_usage("Groq returned invalid JSON; usage is uncertain.")
        if isinstance(data, dict) and data.get("model") != model:
            self._unknown_usage("Groq returned an unexpected model; usage pricing is uncertain.")
        usage = data.get("usage") if isinstance(data, dict) else None
        try:
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                   for n in (prompt_tokens, completion_tokens)):
                raise ValueError
            prices = PRICING[model]
            estimate = (prompt_tokens / 1_000_000 * prices["input_per_million_usd"]
                        + completion_tokens / 1_000_000 * prices["output_per_million_usd"])
            if not math.isfinite(estimate):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            self._unknown_usage("Groq response has no valid token usage; reconcile before continuing.")
        # Retain observed usage even when the response content is malformed.
        self.budget_used_usd += estimate
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            raise GroqError("Malformed Groq response; its usage estimate has been recorded.") from None
        return {
            "message": message, "cost_usd": None, "estimated_cost_usd": estimate,
            "cost_basis": self.cost_basis, "provider": self.provider,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "request_id": data.get("id", ""), "finish_reason": choice.get("finish_reason"),
            "requested_model": model, "response_model": data["model"], "usage": usage,
        }
