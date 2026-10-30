"""Small OpenRouter client with explicit, fail-closed cost accounting."""

import math

import requests


class OpenRouterError(RuntimeError):
    """A sanitized API error suitable for storing in experiment traces."""


class BudgetExceeded(OpenRouterError):
    """The local spending threshold has been reached."""


class CostUnknownError(OpenRouterError):
    """A request may have been billed, but its cost is unknown."""


def _nonnegative_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


class OpenRouterClient:
    """Sequential client; the budget is checked BETWEEN requests.

    The last response may cross the threshold. Use an OpenRouter API key with a
    spending limit for a server-enforced ceiling. POSTs are never retried: after
    an ambiguous failure, further spending is blocked until costs are audited.
    ``spent_usd`` includes only verified charges, not estimates.
    """

    def __init__(self, api_key: str, budget_usd: float = 5.0, spent_usd: float = 0.0):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set OPENROUTER_API_KEY before running live experiments.")
        if not _nonnegative_number(budget_usd) or budget_usd == 0:
            raise ValueError("budget_usd must be a positive finite number")
        if not _nonnegative_number(spent_usd):
            raise ValueError("spent_usd must be a nonnegative finite number")
        self._api_key = api_key.strip()
        self.budget_usd = float(budget_usd)
        self.spent_usd = float(spent_usd)
        self.accounting_uncertain = False

    def _unknown_cost(self, message):
        self.accounting_uncertain = True
        raise CostUnknownError(message) from None

    def complete(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        if self.accounting_uncertain:
            raise CostUnknownError("Previous request cost is unknown; reconcile billing before continuing.")
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded("Local spending limit reached; no request was sent.")
        body = {"model": model, "messages": messages, "max_tokens": 512, "stream": False,
                "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body, timeout=(10, 90), allow_redirects=False,
            )
        except requests.RequestException:
            self._unknown_cost("OpenRouter network failure; request not retried because billing is uncertain.")
        if response.status_code != 200:
            # Rejected requests did not generate an answer. Server failures and
            # timeouts can occur after generation, so their charge is uncertain.
            if response.status_code in {400, 401, 402, 403, 404, 422, 429}:
                raise OpenRouterError(f"OpenRouter rejected request (HTTP {response.status_code}); no retry.")
            self._unknown_cost(f"OpenRouter HTTP {response.status_code}; billing is uncertain, no retry.")
        try:
            data = response.json()
        except ValueError:
            self._unknown_cost("OpenRouter returned invalid JSON; billing is uncertain.")
        usage = data.get("usage") if isinstance(data, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if not _nonnegative_number(cost):
            self._unknown_cost("OpenRouter response has no valid usage.cost; reconcile billing before continuing.")
        # Account even a malformed but paid response before validating content.
        self.spent_usd += float(cost)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                   for n in (prompt_tokens, completion_tokens)):
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            raise OpenRouterError("Malformed OpenRouter response; its reported cost has been recorded.") from None
        return {"message": message, "cost_usd": float(cost), "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "request_id": data.get("id", ""),
                "finish_reason": choice.get("finish_reason")}
