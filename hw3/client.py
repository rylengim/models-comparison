"""One HTTP client and a durable response-hook ledger for every model call."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from freshqa.client import BudgetExceeded, CostUnknownError, OpenRouterError


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class MeteredClient:
    provider = "openrouter"
    cost_basis = "provider_reported"

    def __init__(self, api_key, ledger, budget_usd=2.0, *, transport=None, interval=1.0,
                 rates=None, max_tokens=512):
        if not api_key or not valid_number(budget_usd) or budget_usd == 0:
            raise ValueError("A local API key and positive budget are required")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.inflight = self.ledger.with_suffix(".inflight.json")
        self.records = [json.loads(s) for s in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        self.spent_usd = sum(r["cost_usd"] for r in self.records if r.get("cost_usd") is not None)
        self.accounting_uncertain = self.inflight.exists() or any(r.get("cost_usd") is None for r in self.records)
        self.budget_usd, self.interval = float(budget_usd), interval
        self.rates, self.max_tokens = rates or {}, max_tokens
        self.config, self.question_id, self.max_calls, self.scope_calls = "unscoped", "", 24, 0
        self._last_started = 0.0
        self._key = api_key
        self.http = httpx.Client(timeout=90, trust_env=False, follow_redirects=False,
                                 transport=transport, event_hooks={"response": [self.on_response]})

    @property
    def calls_remaining(self):
        return max(0, self.max_calls - self.scope_calls)

    def set_scope(self, config, question_id, max_calls=24):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.config, self.question_id = config, question_id
        self.max_calls = max_calls
        self.scope_calls = sum(r.get("config") == config and r.get("question_id") == question_id for r in self.records)

    def _append(self, record):
        self.records.append(record)
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def on_response(self, response):
        """The hook sees raw request/response data, never authorization headers."""
        response.read()
        body = json.loads(response.request.content)
        try:
            data = response.json()
        except ValueError:
            data = {}
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        cost = usage.get("cost")
        rejected = response.status_code in {400, 401, 402, 403, 404, 422, 429}
        if not valid_number(cost):
            cost = 0.0 if rejected else None
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "config": self.config,
                  "question_id": self.question_id, "model": body["model"],
                  "status_code": response.status_code, "cost_usd": cost,
                  "prompt_tokens": usage.get("prompt_tokens", 0 if rejected else None),
                  "completion_tokens": usage.get("completion_tokens", 0 if rejected else None),
                  "request_id": data.get("id") if isinstance(data, dict) else None,
                  "request": body, "request_bytes": response.request.content.decode(),
                  "response_message": choice.get("message")}
        self._append(record)
        self.inflight.unlink(missing_ok=True)
        if cost is None:
            self.accounting_uncertain = True
        else:
            self.spent_usd += cost

    def complete(self, *, model, messages, tools=None, temperature=0, response_format=None):
        if self.accounting_uncertain:
            raise CostUnknownError("Unknown charge in HTTP ledger; reconcile before continuing")
        if not self.calls_remaining:
            raise BudgetExceeded("Question model-call limit reached; no request sent")
        body = {"model": model, "messages": messages, "max_tokens": self.max_tokens,
                "stream": False, "temperature": temperature}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if response_format:
            body["response_format"] = response_format
        # One token cannot consume less than one byte of text. Include a generous
        # protocol allowance; reserve output at max_tokens before issuing a call.
        rate = self.rates.get(model)
        if rate is None:
            raise ValueError("Model rates are required before any paid request")
        if any(not valid_number(rate.get(k)) for k in ("prompt", "completion")) or not valid_number(rate.get("request", 0)):
            raise ValueError("Model rates must be finite and nonnegative")
        body["provider"] = {"max_price": {"prompt": rate["prompt"] * 1e6,
                                              "completion": rate["completion"] * 1e6,
                                              "request": rate.get("request", 0)}}
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        reserve = ((len(encoded) + 4096) * rate["prompt"] + self.max_tokens * rate["completion"]
                   + rate.get("request", 0))
        if self.spent_usd + reserve >= self.budget_usd:
            raise BudgetExceeded("Experiment budget reserve reached; no request sent")
        if len(encoded) > 180000:
            raise BudgetExceeded("Prompt byte limit reached; no request sent")
        time.sleep(max(0, self.interval - (time.monotonic() - self._last_started)))
        self.scope_calls += 1
        self._last_started = time.monotonic()
        self.inflight.write_text(json.dumps({"config": self.config, "question_id": self.question_id,
                                            "model": model, "request": body}, ensure_ascii=False))
        try:
            response = self.http.post("https://openrouter.ai/api/v1/chat/completions", content=encoded,
                                      headers={"Authorization": "Bearer " + self._key,
                                               "Content-Type": "application/json"})
        except httpx.HTTPError:
            self.accounting_uncertain = True
            self._append({"timestamp": datetime.now(timezone.utc).isoformat(), "config": self.config,
                          "question_id": self.question_id, "model": model, "cost_usd": None,
                          "prompt_tokens": None, "completion_tokens": None, "status_code": None,
                          "request_id": None, "request": body, "error": "transport_error"})
            raise CostUnknownError("Network failure; no retry because billing is uncertain") from None
        except Exception:
            self.accounting_uncertain = True
            raise CostUnknownError("HTTP recording failed; reconcile the in-flight request") from None
        if self.accounting_uncertain:
            raise CostUnknownError("Response has no valid usage.cost; no further requests")
        if response.status_code != 200:
            raise OpenRouterError(f"OpenRouter rejected HTTP {response.status_code}; recorded without retry")
        data = response.json()
        try:
            usage, choice = data["usage"], data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
            if any(type(usage[k]) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError
            return {"message": message, "cost_usd": usage["cost"], "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"], "request_id": data.get("id", ""),
                    "finish_reason": choice.get("finish_reason")}
        except (KeyError, IndexError, TypeError, ValueError):
            raise OpenRouterError("Malformed paid response; cost retained in HTTP ledger") from None

    def close(self):
        self.http.close()
