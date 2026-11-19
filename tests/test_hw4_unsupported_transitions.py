"""Finite decision table for the early-answer verifier and its bounded fallback."""

import json

import pytest

from freshqa.agent import ToolRequirement
from freshqa.client import BudgetExceeded, CostUnknownError, OpenRouterError
from hw4.mcp_openrouter import MAX_STEPS, _accept_early_final, _require_page_find


SELECTED = "https://example.invalid/selected"
OTHER = "https://example.invalid/already-read"
UNREAD = "https://example.invalid/unread"
QUOTE = "The selected passage contains an unrelated observation."


class VerifierClient:
    def __init__(self, calls, outcome="UNSUPPORTED"):
        self.scope_calls = calls
        self.spent_usd = 0.0
        self.outcome = outcome
        self.requests = []

    def complete(self, **request):
        self.requests.append(request)
        assert request["evidence_check"] is True
        assert "tools" not in request
        payload = json.loads(request["messages"][-1]["content"])
        assert payload["quote"] == QUOTE
        if self.outcome == "budget":
            raise BudgetExceeded("The next request exceeds its local reserve")
        self.scope_calls += 1
        if self.outcome == "unknown-cost":
            raise CostUnknownError("The request has an unknown charge")
        if self.outcome == "provider-error":
            raise OpenRouterError("The provider rejected the request")
        if self.outcome == "unexpected-error":
            raise RuntimeError("Private exception detail")
        self.spent_usd += 0.0001
        message = {"role": "assistant", "content": self.outcome}
        if self.outcome == "truncated":
            message["content"] = "UNSUPPORTED"
        if self.outcome == "wrong-role":
            message.update(role="user", content="UNSUPPORTED")
        if self.outcome == "tool-call":
            message.update(content="UNSUPPORTED", tool_calls=[{"id": "unexpected"}])
        return {
            "message": message,
            "finish_reason": "length" if self.outcome == "truncated" else "stop",
            "request_id": "mock-verifier", "cost_usd": 0.0001,
            "prompt_tokens": 100, "completion_tokens": 2,
        }


def early_history(remaining, unread, other_read):
    judge_step = MAX_STEPS - remaining
    read_step = judge_step - 2
    urls = [SELECTED]
    if other_read:
        urls.append(OTHER)
    if unread:
        urls.append(UNREAD)
    events = [
        {"type": "llm", "step": 1},
        {"type": "tool", "step": 1, "name": "web_search",
         "arguments": {"query": "test observation"},
         "output": json.dumps({"results": [{"url": url} for url in urls]})},
        *[{"type": "llm", "step": step} for step in range(2, read_step + 1)],
    ]
    if other_read:
        events.append({
            "type": "tool", "step": read_step, "name": "page_find", "arguments": {"url": OTHER},
            "output": json.dumps({"url": OTHER, "found": True, "excerpts": ["> An earlier observation."]}),
        })
    events.append({
        "type": "tool", "step": read_step, "name": "page_find", "arguments": {"url": SELECTED},
        "output": json.dumps({"url": SELECTED, "found": True, "excerpts": [f"> {QUOTE}"]}),
    })
    final = f"candidate answer\nQUOTE_ID: r{3 if other_read else 2}-b1"
    events.append({
        "type": "llm", "step": judge_step - 1, "early_final": True,
        "message": {"role": "assistant", "content": "FINAL: " + final},
    })
    return events, final


@pytest.mark.parametrize(("remaining", "unread", "other_read", "expected"), [
    pytest.param(0, False, False, "stop", id="zero-none"),
    pytest.param(0, False, True, "stop", id="zero-other"),
    pytest.param(0, True, False, "stop", id="zero-unread"),
    pytest.param(0, True, True, "stop", id="zero-both"),
    pytest.param(1, False, False, "stop", id="one-none"),
    pytest.param(1, False, True, "final", id="one-other"),
    pytest.param(1, True, False, "stop", id="one-unread"),
    pytest.param(1, True, True, "final", id="one-both"),
    pytest.param(2, False, False, "stop", id="two-none"),
    pytest.param(2, False, True, "final", id="two-other"),
    pytest.param(2, True, False, "read", id="two-unread"),
    pytest.param(2, True, True, "read", id="two-both"),
])
def test_unsupported_transition_table(remaining, unread, other_read, expected):
    events, final = early_history(remaining, unread, other_read)
    client = VerifierClient(MAX_STEPS - remaining - 1)

    decision = _accept_early_final(final, events, question="Which observation?", model="mock", client=client)

    assert decision.accepted is False and decision.additional_steps == 1
    assert len(client.requests) == 1
    judge = events[-1]
    assert judge["step"] == MAX_STEPS - remaining
    assert judge["evidence_check"] is True and judge["evidence_supported"] is False
    if expected == "read":
        assert decision.finalize is False and decision.stop_status is None
        assert judge["fallback_read_urls"] == [UNREAD]
        assert UNREAD in decision.retry_message
        requirement = _require_page_find(events)
        assert requirement == ToolRequirement("page_find", "url", (UNREAD,))
    elif expected == "final":
        assert decision.finalize is True and decision.stop_status is None
        assert "QUOTE_ID" in decision.retry_message
        assert "fallback_read_urls" not in judge
    else:
        assert decision.finalize is False and decision.stop_status == "unsupported_answer"
        assert decision.retry_message is None and "fallback_read_urls" not in judge


@pytest.mark.parametrize(("outcome", "status", "extra_calls"), [
    ("UNSUPPORTED because the quote is unrelated", "unsupported_answer", 1),
    ("UNSUPPORTED.", "unsupported_answer", 1),
    ("truncated", "unsupported_answer", 1),
    ("wrong-role", "unsupported_answer", 1),
    ("tool-call", "unsupported_answer", 1),
    ("budget", "budget_exceeded", 0),
    ("unknown-cost", "cost_unknown", 1),
    ("provider-error", "error", 1),
    ("unexpected-error", "error", 1),
])
def test_verifier_failure_stops_despite_available_fallback(outcome, status, extra_calls):
    events, final = early_history(remaining=2, unread=True, other_read=True)
    client = VerifierClient(calls=3, outcome=outcome)

    decision = _accept_early_final(final, events, question="Which observation?", model="mock", client=client)

    assert decision.accepted is False and decision.stop_status == status
    assert decision.additional_steps == extra_calls and client.scope_calls == 3 + extra_calls
    assert decision.finalize is False and decision.retry_message is None
    assert "fallback_read_urls" not in events[-1]
    assert events[-1]["evidence_supported"] is False
    if outcome == "unexpected-error":
        assert events[-1]["error"] == "Evidence check failed"


def test_supported_verdict_accepts_without_requesting_another_read():
    events, final = early_history(remaining=2, unread=True, other_read=True)
    client = VerifierClient(calls=3, outcome="SUPPORTED")

    decision = _accept_early_final(final, events, question="Which observation?", model="mock", client=client)

    assert decision.accepted is True and decision.additional_steps == 1
    assert decision.finalize is False and decision.stop_status is None
    assert decision.retry_message is None and "fallback_read_urls" not in events[-1]
    assert events[-1]["evidence_supported"] is True
