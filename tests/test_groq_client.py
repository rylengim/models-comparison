import pytest
import requests

from freshqa.client import BudgetExceeded, CostUnknownError
from freshqa.groq_client import GroqClient, GroqError


MODEL = "openai/gpt-oss-20b"


class Response:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def payload(model=MODEL):
    return {
        "id": "chatcmpl-test", "model": model,
        "choices": [{"message": {"role": "assistant", "content": "FINAL: Paris"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                  "prompt_tokens_details": {"cached_tokens": 900}},
    }


@pytest.fixture(autouse=True)
def forbid_unmocked_network(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("Unexpected network request"))


@pytest.mark.parametrize("model,expected", [(MODEL, .000105), ("openai/gpt-oss-120b", .00021)])
def test_estimate_uses_list_rates_without_claiming_a_billed_charge(monkeypatch, model, expected):
    seen = []

    def post(url, **kwargs):
        seen.append((url, kwargs))
        return Response(payload(model))

    monkeypatch.setattr(requests, "post", post)
    client = GroqClient("test-secret", budget_used_usd=.01)
    result = client.complete(model=model, messages=[{"role": "user", "content": "Q"}])
    assert result["cost_usd"] is None
    assert result["estimated_cost_usd"] == pytest.approx(expected)
    assert result["cost_basis"] == client.cost_basis == "list_price_estimate"
    assert result["provider"] == client.provider == "groq"
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert result["request_id"] == "chatcmpl-test"
    assert result["finish_reason"] == "stop"
    assert result["response_model"] == model
    assert client.spent_usd == 0
    assert client.budget_used_usd == pytest.approx(.01 + expected)
    assert not client.accounting_uncertain
    assert len(seen) == 1
    url, kwargs = seen[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert kwargs["timeout"] == (10, 90)
    assert kwargs["allow_redirects"] is False
    assert "tools" not in kwargs["json"]
    assert kwargs["json"]["max_completion_tokens"] == 1024
    assert kwargs["json"]["include_reasoning"] is False
    assert kwargs["json"]["reasoning_effort"] == "low"
    assert kwargs["json"]["parallel_tool_calls"] is False


def test_local_function_tools_are_passed_without_enabling_builtin_tools(monkeypatch):
    seen = []
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    def post(url, **kwargs):
        seen.append(kwargs["json"])
        return Response(payload())

    monkeypatch.setattr(requests, "post", post)
    GroqClient("test-secret").complete(model=MODEL, messages=[], tools=tools)
    assert seen[0]["tools"] == tools
    assert seen[0]["tool_choice"] == "auto"


def test_budget_guard_uses_estimated_usage_and_stops_before_next_request(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return Response(payload())

    monkeypatch.setattr(requests, "post", post)
    client = GroqClient("test-secret", budget_usd=.00001)
    client.complete(model=MODEL, messages=[])
    with pytest.raises(BudgetExceeded):
        client.complete(model=MODEL, messages=[])
    assert len(calls) == 1
    assert client.spent_usd == 0


def test_previously_used_budget_prevents_network():
    with pytest.raises(BudgetExceeded):
        GroqClient("test-secret", budget_usd=.1, budget_used_usd=.1).complete(model=MODEL, messages=[])


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 1},
                                   {"prompt_tokens": True, "completion_tokens": 1},
                                   {"prompt_tokens": -1, "completion_tokens": 1},
                                   {"prompt_tokens": 1, "completion_tokens": float("nan")}])
def test_unknown_usage_blocks_further_requests_without_inventing_cost(monkeypatch, usage):
    calls = []
    data = payload()
    data["usage"] = usage

    def post(*a, **k):
        calls.append(1)
        return Response(data)

    monkeypatch.setattr(requests, "post", post)
    client = GroqClient("test-secret")
    for _ in range(2):
        with pytest.raises(CostUnknownError):
            client.complete(model=MODEL, messages=[])
    assert len(calls) == 1
    assert client.accounting_uncertain
    assert client.budget_used_usd == client.spent_usd == 0


def test_timeout_is_not_retried_and_does_not_leak_key(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise requests.Timeout("Authorization Bearer test-secret")

    monkeypatch.setattr(requests, "post", post)
    client = GroqClient("test-secret")
    with pytest.raises(CostUnknownError) as error:
        client.complete(model=MODEL, messages=[])
    assert "test-secret" not in str(error.value)
    assert len(calls) == 1
    assert client.accounting_uncertain


@pytest.mark.parametrize("status", [401, 429])
def test_rejection_is_safe_and_not_retried(monkeypatch, status):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return Response({"error": "test-secret"}, status)

    monkeypatch.setattr(requests, "post", post)
    client = GroqClient("test-secret")
    with pytest.raises(GroqError) as error:
        client.complete(model=MODEL, messages=[])
    assert str(status) in str(error.value)
    assert "test-secret" not in str(error.value)
    assert not client.accounting_uncertain
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 500])
def test_generation_or_server_error_keeps_billing_uncertain(monkeypatch, status):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response({"error": "test-secret"}, status))
    client = GroqClient("test-secret")
    with pytest.raises(CostUnknownError) as error:
        client.complete(model=MODEL, messages=[])
    assert "test-secret" not in str(error.value)
    assert client.accounting_uncertain


def test_invalid_json_is_uncertain(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(ValueError("test-secret")))
    client = GroqClient("test-secret")
    with pytest.raises(CostUnknownError) as error:
        client.complete(model=MODEL, messages=[])
    assert "test-secret" not in str(error.value)
    assert client.accounting_uncertain


def test_malformed_message_retains_usage_estimate(monkeypatch):
    data = payload()
    data["choices"] = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(data))
    client = GroqClient("test-secret")
    with pytest.raises(GroqError):
        client.complete(model=MODEL, messages=[])
    assert client.budget_used_usd == pytest.approx(.000105)
    assert client.spent_usd == 0
    assert not client.accounting_uncertain


def test_unsupported_model_cannot_substitute_for_sonnet():
    with pytest.raises(ValueError, match="Unsupported Groq model"):
        GroqClient("test-secret").complete(model="anthropic/claude-sonnet-4.6", messages=[])


def test_unexpected_response_model_blocks_further_spending(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(payload("another-model")))
    client = GroqClient("test-secret")
    with pytest.raises(CostUnknownError, match="unexpected model"):
        client.complete(model=MODEL, messages=[])
    assert client.accounting_uncertain
    assert client.budget_used_usd == 0


def test_undocumented_provider_cost_is_not_promoted_to_a_verified_bill(monkeypatch):
    data = payload()
    data["usage"]["cost"] = 42
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(data))
    client = GroqClient("test-secret")
    result = client.complete(model=MODEL, messages=[])
    assert result["cost_usd"] is None
    assert result["estimated_cost_usd"] == pytest.approx(.000105)
    assert client.spent_usd == 0


@pytest.mark.parametrize("budget,used", [(0, 0), (-1, 0), (float("nan"), 0), (.1, -1)])
def test_invalid_budget_is_rejected(budget, used):
    with pytest.raises(ValueError):
        GroqClient("test-secret", budget_usd=budget, budget_used_usd=used)


def test_missing_key_is_actionable():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient("")
