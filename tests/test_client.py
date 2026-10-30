import pytest
import requests

from freshqa.agent import run_agent
from freshqa.client import BudgetExceeded, CostUnknownError, OpenRouterClient, OpenRouterError
from freshqa.reporting import freshness


class Response:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def json(self):
        return self.body


def payload(cost=0.012):
    return {
        "id": "gen-123",
        "choices": [{"message": {"role": "assistant", "content": "FINAL: Paris"},
                     "finish_reason": "stop"}],
        "usage": {"cost": cost, "prompt_tokens": 15, "completion_tokens": 4},
    }


def test_actual_billed_cost_and_token_usage_are_recorded(monkeypatch):
    seen = []

    def post(url, **kwargs):
        seen.append((url, kwargs))
        return Response(payload())

    monkeypatch.setattr(requests, "post", post)
    client = OpenRouterClient("secret", spent_usd=0.1)
    result = client.complete(model="example/model", messages=[{"role": "user", "content": "Q"}])
    assert result == {
        "message": {"role": "assistant", "content": "FINAL: Paris"},
        "cost_usd": 0.012,
        "prompt_tokens": 15,
        "completion_tokens": 4,
        "request_id": "gen-123",
        "finish_reason": "stop",
    }
    assert client.spent_usd == pytest.approx(0.112)
    assert len(seen) == 1
    assert seen[0][1]["timeout"] == (10, 90)
    assert seen[0][1]["allow_redirects"] is False
    assert "tools" not in seen[0][1]["json"]
    assert seen[0][1]["json"]["max_tokens"] == 512


def test_real_zero_cost_is_valid(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(payload(0)))
    assert OpenRouterClient("secret").complete(model="free/model", messages=[])["cost_usd"] == 0


def test_truncated_openrouter_response_preserves_cost_without_passing_freshness(monkeypatch):
    data = payload()
    data["choices"][0].update(finish_reason="length")
    data["choices"][0]["message"]["content"] = "FINAL: incomplete"
    data["usage"]["completion_tokens"] = 512
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(data))
    client = OpenRouterClient("secret")

    result = run_agent("Who won?", model="anthropic/claude-sonnet-4.6", client=client,
                       registry=None, tool_names=[])

    assert result["status"] == "output_truncated"
    assert result["cost_usd"] == result["known_cost_usd"] == pytest.approx(.012)
    assert client.spent_usd == pytest.approx(.012)
    assert result["events"][0]["finish_reason"] == "length"
    result.update(id="synthetic", correct=False)
    report = freshness([result], total=1, model="anthropic/claude-sonnet-4.6")
    assert report["complete"] is False
    assert report["passed"] is None


def test_spending_limit_prevents_network_request(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("budget exceeded POST"))
    with pytest.raises(BudgetExceeded):
        OpenRouterClient("secret", budget_usd=0.5, spent_usd=0.5).complete(model="x", messages=[])


def test_last_response_is_accounted_before_next_call_hits_budget(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(payload(0.02)))
    client = OpenRouterClient("secret", budget_usd=0.01)
    assert client.complete(model="x", messages=[])["cost_usd"] == 0.02
    assert client.spent_usd == 0.02
    with pytest.raises(BudgetExceeded):
        client.complete(model="x", messages=[])


@pytest.mark.parametrize("cost", [None, "bad", -1, float("nan"), float("inf"), True])
def test_unknown_or_invalid_cost_blocks_subsequent_spending(monkeypatch, cost):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return Response(payload(cost))

    monkeypatch.setattr(requests, "post", post)
    client = OpenRouterClient("secret")
    with pytest.raises(CostUnknownError):
        client.complete(model="x", messages=[])
    with pytest.raises(CostUnknownError):
        client.complete(model="x", messages=[])
    assert client.accounting_uncertain is True
    assert len(calls) == 1


def test_timeout_does_not_retry_or_leak_key(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("Authorization Bearer secret")

    monkeypatch.setattr(requests, "post", post)
    client = OpenRouterClient("secret")
    with pytest.raises(CostUnknownError) as error:
        client.complete(model="x", messages=[])
    assert "secret" not in str(error.value)
    assert len(calls) == 1
    assert client.accounting_uncertain is True


def test_authentication_error_is_safe_and_not_retried(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response({"error": "secret"}, 401))
    with pytest.raises(OpenRouterError) as error:
        OpenRouterClient("secret").complete(model="x", messages=[])
    assert "secret" not in str(error.value)
    assert "401" in str(error.value)


def test_server_error_marks_possible_charge_unknown(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response({"error": "secret"}, 500))
    client = OpenRouterClient("secret")
    with pytest.raises(CostUnknownError) as error:
        client.complete(model="x", messages=[])
    assert "secret" not in str(error.value)
    assert client.accounting_uncertain is True


def test_paid_malformed_response_retains_known_cost(monkeypatch):
    data = payload()
    data["choices"] = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response(data))
    client = OpenRouterClient("secret")
    with pytest.raises(OpenRouterError):
        client.complete(model="x", messages=[])
    assert client.spent_usd == pytest.approx(0.012)
    assert client.accounting_uncertain is False


@pytest.mark.parametrize("budget,spent", [(0, 0), (-1, 0), (float("nan"), 0), (5, -1)])
def test_invalid_budget_configuration_is_rejected(budget, spent):
    with pytest.raises(ValueError):
        OpenRouterClient("secret", budget_usd=budget, spent_usd=spent)


def test_missing_api_key_is_actionable_without_network():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient("")
