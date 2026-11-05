import pytest
import requests

from freshqa.client import BudgetExceeded, CostUnknownError
from hw2 import benchmark_client
from hw2.benchmark_client import BenchmarkClient


MODEL = "openai/gpt-4o-mini"
MESSAGES = [{"role": "user", "content": "A question"}]


def test_price_ceiling_and_pacing_do_not_replace_actual_cost(monkeypatch):
    clock, starts = [100.0], []
    client = BenchmarkClient("placeholder", 0.5)
    assert client.session.trust_env is False
    monkeypatch.setattr(benchmark_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark_client.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    class Response:
        status_code = 200

        def json(self):
            return {"id": "local-example", "usage": {"cost": 0.000081, "prompt_tokens": 25, "completion_tokens": 4},
                    "choices": [{"message": {"role": "assistant", "content": "unknown"}, "finish_reason": "stop"}]}

    def post(url, *, json, **kwargs):
        starts.append(clock[0])
        assert json["provider"]["max_price"] == {"prompt": 0.15, "completion": 0.6, "request": 0}
        assert json["max_tokens"] == 512 and not kwargs["allow_redirects"]
        return Response()

    monkeypatch.setattr(client.session, "post", post)
    for _ in range(2):
        assert client.complete(model=MODEL, messages=MESSAGES)["request_id"] == "local-example"
    assert client.spent_usd == pytest.approx(0.000162)
    assert starts == [100.0, 102.5]


def test_unaffordable_request_unknown_price_and_non_text_are_blocked_before_post(monkeypatch):
    client = BenchmarkClient("placeholder", 0.01, spent_usd=0.0099)
    monkeypatch.setattr(client.session, "post", lambda *a, **kw: pytest.fail("unexpected POST"))
    with pytest.raises(BudgetExceeded):
        client.complete(model=MODEL, messages=MESSAGES)
    with pytest.raises(ValueError, match="price cap"):
        client.complete(model="unknown/model", messages=MESSAGES)
    with pytest.raises(ValueError, match="text-only"):
        client.complete(model=MODEL, messages=[{"role": "user", "content": [{"type": "image_url"}]}])
    with pytest.raises(ValueError, match="function tools"):
        client.complete(model=MODEL, messages=MESSAGES, tools=[{"type": "web_search"}])
    assert client.spent_usd == 0.0099


def test_session_timeout_is_not_retried(monkeypatch):
    client = BenchmarkClient("placeholder", 0.5)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("potentially billed")

    monkeypatch.setattr(client.session, "post", timeout)
    for _ in range(2):
        with pytest.raises(CostUnknownError):
            client.complete(model=MODEL, messages=MESSAGES)
    assert calls == [1] and client.accounting_uncertain
