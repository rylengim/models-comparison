from types import SimpleNamespace

import pytest

from freshqa.agent import run_agent
from freshqa.experiment import benchmark


def test_groq_estimate_does_not_become_reported_cost():
    client = SimpleNamespace(spent_usd=0, budget_used_usd=0,
                             cost_basis="list_price_estimate", provider="groq")

    def complete(**kwargs):
        client.budget_used_usd += .001
        return {"message": {"role": "assistant", "content": "FINAL: Alice"},
                "cost_usd": None, "estimated_cost_usd": .001,
                "cost_basis": "list_price_estimate", "provider": "groq",
                "prompt_tokens": 5, "completion_tokens": 2, "request_id": "test"}

    client.complete = complete
    result = run_agent("Who won?", model="test", client=client, registry=None, tool_names=[])
    assert result["status"] == "ok" and result["final"] == "Alice"
    assert result["cost_usd"] is None
    assert result["known_cost_usd"] is None
    assert result["estimated_cost_usd"] == .001
    assert result["events"][0]["cost_usd"] is None
    assert result["events"][0]["estimated_cost_usd"] == .001
    assert result["provider"] == "groq"


def test_main_benchmark_refuses_to_mix_groq_estimates_with_billing(tmp_path):
    client = SimpleNamespace(cost_basis="list_price_estimate")
    with pytest.raises(ValueError, match="groq_check"):
        benchmark([], trace_dir=tmp_path, config_names=[], client=client)
    assert list(tmp_path.iterdir()) == []


def test_truncated_completion_is_not_reported_as_normal_success():
    client = SimpleNamespace(spent_usd=0)
    client.complete = lambda **kwargs: {
        "message": {"role": "assistant", "content": "FINAL: incomplete"},
        "finish_reason": "length", "cost_usd": 0, "prompt_tokens": 1,
        "completion_tokens": 1024, "request_id": "test"}
    result = run_agent("Who won?", model="test", client=client, registry=None, tool_names=[])
    assert result["status"] == "output_truncated"
