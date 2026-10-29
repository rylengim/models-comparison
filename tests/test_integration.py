"""Whole experiment pipeline with synthetic API transport in a temporary directory."""
import json

from freshqa.client import OpenRouterClient
from freshqa.experiment import REQUIRED_CONFIGS, benchmark
from freshqa.reporting import render_report


def test_all_required_rows_and_costs_from_real_loop_with_fake_transport(tmp_path, monkeypatch):
    requests_seen = []

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def post(_url, **kwargs):
        body = kwargs["json"]
        requests_seen.append(body)
        if body.get("tools") and not any(m["role"] == "tool" for m in body["messages"]):
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "calc", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression":"6 * 7"}'}}]}
        else:
            message = {"role": "assistant", "content": "FINAL: 42"}
        return Response({"id": "synthetic-test-only", "choices": [{"message": message}],
                         "usage": {"cost": .01, "prompt_tokens": 10, "completion_tokens": 5}})

    monkeypatch.setattr("freshqa.client.requests.post", post)
    rows = [{"id": "one", "question": "What is six times seven?", "answer": "42"},
            {"id": "two", "question": "What is fourteen times three?", "answer": "42"}]
    client = OpenRouterClient("test-placeholder", budget_usd=1)
    traces, reports = tmp_path / "traces", tmp_path / "reports"
    assert benchmark(rows, trace_dir=traces, config_names=REQUIRED_CONFIGS, client=client, max_steps=2)
    assert len(requests_seen) == 16
    assert abs(client.spent_usd - .16) < 1e-8
    summary, fresh = render_report(rows, traces, reports)
    assert all(summary[name]["complete"] for name in REQUIRED_CONFIGS)
    assert summary["strong_no_tools"]["mean_steps"] == 1
    assert summary["cheap_search_read"]["mean_steps"] == 2
    assert abs(summary["cheap_search_read"]["cost_per_correct"] - .02) < 1e-8
    assert fresh["passed"] is False  # A fully known dataset is not fresh.
    trace = json.loads((traces / "cheap_search_read" / "one.json").read_text())
    assert any(e["type"] == "tool" and '42' in e["output"] for e in trace["events"])
    assert (reports / "money_chart.png").stat().st_size > 1000
    assert not (traces / "inflight.json").exists()
    assert benchmark(rows, trace_dir=traces, config_names=REQUIRED_CONFIGS,
                     client=client, max_steps=2, resume=True)
    assert len(requests_seen) == 16  # Resume incurs no duplicate calls.
