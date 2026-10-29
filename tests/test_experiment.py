import json
from types import SimpleNamespace

import pytest

from freshqa.experiment import aggregate, benchmark, run_cascade


def test_aggregate_counts_failures_and_no_correct_answers():
    rows = [dict(correct=False, cost_usd=0.1, steps=2, elapsed_s=3),
            dict(correct=False, cost_usd=0.2, steps=4, elapsed_s=5)]
    value = aggregate(rows, total=3)
    assert value["n"] == 2 and not value["complete"]
    assert value["accuracy"] == 0
    assert value["cost_per_correct"] is None
    assert abs(value["cost_per_task"] - 0.15) < 1e-9
    assert value["mean_steps"] == 3


def test_unknown_cost_is_not_silently_counted_as_zero():
    value = aggregate([dict(correct=True, cost_usd=None, steps=1, elapsed_s=1)], 1)
    assert value["cost_per_task"] is None
    assert value["cost_per_correct"] is None
    assert not value["complete"]


@pytest.mark.parametrize("status", ["error", "budget_exceeded", "cost_unknown"])
def test_api_failure_is_not_a_complete_experiment(status):
    value = aggregate([dict(correct=False, cost_usd=0, steps=1, elapsed_s=1, status=status)], 1)
    assert value["n"] == 1
    assert value["accuracy"] == 0
    assert not value["complete"]


def test_wrong_model_answer_is_still_a_completed_measurement():
    value = aggregate([dict(correct=False, cost_usd=.01, steps=2, elapsed_s=1, status="max_steps")], 1)
    assert value["complete"]
    assert value["accuracy"] == 0


def test_empty_experiment_has_missing_metrics():
    value = aggregate([], 10)
    assert value["accuracy"] is None
    assert value["cost_per_task"] is None
    assert not value["complete"]


def test_cascade_fallback_uses_unknown_not_gold_answer():
    calls = []

    def run(question, **kwargs):
        calls.append(kwargs["model"])
        return dict(final="unknown" if len(calls) == 1 else "Alice", status="ok",
                    cost_usd=0.01, steps=1, elapsed_s=0.1, events=[])

    value = run_cascade("Who won?", cheap="small", medium="medium", client=None,
                        registry=None, tool_names=[], max_steps=3, run=run)
    assert calls == ["small", "medium"]
    assert value["final"] == "Alice"
    assert value["cost_usd"] == 0.02
    assert value["steps"] == 2


def dataset():
    return [{"id": "one", "question": "Who won?", "answer": "Alice",
             "source": "test", "evidence": "Alice won.", "url": "https://example.org"}]


@pytest.fixture
def fake_environment(monkeypatch):
    calls = []

    def run(question, *, client, **kwargs):
        calls.append((question, kwargs))
        client.spent_usd += .01
        return dict(final="Alice", status="ok", cost_usd=.01, known_cost_usd=.01,
                    steps=1, elapsed_s=.1, events=[])

    monkeypatch.setattr("freshqa.agent.run_agent", run)
    monkeypatch.setattr("freshqa.tools.ToolRegistry", lambda **kwargs: object())
    return calls


def test_new_trace_directory_preserves_existing_client_spending(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=5.0, budget_usd=5.0)
    complete = benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    assert not complete
    assert client.spent_usd == 5.0
    assert fake_environment == []


def test_resume_restores_cost_and_does_not_repeat_saved_task(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    assert benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    fresh_client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    assert benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"],
                     client=fresh_client, resume=True)
    assert fresh_client.spent_usd == pytest.approx(.01)
    assert len(fake_environment) == 1
    question, arguments = fake_environment[0]
    assert question == "Who won?"
    assert all(name not in arguments for name in ("answer", "expected", "url", "evidence"))
    assert not (tmp_path / "inflight.json").exists()


def test_resume_preserves_client_spending_larger_than_saved_trace_total(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    client.spent_usd = 1.0
    benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client, resume=True)
    assert client.spent_usd == 1.0
    assert len(fake_environment) == 1


def test_interrupted_paid_task_blocks_resume_before_another_request(tmp_path, fake_environment, monkeypatch):
    def interrupted(question, *, client, **kwargs):
        client.spent_usd += .02
        raise KeyboardInterrupt

    monkeypatch.setattr("freshqa.agent.run_agent", interrupted)
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    with pytest.raises(KeyboardInterrupt):
        benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    marker = tmp_path / "inflight.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["id"] == "one"
    with pytest.raises(ValueError, match="незавершён"):
        benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"],
                  client=SimpleNamespace(spent_usd=0.0, budget_usd=5.0), resume=True)


def test_inflight_marker_blocks_resume_even_when_a_trace_exists(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    (tmp_path / "inflight.json").write_text("{}")
    with pytest.raises(ValueError, match="незавершён"):
        benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client, resume=True)
    assert len(fake_environment) == 1


def test_resume_with_unknown_cost_is_blocked(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    path = tmp_path / "cheap_no_tools" / "one.json"
    trace = json.loads(path.read_text())
    trace["cost_usd"] = None
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="неизвестные списания"):
        benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client, resume=True)
    assert len(fake_environment) == 1


def test_resumed_api_failure_cannot_be_reported_as_successful(tmp_path, fake_environment):
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    path = tmp_path / "cheap_no_tools" / "one.json"
    trace = json.loads(path.read_text())
    trace["status"] = "error"
    path.write_text(json.dumps(trace))
    assert not benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"],
                         client=client, resume=True)
    assert len(fake_environment) == 1
