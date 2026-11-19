import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
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


def test_concurrent_resume_does_not_repeat_a_pending_task(tmp_path, fake_environment, monkeypatch):
    before_marker, release_writer = Event(), Event()
    marker = tmp_path / "inflight.json"
    original_open = Path.open

    def pause_before_first_marker(path, mode="r", *args, **kwargs):
        if path == marker and mode == "x" and not before_marker.is_set():
            before_marker.set()
            assert release_writer.wait(5), "The competing benchmark did not finish"
        return original_open(path, mode, *args, **kwargs)

    options = {"trace_dir": tmp_path, "config_names": ["cheap_no_tools"]}
    exhausted_client = SimpleNamespace(spent_usd=5.0, budget_usd=5.0)
    assert not benchmark(dataset(), client=exhausted_client, **options)
    options["resume"] = True
    monkeypatch.setattr(Path, "open", pause_before_first_marker)
    first_client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    second_client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(benchmark, dataset(), client=first_client, **options)
        try:
            assert before_marker.wait(5), "The first benchmark did not reach the in-flight marker"
            with pytest.raises(ValueError, match=r"^Этот каталог уже занят работающим benchmark\.$"):
                benchmark(dataset(), client=second_client, **options)
        finally:
            release_writer.set()
        assert first.result(timeout=5)

    assert first_client.spent_usd == pytest.approx(.01)
    assert second_client.spent_usd == 0.0
    assert len(fake_environment) == 1
    trace_path = tmp_path / "cheap_no_tools" / "one.json"
    assert list(tmp_path.glob("*/*.json")) == [trace_path]
    trace = trace_path.read_bytes()
    decoded = json.loads(trace)
    assert decoded["id"] == "one" and decoded["status"] == "ok"
    assert decoded["cost_usd"] == pytest.approx(.01)
    assert not marker.exists()

    assert benchmark(dataset(), client=second_client, **options)
    assert second_client.spent_usd == pytest.approx(.01)
    assert len(fake_environment) == 1
    assert trace_path.read_bytes() == trace


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


@pytest.mark.parametrize("status,finish_reason", [
    ("output_truncated", "length"),
    ("output_truncated", "stop"),
    ("ok", "length"),
])
def test_benchmark_does_not_credit_truncated_matching_final(tmp_path, monkeypatch, status, finish_reason):
    def run(question, *, client, **kwargs):
        client.spent_usd += .01
        return dict(final="Alice", status=status, cost_usd=.01, known_cost_usd=.01,
                    steps=1, elapsed_s=.1, events=[
                        {"type": "llm", "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": "FINAL: Alice"}}])

    monkeypatch.setattr("freshqa.agent.run_agent", run)
    monkeypatch.setattr("freshqa.tools.ToolRegistry", lambda **kwargs: object())
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    assert benchmark(dataset(), trace_dir=tmp_path, config_names=["cheap_no_tools"], client=client)
    trace = json.loads((tmp_path / "cheap_no_tools" / "one.json").read_text())
    assert trace["correct"] is False
    result = aggregate([trace], total=1)
    assert result["n"] == 1 and result["complete"]
    assert result["cost_total"] == pytest.approx(.01)
    assert result["accuracy"] == 0


@pytest.mark.parametrize("status,cost,finish_reason,content,tool_calls,expected", [
    ("ok", .01, "stop", "FINAL: Alice", [], True),
    ("ok", .01, None, "FINAL: Alice", [], True),
    ("ok", .01, "content_filter", "FINAL: Alice", [], False),
    ("ok", .01, "tool_calls", "FINAL: Alice", [], False),
    ("ok", .01, "future_unknown_reason", "FINAL: Alice", [], False),
    ("max_steps", .01, "stop", "FINAL: Alice", [], True),
    ("tool_limit", .01, "stop", "FINAL: Alice", [], True),
    ("repeated_tool_call", .01, "stop", "FINAL: Alice", [], True),
    ("error", .01, "stop", "FINAL: Alice", [], False),
    ("budget_exceeded", .01, "stop", "FINAL: Alice", [], False),
    ("cost_unknown", None, "stop", "FINAL: Alice", [], False),
    ("ok", None, "stop", "FINAL: Alice", [], False),
    ("empty_response", .01, "stop", "FINAL: Alice", [], False),
    ("invalid_tool_calls", .01, "stop", "FINAL: Alice", [], False),
    ("unexpected_tool_call", .01, "stop", "FINAL: Alice", [], False),
    ("max_steps", .01, "tool_calls", "FINAL: Alice", [{"id": "pending"}], False),
    ("ok", .01, "stop", "", [], False),
])
def test_benchmark_and_report_share_versioned_final_policy(
        tmp_path, monkeypatch, status, cost, finish_reason, content, tool_calls, expected):
    from freshqa.dataset import METRIC_VERSION
    from freshqa.evaluation import GRADING_POLICY_VERSION, HARD_FAILURE_STATUSES
    from freshqa.reporting import render_report

    def run(question, *, client, **kwargs):
        client.spent_usd += .01
        return dict(final="Alice", status=status, cost_usd=cost, known_cost_usd=.01,
                    steps=2, elapsed_s=.1, events=[
                        {"type": "llm", "early_final": True, "finish_reason": "length",
                         "message": {"content": "unfinished candidate"}},
                        {"type": "llm", "finish_reason": finish_reason,
                         "message": {"content": content, "tool_calls": tool_calls}}])

    monkeypatch.setattr("freshqa.agent.run_agent", run)
    monkeypatch.setattr("freshqa.tools.ToolRegistry", lambda **kwargs: object())
    monkeypatch.setattr("freshqa.reporting._chart", lambda *args: None)
    root, out = tmp_path / "traces", tmp_path / "reports"
    client = SimpleNamespace(spent_usd=0.0, budget_usd=5.0)
    assert benchmark(dataset(), trace_dir=root, config_names=["cheap_no_tools"], client=client) == (
        status not in HARD_FAILURE_STATUSES)
    path = root / "cheap_no_tools" / "one.json"
    original = path.read_bytes()
    trace = json.loads(original)
    assert trace["correct"] is expected
    assert trace["grading_policy_version"] == GRADING_POLICY_VERSION
    assert trace["metric_version"] == METRIC_VERSION
    signature = json.loads((root / "manifest.json").read_text())["signature"]
    assert signature["grading_policy_version"] == GRADING_POLICY_VERSION
    assert signature["metric_version"] == METRIC_VERSION
    summary, _ = render_report(dataset(), root, out)
    value = summary["cheap_no_tools"]
    assert value["correct"] == int(expected)
    assert value["n"] == 1 and value["accuracy"] == int(expected)
    assert value["cost_total"] == cost
    assert value["complete"] == (cost is not None and status not in HARD_FAILURE_STATUSES)
    sidecar = json.loads((out / "grading.json").read_text())
    assert sidecar["grading_policy_version"] == GRADING_POLICY_VERSION
    assert sidecar["grades"][0]["historical_correct"] is expected
    assert sidecar["grades"][0]["current_correct"] is expected
    assert path.read_bytes() == original


@pytest.mark.parametrize("changes", [
    {"status": "output_truncated"},
    {"status": "error"},
    {"cost_usd": None},
    {"finish_reason": "length"},
    {"finish_reason": "content_filter"},
    {"finish_reason": "future_unknown_reason"},
    {"events": [{"type": "llm", "finish_reason": "length"}]},
    {"events": [{"type": "llm", "error": "failed"}]},
])
def test_ineligible_final_never_enters_text_metric(monkeypatch, changes):
    from freshqa.evaluation import score_trace

    def forbidden(*args):
        raise AssertionError("The text metric must not see an ineligible final")

    monkeypatch.setattr("freshqa.evaluation.score_row", forbidden)
    trace = dict(final="Alice", status="ok", cost_usd=.01, events=[])
    assert score_trace(dataset()[0], {**trace, **changes}) is False
