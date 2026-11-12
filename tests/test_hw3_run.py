"""Offline checks for experiment restart, accounting and executor selection."""
import json
import sys

import pytest

import hw3.run as experiment


@pytest.fixture
def offline_experiment(tmp_path, monkeypatch):
    rows = [dict(id=f"q{i}", question=f"Question {i}?", answer="Alice") for i in range(2)]
    data = tmp_path / "hw3/data"
    data.mkdir(parents=True)
    (data / "questions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "hw3/model_rates.json").write_text("{}")
    for source in ("hw3/agents.py", "hw3/client.py", "freshqa/agent.py", "freshqa/tools.py"):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source\n")
    invocations, clients = [], []

    class Client:
        def __init__(self, api_key, ledger, **kwargs):
            self.ledger = ledger
            self.records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
            self.spent_usd = sum(record["cost_usd"] for record in self.records)
            self.closed = False
            clients.append(self)

        def set_scope(self, config, question_id, **kwargs):
            self.config, self.question_id = config, question_id

        def close(self):
            self.closed = True

    class Registry:
        def __init__(self, path):
            pass

        def close(self):
            pass

    def runner(question, *, client, **kwargs):
        invocations.append((client.config, client.question_id))
        call = dict(config=client.config, question_id=client.question_id,
                    cost_usd=.01, prompt_tokens=10, completion_tokens=2,
                    request_id=f"fixture-{len(client.records)}")
        client.records.append(call)
        client.spent_usd += .01
        with client.ledger.open("a") as stream:
            stream.write(json.dumps(call) + "\n")
        # Deliberately inconsistent runner totals: the HTTP ledger is authoritative.
        return dict(final="Alice", status="ok", cost_usd=999, steps=999, events=[])

    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "dotenv_values", lambda path: {"OPENROUTER_API_KEY": "test-only"})
    monkeypatch.setattr(experiment, "MeteredClient", Client)
    monkeypatch.setattr(experiment, "CachedRegistry", Registry)
    monkeypatch.setattr(experiment, "run_bare", runner)
    monkeypatch.setattr(experiment, "run_graph", runner)
    monkeypatch.setattr(experiment, "probe", lambda *args: None)
    monkeypatch.setattr(experiment, "render_report", lambda *args: None)
    root = tmp_path / "hw3/results/main"

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["hw3.run", "--configs", "bare", "--out", str(root), *args])
        experiment.main()

    return run, root, rows, invocations, clients


def test_resume_skips_completed_questions_without_duplicate_http_cost(offline_experiment):
    run, root, _, invocations, clients = offline_experiment
    run()
    before = (root.parent / "calls.jsonl").read_bytes()
    run("--resume")
    assert invocations == [("bare", "q0"), ("bare", "q1")]
    assert (root.parent / "calls.jsonl").read_bytes() == before
    assert clients[-1].spent_usd == pytest.approx(.02)
    assert all(client.closed for client in clients)


def test_results_take_calls_tokens_and_cost_from_http_records(offline_experiment):
    run, root, _, _, _ = offline_experiment
    run()
    result = json.loads((root / "bare/q0.json").read_text())
    assert result["steps"] == 1
    assert result["cost_usd"] == .01
    assert result["prompt_tokens"] == 10 and result["completion_tokens"] == 2
    assert result["measured"] is True and result["correct"] is True
    assert not (root / "inflight_question.json").exists()


def test_changed_agent_source_blocks_resume_before_client_creation(offline_experiment):
    run, root, _, invocations, clients = offline_experiment
    run()
    (root.parents[1] / "agents.py").write_text("# changed agent\n")
    with pytest.raises(ValueError, match="exactly the same"):
        run("--resume")
    assert len(clients) == 1 and len(invocations) == 2


def test_existing_run_requires_explicit_resume(offline_experiment):
    run, _, _, invocations, clients = offline_experiment
    run()
    with pytest.raises(ValueError, match="--resume"):
        run()
    assert len(clients) == 1 and len(invocations) == 2


def test_inflight_question_blocks_resume_before_any_model_call(offline_experiment):
    run, root, _, invocations, clients = offline_experiment
    run()
    experiment.save(root / "inflight_question.json", {"config": "bare", "id": "q0"})
    with pytest.raises(RuntimeError, match="Interrupted question"):
        run("--resume")
    assert len(clients) == 1 and len(invocations) == 2


def write_candidate(root, config, row, *, final="Alice", cost=.01, status="ok"):
    experiment.save(root / config / (row["id"] + ".json"),
                    dict(final=final, cost_usd=cost, status=status))


@pytest.mark.parametrize("bare_answer,bare_cost,graph_answer,graph_cost,expected", [
    ("Alice", .5, "Bob", .001, "bare"),
    ("Alice", .5, "Alice", .01, "langgraph"),
    ("Bob", .001, "Alice", .5, "langgraph"),
])
def test_executor_selection_prioritizes_accuracy_then_cost(
        tmp_path, bare_answer, bare_cost, graph_answer, graph_cost, expected):
    rows = [dict(id="q", question="Who?", answer="Alice")]
    write_candidate(tmp_path, "bare", rows[0], final=bare_answer, cost=bare_cost)
    write_candidate(tmp_path, "langgraph", rows[0], final=graph_answer, cost=graph_cost)
    runner, selected = experiment.choose_runner(tmp_path, rows)
    assert selected == expected
    assert runner is (experiment.run_bare if expected == "bare" else experiment.run_graph)
    assert json.loads((tmp_path / "executor_selection.json").read_text())["selected"] == expected


def test_executor_selection_refuses_incomplete_baseline(tmp_path):
    rows = [dict(id="q", question="Who?", answer="Alice")]
    write_candidate(tmp_path, "bare", rows[0])
    with pytest.raises(ValueError, match="Complete bare and langgraph"):
        experiment.choose_runner(tmp_path, rows)


@pytest.mark.parametrize("cost,status", [(None, "ok"), (.01, "error"), (.01, "budget_exceeded")])
def test_executor_selection_refuses_uncertain_or_failed_baseline(tmp_path, cost, status):
    rows = [dict(id="q", question="Who?", answer="Alice")]
    write_candidate(tmp_path, "bare", rows[0], cost=cost, status=status)
    write_candidate(tmp_path, "langgraph", rows[0])
    with pytest.raises(ValueError, match="interrupted configuration"):
        experiment.choose_runner(tmp_path, rows)
