import copy
import json

import pytest

from freshqa.client import BudgetExceeded, CostUnknownError
from hw3.agents import run_bare, run_graph, run_planned, run_critic


def answer(text):
    return {"role": "assistant", "content": text}


def search(query="q", call_id="one"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "web_search", "arguments": json.dumps({"query": query})}}]}


class Client:
    def __init__(self, responses, limit=100):
        self.responses = iter(responses)
        self.calls = []
        self.calls_remaining = limit
        self.spent_usd = 0.0
        self.accounting_uncertain = False

    def complete(self, **kwargs):
        if self.calls_remaining <= 0:
            raise BudgetExceeded("Question call limit reached")
        self.calls_remaining -= 1
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            if isinstance(response, CostUnknownError):
                self.accounting_uncertain = True
            raise response
        self.spent_usd += 0.01
        return {"message": response, "cost_usd": 0.01, "prompt_tokens": 10,
                "completion_tokens": 5, "request_id": str(len(self.calls))}


class Registry:
    def __init__(self):
        self.calls = []

    def schemas(self, names):
        return [{"type": "function", "function": {"name": name}} for name in names]

    def run(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"source":"fixture", "text":"Paris is the capital of France."}'


def execute(function, responses, **kwargs):
    client = Client(responses)
    registry = Registry()
    result = function("Capital of France?", client=client, registry=registry, model="m", **kwargs)
    return result, client, registry


def test_framework_first_payload_and_tool_conversation_match_bare():
    responses = [search(), answer("FINAL: Paris")]
    bare, bc, _ = execute(run_bare, responses)
    graph, gc, _ = execute(run_graph, responses)
    assert bc.calls == gc.calls
    assert graph["final"] == bare["final"] == "Paris"
    assert graph["steps"] == bare["steps"] == 2
    assert graph["cost_usd"] == bare["cost_usd"] == 0.02


def test_framework_repeat_guard_and_reserved_final_match_bare():
    responses = [search(), search(call_id="two"), answer("FINAL: Paris")]
    bare, bc, br = execute(run_bare, responses)
    graph, gc, gr = execute(run_graph, responses)
    assert gc.calls == bc.calls
    assert gr.calls == br.calls == [("web_search", {"query": "q"})]
    assert graph["status"] == bare["status"] == "repeated_tool_call"
    assert gc.calls[-1]["tools"] is None


@pytest.mark.parametrize("runner", [run_bare, run_graph])
def test_question_call_limit_stops_without_extra_request(runner):
    client = Client([search()], limit=1)
    result = runner("Q", client=client, registry=Registry(), model="m")
    assert result["status"] == "budget_exceeded"
    assert result["final"] == "unknown"
    assert len(client.calls) == result["steps"] == 1


@pytest.mark.parametrize("runner", [run_bare, run_graph])
def test_last_step_has_no_tools(runner):
    result, client, _ = execute(runner, [search(), answer("FINAL: Paris")], max_steps=2)
    assert result["status"] == "max_steps"
    assert client.calls[-1]["tools"] is None


def test_plan_accounts_for_planner_and_does_not_include_gold():
    result, client, _ = execute(run_planned, [answer("Search official sources."),
                                            answer("FINAL: Paris")], runner=run_graph)
    assert result["final"] == "Paris"
    assert result["steps"] == len(client.calls) == 2
    assert result["cost_usd"] == 0.02
    assert "Search official sources." in client.calls[1]["messages"][1]["content"]
    assert client.calls[0].get("tools") is None


def test_planner_spends_last_call_and_executor_does_not_bypass_cap():
    client = Client([answer("Search France")], limit=1)
    result = run_planned("Q", runner=run_graph, client=client, registry=Registry(), model="m")
    assert result["status"] == "budget_exceeded"
    assert result["final"] == "unknown"
    assert len(client.calls) == 1


def test_critic_records_initial_answer_and_corrects_using_evidence():
    responses = [search(), answer("FINAL: London"),
                 answer('{"supported":false,"issue":"The retrieved source names Paris."}'),
                 search("France", "two"), answer("FINAL: Paris"),
                 answer('{"supported":true,"issue":""}')]
    result, client, _ = execute(run_critic, responses, runner=run_graph)
    assert result["initial_answer"] == "London"
    assert result["final"] == "Paris"
    assert result["verified"] is True
    assert result["checks"] == 2
    assert result["corrections"] == 1
    assert result["steps"] == len(client.calls) == 6
    assert client.calls[2]["response_format"] == {"type": "json_object"}
    assert "Paris is the capital" in client.calls[2]["messages"][1]["content"]


@pytest.mark.parametrize("verdict", ['{}', '{"supported":"true","issue":""}',
                                     '{"supported":true,"issue":"","extra":1}', 'not json'])
def test_invalid_critic_verdict_is_never_accepted(verdict):
    result, client, _ = execute(run_critic, [search(), answer("FINAL: Paris"), answer(verdict)],
                                runner=run_graph)
    assert result["status"] == "unverified"
    assert result["verified"] is False
    assert result["final"] == "unknown"
    assert result["candidate_answer"] == "Paris"
    assert len(client.calls) == 3


def test_critic_stops_after_two_corrections():
    responses = []
    for i in range(3):
        responses += [search(str(i), str(i)), answer("FINAL: London"),
                      answer('{"supported":false,"issue":"Source says Paris."}')]
    result, client, _ = execute(run_critic, responses, runner=run_graph)
    assert result["corrections"] == 2 and result["checks"] == 3
    assert result["status"] == "unverified" and result["final"] == "unknown"
    assert len(client.calls) == 9


def test_critic_stops_on_budget_and_keeps_candidate_for_audit():
    client = Client([search(), answer("FINAL: Paris")], limit=2)
    result = run_critic("Q", runner=run_graph, client=client, registry=Registry(), model="m")
    assert result["status"] == "budget_exceeded"
    assert result["final"] == "unknown"
    assert result["candidate_answer"] == result["initial_answer"] == "Paris"
    assert len(client.calls) == 2


def test_graph_unknown_cost_is_not_zero():
    result, _, _ = execute(run_graph, [CostUnknownError("unknown charge")])
    assert result["status"] == "cost_unknown"
    assert result["cost_usd"] is None
    assert result["known_cost_usd"] == 0


@pytest.mark.parametrize("tool_output", [None, '{"results":[]}', '{"error":"tool failed"}'])
def test_critic_uses_local_feedback_when_evidence_is_missing(tool_output):
    class FailedRegistry(Registry):
        def run(self, name, arguments):
            return tool_output

    responses = []
    for attempt in range(3):
        if tool_output is not None:
            responses.append(search(call_id=f"search-{attempt}"))
        responses.append(answer("FINAL: Paris"))
    client = Client(responses, limit=len(responses))
    result = run_critic("Q", runner=run_graph, client=client, registry=FailedRegistry(), model="m")

    assert all("response_format" not in call for call in client.calls)
    assert result["steps"] == len(client.calls) == len(responses)
    assert result["cost_usd"] == pytest.approx(.01 * len(responses))
    reviews = [event for event in result["events"] if event["type"] == "critic"]
    assert [event["reason"] for event in reviews] == ["missing_evidence"] * 3
    assert result["checks"] == len(reviews) == 3
    assert result["corrections"] == 2
    assert result["status"] == "unverified"
    assert result["verified"] is False
    assert result["final"] == "unknown"
    calls_per_attempt = len(responses) // 3
    for call in client.calls[calls_per_attempt::calls_per_attempt]:
        assert "Retrieve source evidence for the answer." in call["messages"][1]["content"]


def test_critic_checks_only_after_runner_retrieves_evidence():
    result, client, _ = execute(run_critic, [answer("FINAL: London"), search(), answer("FINAL: Paris"),
                                            answer('{"supported":true,"issue":""}')], runner=run_graph)

    assert [index for index, call in enumerate(client.calls) if "response_format" in call] == [3]
    assert "Retrieve source evidence for the answer." in client.calls[1]["messages"][1]["content"]
    evidence_request = json.loads(client.calls[-1]["messages"][1]["content"])
    assert "Paris is the capital of France." in evidence_request["evidence"]
    assert result["steps"] == len(client.calls) == 4
    assert result["cost_usd"] == pytest.approx(.04)
    assert result["checks"] == 2
    assert result["corrections"] == 1
    assert result["verified"] is True and result["final"] == "Paris"


@pytest.mark.parametrize("arguments", ['not json', '[]', '{"query":NaN}'])
def test_graph_malformed_tool_arguments_match_bare(arguments):
    request = search()
    request["tool_calls"][0]["function"]["arguments"] = arguments
    script = [request, answer("FINAL: unknown")]
    bare, bc, br = execute(run_bare, script)
    graph, gc, gr = execute(run_graph, script)
    assert gc.calls == bc.calls
    assert graph["status"] == bare["status"]
    assert gr.calls == br.calls == []
