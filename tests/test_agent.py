import copy
import json

import pytest

from freshqa.agent import run_agent
from freshqa.client import BudgetExceeded, CostUnknownError


def answer(text):
    return {"role": "assistant", "content": text}


def call(name="web_search", arguments='{"query":"capital France"}', call_id="one"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_request(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.spent_usd = 0.0
        self.budget_usd = 5.0
        self.accounting_uncertain = False

    def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            if isinstance(item, CostUnknownError):
                self.accounting_uncertain = True
            raise item
        self.spent_usd += 0.01
        return {"message": item, "cost_usd": 0.01, "prompt_tokens": 30,
                "completion_tokens": 10, "request_id": f"gen-{len(self.calls)}"}


class Registry:
    def __init__(self):
        self.calls = []

    def schemas(self, names):
        return [{"type": "function", "function": {"name": name}} for name in names]

    def run(self, name, args):
        self.calls.append((name, args))
        return json.dumps({"text": "Paris is the capital of France."})


def test_baseline_makes_exactly_one_request_with_question_only():
    client = ScriptedClient([answer("FINAL: Paris")])
    registry = Registry()
    result = run_agent("What is the capital of France?", model="m", client=client,
                       registry=registry, tool_names=[], max_steps=6)
    assert result["final"] == "Paris"
    assert result["status"] == "ok"
    assert result["steps"] == 1
    assert len(client.calls) == 1
    assert client.calls[0]["tools"] is None
    assert [m["content"] for m in client.calls[0]["messages"] if m["role"] == "user"] == [
        "What is the capital of France?"]
    assert registry.calls == []
    assert result["cost_usd"] == 0.01
    assert result["elapsed_s"] >= 0
    assert result["events"][0]["prompt_tokens"] == 30


def test_multiple_tools_form_valid_conversation_and_trace():
    client = ScriptedClient([
        tool_request(call(call_id="a"), call("open_page", '{"url":"https://example.org"}', "b")),
        answer("FINAL: Paris"),
    ])
    registry = Registry()
    result = run_agent("Question", model="m", client=client, registry=registry,
                       tool_names=["web_search", "open_page"])
    assert result["final"] == "Paris"
    assert result["cost_usd"] == pytest.approx(0.02)
    assert len(registry.calls) == 2
    messages = client.calls[1]["messages"]
    assert messages[-3]["role"] == "assistant"
    assert [m["tool_call_id"] for m in messages[-2:]] == ["a", "b"]
    assert all(m["role"] == "tool" for m in messages[-2:])
    assert len([event for event in result["events"] if event["type"] == "tool"]) == 2
    json.dumps(result, allow_nan=False)


def test_malformed_json_is_returned_as_tool_error_then_can_recover():
    client = ScriptedClient([tool_request(call(arguments="{broken")), answer("FINAL: unknown")])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert result["final"] == "unknown"
    assert registry.calls == []
    assert "invalid_arguments" in client.calls[1]["messages"][-1]["content"]


def test_repeated_call_with_reordered_arguments_forces_final_without_tools():
    client = ScriptedClient([
        tool_request(call(arguments='{"query":"q","limit":1}', call_id="a")),
        tool_request(call(arguments='{"limit":1,"query":"q"}', call_id="b")),
        answer("FINAL: Paris"),
    ])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert len(registry.calls) == 1
    assert result["steps"] == 3
    assert result["status"] == "repeated_tool_call"
    assert result["final"] == "Paris"
    assert client.calls[-1]["tools"] is None
    assert any(m["role"] == "tool" and "repeated_tool_call" in m["content"]
               for m in client.calls[-1]["messages"])


def test_step_limit_reserves_last_request_for_final_answer():
    client = ScriptedClient([tool_request(call()), answer("FINAL: Paris")])
    result = run_agent("Q", model="m", client=client, registry=Registry(),
                       tool_names=["web_search"], max_steps=2)
    assert len(client.calls) == 2
    assert client.calls[0]["tools"]
    assert client.calls[1]["tools"] is None
    assert result["steps"] == 2
    assert result["status"] == "max_steps"


def test_single_step_agent_still_gets_one_final_request_without_tools():
    client = ScriptedClient([answer("FINAL: unknown")])
    result = run_agent("Q", model="m", client=client, registry=Registry(),
                       tool_names=["web_search"], max_steps=1)
    assert result["steps"] == 1
    assert client.calls[0]["tools"] is None


def test_budget_stop_does_not_force_extra_request():
    client = ScriptedClient([tool_request(call()), BudgetExceeded("Budget exhausted")])
    result = run_agent("Q", model="m", client=client, registry=Registry(), tool_names=["web_search"])
    assert result["status"] == "budget_exceeded"
    assert result["final"] == "unknown"
    assert result["cost_usd"] == 0.01
    assert len(client.calls) == 2


def test_unknown_cost_is_never_reported_as_zero():
    client = ScriptedClient([CostUnknownError("Cost unknown")])
    result = run_agent("Q", model="m", client=client, registry=Registry(), tool_names=[])
    assert result["status"] == "cost_unknown"
    assert result["cost_usd"] is None
    assert result["known_cost_usd"] == 0


def test_unoffered_tool_is_not_executed():
    client = ScriptedClient([tool_request(call("secret_tool")), answer("FINAL: unknown")])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert registry.calls == []
    assert "tool_not_allowed" in client.calls[1]["messages"][-1]["content"]
    assert result["final"] == "unknown"


def test_many_calls_execute_at_most_four_and_receive_matching_tool_results():
    calls = [call(arguments=json.dumps({"query": str(i)}), call_id=str(i)) for i in range(7)]
    client = ScriptedClient([tool_request(*calls), answer("FINAL: unknown")])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert len(registry.calls) == 4
    assert len([m for m in client.calls[-1]["messages"] if m["role"] == "tool"]) == 7
    assert client.calls[-1]["tools"] is None
    assert result["status"] == "tool_limit"


def test_tool_exception_becomes_safe_tool_error():
    class BrokenRegistry(Registry):
        def run(self, name, args):
            raise RuntimeError("private-api-key")

    client = ScriptedClient([tool_request(call()), answer("FINAL: unknown")])
    result = run_agent("Q", model="m", client=client, registry=BrokenRegistry(), tool_names=["web_search"])
    assert "private-api-key" not in json.dumps(result)
    assert "tool_error" in client.calls[1]["messages"][-1]["content"]


def test_invalid_function_name_is_rejected_without_crashing():
    invalid = call()
    invalid["function"]["name"] = {"not": "a name"}
    client = ScriptedClient([tool_request(invalid)])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert result["status"] == "invalid_tool_calls"
    assert registry.calls == []


def test_duplicate_call_ids_stop_before_invalid_conversation_is_sent():
    client = ScriptedClient([tool_request(call(), call(arguments='{"query":"other"}'))])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=["web_search"])
    assert result["status"] == "invalid_tool_calls"
    assert registry.calls == []
    assert len(client.calls) == 1


def test_baseline_does_not_execute_hallucinated_tools():
    client = ScriptedClient([tool_request(call())])
    registry = Registry()
    result = run_agent("Q", model="m", client=client, registry=registry, tool_names=[])
    assert result["status"] == "unexpected_tool_call"
    assert result["steps"] == 1
    assert registry.calls == []


@pytest.mark.parametrize("max_steps", [0, -1, 2.5, True])
def test_invalid_step_limit_is_rejected(max_steps):
    with pytest.raises(ValueError):
        run_agent("Q", model="m", client=ScriptedClient([]), registry=Registry(),
                  tool_names=[], max_steps=max_steps)
