import json

import pytest

from freshqa.tools import ToolRegistry
from hw2.memory_agent import MemorySession, extract_facts


class Client:
    spent_usd = 0.0

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, *, model, messages, tools=None):
        self.messages.append(messages)
        return {"message": {"role": "assistant", **next(self.replies)}, "finish_reason": "stop",
                "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.0,
                "request_id": str(len(self.messages))}


class Journal:
    def __init__(self):
        self.rows = []

    def append(self, session_id, role, content):
        self.rows.append((session_id, role, content))

    def session(self, session_id):
        raise AssertionError("journal must not be read for a reply")


class Store:
    def search(self, question, limit):
        return [{"key": "reminder_day", "value": "Tuesday", "session_id": "a"}]


def test_window_and_saved_fact_reach_the_agent_during_tool_use():
    client = Client([{"content": "FINAL: Sure."},
                     {"tool_calls": [{"id": "q", "type": "function", "function": {
                         "name": "knowledge_base", "arguments": '{"query": "jobs"}'}}]},
                     {"content": "FINAL: 850"}, {"content": "FINAL: Tuesday"}])
    journal = Journal()
    registry = ToolRegistry(knowledge_search=lambda query, page=None: [])
    kwargs = dict(store=Store(), journal=journal, client=client, registry=registry, model="test")
    first = MemorySession("a", **kwargs)
    first.ask("First session marker.")
    result = first.ask("What are the job cuts?")
    assert result["tool_calls"] == ["knowledge_base"]
    assert result["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "cost_usd": 0.0}
    assert client.messages[2][1:3] == result["history"]
    second = MemorySession("b", **kwargs)
    result = second.ask("When should I get reminders?")
    assert result["history"] == [] and result["answer"] == "Tuesday"
    assert "Tuesday" in client.messages[-1][-1]["content"]
    assert "First session marker" not in str(client.messages[-1])
    assert journal.rows[-1] == ("b", "assistant", "Tuesday")


def test_extraction_rejects_a_fact_supported_only_by_the_assistant():
    bad = {"key": "reminder_day", "value": "Tuesday", "evidence": "You prefer Tuesday."}
    obj = {"reminder_day": "Tuesday", "distance_unit": "", "project_name": ""}
    client = Client([{"content": json.dumps(obj)}])
    messages = [{"role": "user", "content": "When were you going to remind me?"},
                {"role": "assistant", "content": bad["evidence"]}]
    with pytest.raises(ValueError, match="Факт"):
        extract_facts(messages, client, "test")
    assert bad["evidence"] not in client.messages[0][1]["content"]
