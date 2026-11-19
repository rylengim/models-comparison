import json
from copy import deepcopy

import pytest

from freshqa.tools import ToolRegistry
from freshqa.client import BudgetExceeded
from hw2.memory_agent import (
    EXTRACT_RULES, FactExtractionError, MemorySession, candidate_statements, extract_facts,
)


@pytest.fixture(autouse=True)
def forbid_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Memory regression tests must never send HTTP requests")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)


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


class MutableStore:
    def __init__(self, user_id="alice", backend=None):
        self.user_id = user_id
        self.backend = backend if backend is not None else {}
        self.writes = 0

    def upsert(self, session_id, facts):
        self.writes += 1
        for fact in facts:
            self.backend[self.user_id, fact["key"]] = {**fact, "session_id": session_id}

    def all(self):
        return [dict(value) for (user, _), value in self.backend.items() if user == self.user_id]

    def search(self, question, limit):
        return self.all()[:limit]


class PreferenceClient:
    """Deterministic completions: answers use only the saved preference card."""
    def __init__(self):
        self.spent_usd = 0.0
        self.messages = []
        self.extractions = 0
        self.invalid_extraction = False

    def complete(self, *, model, messages, tools=None):
        self.messages.append(messages)
        self.spent_usd += 0.01
        if messages[0]["content"] == EXTRACT_RULES:
            self.extractions += 1
            text = messages[-1]["content"]
            day = "Friday" if "Friday" in text else "Tuesday" if "Tuesday" in text else ""
            content = "not json" if self.invalid_extraction else json.dumps({
                "reminder_day": day, "distance_unit": "", "project_name": ""})
        else:
            text = messages[-1]["content"]
            day = "Friday" if "reminder day: Friday" in text else "Tuesday" if "reminder day: Tuesday" in text else "unknown"
            content = "FINAL: Train connections" if "Suggest a heading" in text else "FINAL: " + day
        return {"message": {"role": "assistant", "content": content}, "finish_reason": "stop",
                "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01,
                "request_id": f"call-{len(self.messages)}"}


def memory_session(store, client, session_id="a", **kwargs):
    return MemorySession(session_id, store=store, journal=Journal(), client=client,
                         registry=ToolRegistry(knowledge_search=lambda query, page=None: []),
                         model="test", **kwargs)


def test_correction_is_saved_before_answer_and_survives_window_eviction():
    store, client = MutableStore(), PreferenceClient()
    session = memory_session(store, client)
    assert session.ask("I prefer Tuesday for reminders.")["answer"] == "Tuesday"
    corrected = session.ask("I now prefer Friday for reminders.")
    assert corrected["answer"] == "Friday"
    assert corrected["facts"][0]["value"] == "Friday"
    session.ask("Suggest a heading for train connections.")
    recalled = session.ask("Which reminder day should you use?")
    assert "Friday" not in str(recalled["history"])
    assert recalled["answer"] == "Friday"
    assert client.extractions == 2
    assert store.all() == [{"key": "reminder_day", "value": "Friday", "session_id": "a"}]


@pytest.mark.parametrize("statement", [
    "I no longer prefer Monday for reminders; use Tuesday now.",
    "I prefer reminders on Tuesday, not Monday.",
    "I moved reminders from Monday to Friday.",
    "I moved reminders off Monday.",
    "I moved reminders OFF Monday.",
    "I want to cancel my Monday reminders.",
    "I might move reminders to Monday.",
    "I held a Monday meeting about a book.",
    "I no longer want Friday reminders; I want a Monday meeting.",
    "I want reminders on any weekday except Monday.",
    "I prefer any day other than Monday for reminders.",
    "I prefer Monday for reminders, but I don't want Monday anymore.",
    "I want a Monday meeting about reminders",
    "I want a Monday meeting about reminders.",
    "I prefer reading the Monday news about reminder apps.",
    "I prefer reading Monday reminders.",
    "I want a Friday meeting about reminders; use Monday now.",
    "I prefer Monday reminders; I prefer any day other than Monday for reminders.",
])
def test_unsupported_reminder_day_does_not_overwrite_saved_preference(statement):
    store = MutableStore()
    store.upsert("previous", [{"key": "reminder_day", "value": "Tuesday"}])
    stale_extraction = {"reminder_day": "Monday", "distance_unit": "", "project_name": ""}
    client = Client([{"content": json.dumps(stale_extraction)}, {"content": "FINAL: Monday"}])
    session = memory_session(store, client)

    with pytest.raises(FactExtractionError):
        session.ask(statement)

    assert store.writes == len(client.messages) == 1
    assert store.all() == [{"key": "reminder_day", "value": "Tuesday", "session_id": "previous"}]
    summary = session.finish()
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["extracted"] == []


@pytest.mark.parametrize("statement", [
    "I want reminders on Monday.",
    "I want Monday reminders.",
    "My reminder day is Monday.",
    "I want you to remind me on Monday.",
])
def test_explicit_reminder_assignment_is_saved_before_answer_and_recalled(statement):
    store = MutableStore()
    store.upsert("previous", [{"key": "reminder_day", "value": "Tuesday"}])
    extraction = {"reminder_day": "Monday", "distance_unit": "", "project_name": ""}
    client = Client([{"content": json.dumps(extraction)}, {"content": "FINAL: Monday"},
                     {"content": "FINAL: Monday"}])
    session = memory_session(store, client, window_pairs=0)

    result = session.ask(statement)
    recalled = session.ask("Which day should I get reminders?")

    assert result["memory_update"]["status"] == "persisted"
    assert result["memory_update"]["extracted"] == [
        {"key": "reminder_day", "value": "Monday", "evidence": statement}]
    assert result["answer"] == recalled["answer"] == "Monday"
    assert recalled["history"] == []
    assert recalled["facts"] == store.all() == [
        {"key": "reminder_day", "value": "Monday", "session_id": "a"}]
    assert "reminder day: Monday" in client.messages[1][-1]["content"]
    assert "reminder day: Monday" in client.messages[2][-1]["content"]
    assert store.writes == 2


@pytest.mark.parametrize("statement", [
    "I no longer use miles. I now measure distances in kilometers.",
    "I don't use miles for distances anymore.",
    "I use miles for distances. I now measure distances in kilometers.",
])
def test_rejected_or_superseded_distance_unit_does_not_overwrite_saved_preference(statement):
    store = MutableStore()
    store.upsert("previous", [{"key": "distance_unit", "value": "kilometers"}])
    stale_extraction = {"reminder_day": "", "distance_unit": "miles", "project_name": ""}
    client = Client([{"content": json.dumps(stale_extraction)}, {"content": "FINAL: miles"}])
    session = memory_session(store, client)

    with pytest.raises(FactExtractionError):
        session.ask(statement)

    assert store.writes == len(client.messages) == 1
    assert store.all() == [{"key": "distance_unit", "value": "kilometers", "session_id": "previous"}]
    summary = session.finish()
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["memory_updates"][0]["extraction_usage"]["request_id"] == "1"
    assert summary["extracted"] == []


@pytest.mark.parametrize(("statement", "unit", "evidence"), [
    ("I now measure distances in miles.", "miles", "I now measure distances in miles."),
    ("I use miles for distances.", "miles", "I use miles for distances."),
    ("I no longer use kilometers. I now prefer miles for distances.",
     "miles", "I now prefer miles for distances."),
    ("I measure distances in nautical miles.", "nautical miles", "I measure distances in nautical miles."),
])
def test_explicit_distance_preference_is_saved_before_answer_and_recalled(statement, unit, evidence):
    store = MutableStore()
    store.upsert("previous", [{"key": "distance_unit", "value": "kilometers"}])
    extraction = {"reminder_day": "", "distance_unit": unit, "project_name": ""}
    client = Client([{"content": json.dumps(extraction)}, {"content": f"FINAL: {unit}"},
                     {"content": f"FINAL: {unit}"}])
    session = memory_session(store, client, window_pairs=0)

    result = session.ask(statement)
    recalled = session.ask("Which distance unit should I use?")

    assert result["memory_update"]["status"] == "persisted"
    assert result["memory_update"]["extracted"] == [
        {"key": "distance_unit", "value": unit, "evidence": evidence}]
    assert result["answer"] == recalled["answer"] == unit
    assert recalled["history"] == []
    assert recalled["facts"] == store.all() == [
        {"key": "distance_unit", "value": unit, "session_id": "a"}]
    assert f"distance unit: {unit}" in client.messages[1][-1]["content"]
    assert f"distance unit: {unit}" in client.messages[2][-1]["content"]
    assert store.writes == 2


@pytest.mark.parametrize("statement", [
    "My project is Atlas. I no longer work on Orion.",
    "I no longer work on Orion.",
    "My project is Orion. My project is Atlas.",
    "My colleague's project is Orion.",
    "My project is not Orion.",
    "I might call my project Orion.",
    "My project is Orion. I no longer work on Orion.",
    "My project is Orion. My project is not Orion.",
    "My project is Orion. I don't call my project Orion anymore.",
    "My project is Orion. I no longer work on ORION.",
    "My project is Orion Research.",
])
def test_rejected_or_superseded_project_does_not_overwrite_saved_name(statement):
    store = MutableStore()
    store.upsert("previous", [{"key": "project_name", "value": "Atlas"}])
    stale_extraction = {"reminder_day": "", "distance_unit": "", "project_name": "Orion"}
    client = Client([{"content": json.dumps(stale_extraction)}, {"content": "FINAL: Orion"}])
    session = memory_session(store, client)

    with pytest.raises(FactExtractionError):
        session.ask(statement)

    assert store.writes == len(client.messages) == 1
    assert store.all() == [{"key": "project_name", "value": "Atlas", "session_id": "previous"}]
    summary = session.finish()
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["extraction_receipts"][0]["request_id"] == "1"
    assert summary["extracted"] == []


@pytest.mark.parametrize(("statement", "name", "evidence"), [
    ("My project is Orion.", "Orion", "My project is Orion."),
    ("I named my project Orion.", "Orion", "I named my project Orion."),
    ("Our project is now called Orion.", "Orion", "Our project is now called Orion."),
    ("I renamed my project to Orion.", "Orion", "I renamed my project to Orion."),
    ("My project is Orion Research.", "Orion Research", "My project is Orion Research."),
    ('My project is "Orion Research".', "Orion Research", 'My project is "Orion Research".'),
    ("My project is May.", "May", "My project is May."),
    ("My project is No Limits.", "No Limits", "My project is No Limits."),
    ("My project is Atlas. I no longer work on Orion.", "Atlas", "My project is Atlas."),
    ("I no longer work on Orion. My project is Orion Research.",
     "Orion Research", "My project is Orion Research."),
    ("My project is Orion. My project is not Orion Research.", "Orion", "My project is Orion."),
    ("My project is Orion. I no longer work on Orion Research.", "Orion", "My project is Orion."),
])
def test_explicit_project_name_is_saved_before_answer_and_recalled(statement, name, evidence):
    store = MutableStore()
    store.upsert("previous", [{"key": "project_name", "value": "Atlas"}])
    extraction = {"reminder_day": "", "distance_unit": "", "project_name": name}
    client = Client([{"content": json.dumps(extraction)}, {"content": f"FINAL: {name}"},
                     {"content": f"FINAL: {name}"}])
    session = memory_session(store, client, window_pairs=0)

    result = session.ask(statement)
    recalled = session.ask("What is my project called?")

    assert result["memory_update"]["status"] == "persisted"
    assert result["memory_update"]["extracted"] == [
        {"key": "project_name", "value": name, "evidence": evidence}]
    assert result["answer"] == recalled["answer"] == name
    assert recalled["history"] == []
    assert recalled["facts"] == store.all() == [
        {"key": "project_name", "value": name, "session_id": "a"}]
    assert f"project name: {name}" in client.messages[1][-1]["content"]
    assert f"project name: {name}" in client.messages[2][-1]["content"]
    assert store.writes == 2


def test_all_three_explicit_memory_fields_are_persisted_in_one_update():
    store = MutableStore()
    store.upsert("previous", [
        {"key": "reminder_day", "value": "Tuesday"},
        {"key": "distance_unit", "value": "kilometers"},
        {"key": "project_name", "value": "Atlas"},
    ])
    extraction = {"reminder_day": "Friday", "distance_unit": "miles", "project_name": "Orion"}
    client = Client([{"content": json.dumps(extraction)}, {"content": "FINAL: Friday"}])
    session = memory_session(store, client)

    result = session.ask("I prefer Friday for reminders. I use miles for distances. My project is Orion.")

    assert result["memory_update"]["status"] == "persisted"
    assert result["memory_update"]["extracted"] == [
        {"key": "reminder_day", "value": "Friday", "evidence": "I prefer Friday for reminders."},
        {"key": "distance_unit", "value": "miles", "evidence": "I use miles for distances."},
        {"key": "project_name", "value": "Orion", "evidence": "My project is Orion."},
    ]
    assert store.all() == [
        {"key": "reminder_day", "value": "Friday", "session_id": "a"},
        {"key": "distance_unit", "value": "miles", "session_id": "a"},
        {"key": "project_name", "value": "Orion", "session_id": "a"},
    ]
    assert store.writes == 2
    assert all(f"{fact['key'].replace('_', ' ')}: {fact['value']}" in client.messages[1][-1]["content"]
               for fact in result["facts"])


@pytest.mark.parametrize(("key", "stale_value"), [
    ("reminder_day", "Tuesday"),
    ("distance_unit", "kilometers"),
    ("project_name", "Atlas"),
])
def test_invalid_memory_field_prevents_partial_update_of_all_three_fields(key, stale_value):
    store = MutableStore()
    store.upsert("previous", [
        {"key": "reminder_day", "value": "Tuesday"},
        {"key": "distance_unit", "value": "kilometers"},
        {"key": "project_name", "value": "Atlas"},
    ])
    saved = store.all()
    extraction = {"reminder_day": "Friday", "distance_unit": "miles", "project_name": "Orion"}
    extraction[key] = stale_value
    client = Client([{"content": json.dumps(extraction)}, {"content": "FINAL: Friday"}])
    session = memory_session(store, client)

    with pytest.raises(FactExtractionError):
        session.ask("I no longer prefer Tuesday for reminders; use Friday now. "
                    "I no longer use kilometers. I now measure distances in miles. "
                    "I no longer work on Atlas. My project is Orion.")

    assert store.all() == saved
    assert store.writes == len(client.messages) == 1
    summary = session.finish()
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["extraction_receipts"][0]["request_id"] == "1"
    assert summary["extracted"] == []


@pytest.mark.parametrize("statement", [
    "I moved reminders to Friday.",
    "I moved reminders from Monday to Friday.",
    "I no longer prefer Monday for reminders; use Friday now.",
    "I don't want reminders on Monday anymore, I prefer Friday.",
])
def test_explicit_reschedule_updates_saved_preference(statement):
    store = MutableStore()
    store.upsert("previous", [{"key": "reminder_day", "value": "Tuesday"}])
    session = memory_session(store, PreferenceClient())

    result = session.ask(statement)

    assert result["memory_update"]["status"] == "persisted"
    assert result["facts"][0]["value"] == result["answer"] == "Friday"
    assert store.writes == 2
    assert store.all() == [{"key": "reminder_day", "value": "Friday", "session_id": "a"}]


def test_finish_is_a_zero_call_idempotent_summary_and_old_session_cannot_overwrite_newer_facts():
    store, client = MutableStore(), PreferenceClient()
    first = memory_session(store, client, "first")
    first.ask("I prefer Tuesday for reminders.")
    second = memory_session(store, client, "second", window_pairs=0)
    second.ask("I now prefer Friday for reminders.")
    calls, writes = len(client.messages), store.writes
    summary = first.finish()
    assert first.finish() == summary
    assert len(client.messages) == calls and store.writes == writes
    assert summary["extraction_usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01}
    assert summary["extraction_receipts"][0]["request_id"] == "call-1"
    assert store.all()[0]["value"] == "Friday"
    third = memory_session(store, client, "third")
    assert third.ask("Which day should I use?")["answer"] == "Friday"
    with pytest.raises(RuntimeError, match="finished"):
        first.ask("I prefer Tuesday for reminders.")


def test_memory_update_and_answer_costs_are_separate_and_extractions_count_once():
    store, client = MutableStore(), PreferenceClient()
    session = memory_session(store, client)
    turns = [session.ask("I prefer Tuesday for reminders."),
             session.ask("I now prefer Friday for reminders."),
             session.ask("Which day should I use?")]
    summary = session.finish()
    assert summary["extraction_usage"]["cost_usd"] == pytest.approx(0.02)
    assert len(summary["extraction_receipts"]) == 2
    total = sum(t["usage"]["cost_usd"] for t in turns) + summary["extraction_usage"]["cost_usd"]
    assert total == pytest.approx(client.spent_usd)
    assert turns[-1]["memory_update"]["status"] == "skipped"
    assert client.extractions == 2


def test_invalid_extraction_does_not_answer_or_claim_a_persisted_fact_and_retains_charge():
    store, client = MutableStore(), PreferenceClient()
    store.upsert("previous", [{"key": "reminder_day", "value": "Tuesday"}])
    client.invalid_extraction = True
    session = memory_session(store, client)
    with pytest.raises(ValueError):
        session.ask("I now prefer Friday for reminders.")
    assert len(client.messages) == 1 and client.extractions == 1
    assert store.all()[0]["value"] == "Tuesday"
    summary = session.finish()
    assert summary["extraction_usage"]["cost_usd"] == pytest.approx(0.01)
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["extracted"] == []


def test_failed_persistence_stops_answer_and_keeps_extraction_receipt():
    class BrokenStore(MutableStore):
        def upsert(self, session_id, facts):
            raise OSError("local store unavailable")

    store, client = BrokenStore(), PreferenceClient()
    session = memory_session(store, client)
    with pytest.raises(OSError, match="unavailable"):
        session.ask("I now prefer Friday for reminders.")
    assert len(client.messages) == 1
    summary = session.finish()
    assert summary["extraction_usage"]["cost_usd"] == pytest.approx(0.01)
    assert summary["memory_updates"][0]["status"] == "persistence_failed"
    assert summary["extracted"] == []


def test_updates_stay_scoped_to_the_user_store():
    backend, client = {}, PreferenceClient()
    alice, bob = MutableStore("alice", backend), MutableStore("bob", backend)
    memory_session(alice, client).ask("I prefer Friday for reminders.")
    assert bob.all() == []
    assert memory_session(bob, client).ask("Which day should I use?")["answer"] == "unknown"


def test_explicit_turn_id_replays_the_result_without_another_update_or_charge():
    store, client = MutableStore(), PreferenceClient()
    session = memory_session(store, client)
    first = session.ask("I prefer Friday for reminders.", turn_id="correction-1")
    calls, writes = len(client.messages), store.writes
    assert session.ask("I prefer Friday for reminders.", turn_id="correction-1") == first
    assert len(client.messages) == calls and store.writes == writes
    assert len(session.journal.rows) == 2
    session.finish()
    assert session.ask("I prefer Friday for reminders.", turn_id="correction-1") == first
    assert len(client.messages) == calls and store.writes == writes
    with pytest.raises(RuntimeError, match="finished"):
        session.ask("I prefer Friday for reminders.", turn_id="new-turn")
    with pytest.raises(ValueError, match="different question"):
        session.ask("I prefer Tuesday for reminders.", turn_id="correction-1")


def test_failed_turn_id_does_not_repeat_paid_extraction_automatically():
    store, client = MutableStore(), PreferenceClient()
    client.invalid_extraction = True
    session = memory_session(store, client)
    with pytest.raises(ValueError):
        session.ask("I prefer Friday for reminders.", turn_id="correction-1")
    with pytest.raises(RuntimeError, match="failed"):
        session.ask("I prefer Friday for reminders.", turn_id="correction-1")
    assert len(client.messages) == 1


def test_unreported_extraction_failure_is_not_counted_as_zero_cost():
    class InterruptedClient(PreferenceClient):
        def complete(self, **kwargs):
            raise RuntimeError("interrupted extraction")

    session = memory_session(MutableStore(), InterruptedClient())
    with pytest.raises(RuntimeError, match="interrupted"):
        session.ask("I prefer Friday for reminders.")
    summary = session.finish()
    assert summary["extraction_usage"]["cost_usd"] is None
    assert summary["memory_updates"][0]["status"] == "extraction_failed"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["extraction", "persistence"])
def test_interrupted_turn_keeps_receipt_state_and_blocks_automatic_retry(interruption, stage):
    class InterruptedClient(PreferenceClient):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            if stage == "extraction":
                raise interruption("interrupted extraction")
            return response

    class InterruptedStore(MutableStore):
        def upsert(self, session_id, facts):
            raise interruption("interrupted persistence")

    store = InterruptedStore() if stage == "persistence" else MutableStore()
    client = InterruptedClient()
    session = memory_session(store, client)
    with pytest.raises(interruption):
        session.ask("I prefer Friday for reminders.", turn_id="interrupted-1")
    with pytest.raises(RuntimeError, match="failed"):
        session.ask("I prefer Friday for reminders.", turn_id="interrupted-1")
    assert len(client.messages) == 1 and client.spent_usd == pytest.approx(0.01)
    summary = session.finish()
    assert summary["extracted"] == []
    assert summary["memory_updates"][0]["status"] == f"{stage}_failed"
    if stage == "extraction":
        assert summary["extraction_usage"]["cost_usd"] is None
    else:
        assert summary["extraction_usage"]["cost_usd"] == pytest.approx(0.01)
        assert summary["extraction_receipts"][0]["request_id"] == "call-1"


def test_local_budget_rejection_has_no_provider_receipt_or_unknown_charge():
    class BudgetClient(PreferenceClient):
        def complete(self, **kwargs):
            raise BudgetExceeded("No request was sent")

    session = memory_session(MutableStore(), BudgetClient())
    with pytest.raises(BudgetExceeded):
        session.ask("I prefer Friday for reminders.")
    summary = session.finish()
    assert summary["extraction_usage"]["cost_usd"] == 0
    assert summary["extraction_receipts"] == []
    assert summary["memory_updates"][0]["status"] == "budget_exceeded"


def test_new_demo_validates_recall_after_correction_and_accounts_for_all_extractions(tmp_path):
    from hw2 import demo_memory

    store, client, sessions = MutableStore(), PreferenceClient(), []
    for session_id, questions in demo_memory.SESSIONS:
        session = memory_session(store, client, session_id)
        turns = [{**session.ask(question), "full_usage": {
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}} for question in questions]
        sessions.append({"session_id": session_id, "turns": turns, **session.finish()})
    demo = {"memory_policy": "update_before_answer_v2", "model": "test", "price_basis": "synthetic",
            "sessions": sessions, "final_lookup": store.search("Which day?", limit=2)}
    demo_memory.validate_demo(demo)
    assert demo["sessions"][1]["turns"][-1]["answer"] == "Friday"
    report = demo_memory.compare(demo, tmp_path / "report")
    assert report["totals"]["full"]["cost_usd"] + report["totals"]["window"]["cost_usd"] == pytest.approx(client.spent_usd)
    broken = deepcopy(demo)
    broken["sessions"][1]["turns"][-1]["answer"] = "Tuesday"
    with pytest.raises(ValueError, match="Friday"):
        demo_memory.validate_demo(broken)
    # Accepted artifacts without the new marker retain their historical checks.
    del broken["memory_policy"]
    demo_memory.validate_demo(broken)
    broken["memory_policy"] = "invented_policy"
    with pytest.raises(ValueError, match="policy"):
        demo_memory.validate_demo(broken)


# Each row varies the value as well as the field: the transition contract must
# not depend on a list of known project names or on a single weekday/unit.
_MEMORY_TRANSITION_ROWS = [
    ("reminder_day", "Monday", "Tuesday", "Friday",
     "I prefer Monday for reminders.", "I don't want Monday reminders anymore.",
     "I don’t discuss reminders in my Monday meeting.", "I prefer Friday for reminders."),
    ("reminder_day", "Thursday", "Sunday", "Wednesday",
     "My reminder day is Thursday.", "I no longer want Thursday reminders.",
     "I don't discuss reminders in my Thursday meeting.", "My reminder day is Wednesday."),
    ("distance_unit", "miles", "kilometers", "meters",
     "I use miles for distances.", "I don't use miles for distances anymore.",
     "I don’t know how many miles the route is.", "I use meters for distances."),
    ("distance_unit", "nautical miles", "meters", "kilometers",
     "I measure distances in nautical miles.", "I no longer use nautical miles.",
     "I don't know how many nautical miles the route is.", "I measure distances in kilometers."),
    ("project_name", "Orion", "Atlas", "Lyra",
     "My project is Orion.", "My project is not Orion anymore.",
     "I don't discuss Orion in my project meeting.", "My project is Lyra."),
    ("project_name", "Orion Research", "Atlas", "Orion",
     'My project is "Orion Research".', "My project is no longer called Orion Research.",
     "I don't discuss Orion Research in my project meeting.", "My project is Orion."),
    ("project_name", "May", "Atlas", "Autumn",
     "My project is May.", "My project is no longer called May.",
     "I don't discuss May in my project meeting.", "My project is Autumn."),
    ("project_name", "No Limits", "Atlas", "New Horizons",
     "My project is No Limits.", "My project is not No Limits anymore.",
     "I don't discuss No Limits in my project meeting.", "My project is New Horizons."),
]


def _memory_transition_cases():
    for key, value, old, other, assignment, rejection, irrelevant, replacement in _MEMORY_TRANSITION_ROWS:
        for rule, statement, proposed, valid, evidence in [
            ("set-reject", f"{assignment} {rejection}", value, False, None),
            ("reject-set", f"{rejection} {assignment}", value, True, assignment),
            ("set-irrelevant", f"{assignment} {irrelevant}", value, True, assignment),
            ("irrelevant-set", f"{irrelevant} {assignment}", value, True, assignment),
            ("set-new-set-stale", f"{assignment} {replacement}", value, False, None),
            ("set-new-set-current", f"{assignment} {replacement}", other, True, replacement),
        ]:
            yield pytest.param(key, old, statement, proposed, valid, evidence,
                               id=f"{key}-{value}-{rule}")
    # Exercise this exact rejection without another invalid field hiding it.
    assignment = "My project is Orion."
    rejection = "My project is no longer called Orion."
    yield pytest.param("project_name", "Atlas", f"{assignment} {rejection}", "Orion", False, None,
                       id="project_name-Orion-set-no-longer-called")
    yield pytest.param("project_name", "Atlas", f"{rejection} {assignment}", "Orion", True, assignment,
                       id="project_name-Orion-no-longer-called-set")
    yield pytest.param("reminder_day", "Tuesday",
                       "I prefer Monday for reminders. I might move reminders off Monday.",
                       "Monday", True, "I prefer Monday for reminders.", id="reminder_day-set-hypothetical-off")


class SavedFactClient(Client):
    """Answers depend on the store and supplied context at the answer boundary."""
    def __init__(self, store, extraction):
        self.store = store
        self.answer_snapshots = []
        self.extraction = extraction
        self.spent_usd = 0.0
        super().__init__([])

    def complete(self, *, model, messages, tools=None):
        if messages[0]["content"] == EXTRACT_RULES:
            content = json.dumps(self.extraction)
        else:
            saved = self.store.all()
            self.answer_snapshots.append(deepcopy(saved))
            assert len(saved) == 1
            fact = saved[0]
            assert f"{fact['key'].replace('_', ' ')}: {fact['value']}" in messages[-1]["content"]
            content = "FINAL: " + fact["value"]
        self.messages.append(messages)
        self.spent_usd += 0.01
        return {"message": {"role": "assistant", "content": content}, "finish_reason": "stop",
                "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01,
                "request_id": str(len(self.messages))}


@pytest.mark.parametrize(("key", "old", "statement", "value", "valid", "evidence"),
                         list(_memory_transition_cases()))
def test_memory_event_order_preserves_or_replaces_saved_state(key, old, statement, value, valid, evidence):
    store = MutableStore()
    store.upsert("previous", [{"key": key, "value": old}])
    before = store.all()
    extraction = dict.fromkeys(("reminder_day", "distance_unit", "project_name"), "")
    extraction[key] = value
    client = SavedFactClient(store, extraction)
    session = memory_session(store, client, window_pairs=0)

    if not valid:
        with pytest.raises(FactExtractionError):
            session.ask(statement)
        assert store.all() == before
        assert store.writes == len(client.messages) == 1
        assert client.answer_snapshots == []
        summary = session.finish()
        assert summary["memory_updates"][0]["status"] == "invalid_extraction"
        assert summary["extracted"] == []
    else:
        updated = session.ask(statement)
        recalled = session.ask("What value is saved for this preference?")
        saved = [{"key": key, "value": value, "session_id": "a"}]
        expected_fact = {"key": key, "value": value, "evidence": evidence}
        assert updated["memory_update"]["status"] == "persisted"
        assert updated["memory_update"]["extracted"] == [expected_fact]
        assert updated["answer"] == recalled["answer"] == value
        assert recalled["history"] == []
        assert recalled["memory_update"]["status"] == "skipped"
        assert updated["facts"] == recalled["facts"] == store.all() == saved
        assert client.answer_snapshots == [saved, saved]
        assert store.writes == 2 and len(client.messages) == 3
        summary = session.finish()
        assert summary["extracted"] == [expected_fact]
    assert summary["extraction_receipts"] == [
        {"request_id": "1", "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01}]
    assert summary["extraction_usage"] == {
        "prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01}


def test_retracted_project_blocks_other_valid_fields_in_the_same_batch():
    store = MutableStore()
    store.upsert("previous", [
        {"key": "reminder_day", "value": "Tuesday"},
        {"key": "distance_unit", "value": "kilometers"},
        {"key": "project_name", "value": "Atlas"},
    ])
    before = store.all()
    extraction = {"reminder_day": "Monday", "distance_unit": "miles", "project_name": "Orion"}
    client = Client([{"content": json.dumps(extraction)}, {"content": "FINAL: Orion"}])
    session = memory_session(store, client, window_pairs=0)

    with pytest.raises(FactExtractionError):
        session.ask("I prefer Monday for reminders. I don’t discuss reminders in my Monday meeting. "
                    "I use miles for distances. I don’t know how many miles the route is. "
                    "My project is Orion. My project is no longer called Orion.")

    assert store.all() == before
    assert store.writes == len(client.messages) == 1
    summary = session.finish()
    assert summary["memory_updates"][0]["status"] == "invalid_extraction"
    assert summary["extracted"] == []
    assert summary["extraction_receipts"][0]["request_id"] == "1"


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("name", [
    "Orion but Better",
    "Orion and Our Research",
    "Orion, not Atlas",
    "Orion; Research",
    "Orion. Research",
    "Orion? Research",
    "Orion. I prefer Monday for reminders. Research",
])
def test_quoted_project_name_is_saved_whole_before_answer_and_recalled(quote, name):
    statement = f"My project is {quote}{name}{quote}."
    store = MutableStore()
    store.upsert("previous", [{"key": "project_name", "value": "Atlas"}])
    client = SavedFactClient(store, {
        "reminder_day": "", "distance_unit": "", "project_name": name})
    session = memory_session(store, client, window_pairs=0)

    updated = session.ask(statement)
    recalled = session.ask("What is my project called?")

    expected_fact = {"key": "project_name", "value": name, "evidence": statement}
    saved = [{"key": "project_name", "value": name, "session_id": "a"}]
    assert updated["memory_update"]["extracted"] == [expected_fact]
    assert updated["memory_update"]["status"] == "persisted"
    assert updated["answer"] == recalled["answer"] == name
    assert recalled["history"] == []
    assert recalled["memory_update"]["status"] == "skipped"
    assert store.all() == updated["facts"] == recalled["facts"] == saved
    assert client.answer_snapshots == [saved, saved]
    assert store.writes == 2 and len(client.messages) == 3
    assert session.finish()["extracted"] == [expected_fact]


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("name", ["Orion. Research", "Orion? Research", "Orion! Research"])
def test_candidate_statements_preserve_quoted_sentence_boundaries_and_exact_evidence(quote, name):
    statement = f"My project is {quote}{name}{quote}."
    correction = "My project is Lyra."

    assert candidate_statements([
        {"role": "user", "content": f"{statement} {correction}"},
        {"role": "assistant", "content": "My project is Atlas."},
    ]) == [statement, correction]


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("separator", ["; ", ". "])
def test_assignment_text_inside_quoted_name_cannot_update_another_field(quote, separator):
    name = separator.join(["Orion", "I prefer Monday for reminders", "Research"])
    statement = f"My project is {quote}{name}{quote}."
    store = MutableStore()
    store.upsert("previous", [{"key": "reminder_day", "value": "Tuesday"}])
    saved = store.all()
    client = SavedFactClient(store, {
        "reminder_day": "Monday", "distance_unit": "", "project_name": ""})
    session = memory_session(store, client, window_pairs=0)

    with pytest.raises(FactExtractionError):
        session.ask(statement)

    assert store.all() == saved
    assert store.writes == len(client.messages) == 1
    assert client.answer_snapshots == []
    assert session.finish()["memory_updates"][0]["status"] == "invalid_extraction"


@pytest.mark.parametrize("quote", ['"', "'"])
def test_unclosed_quote_does_not_persist_a_partial_project_name(quote):
    statement = f"My project is {quote}Orion but Better."
    store = MutableStore()
    store.upsert("previous", [{"key": "project_name", "value": "Atlas"}])
    saved = store.all()
    client = SavedFactClient(store, {
        "reminder_day": "", "distance_unit": "", "project_name": quote + "Orion"})
    session = memory_session(store, client)

    with pytest.raises(FactExtractionError):
        session.ask(statement)

    assert store.all() == saved
    assert store.writes == len(client.messages) == 1
    assert client.answer_snapshots == []
    assert session.finish()["extracted"] == []
