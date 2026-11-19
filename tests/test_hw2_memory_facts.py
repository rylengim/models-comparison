import pytest

from hw2.memory_facts import FACT_KEYS, MemoryEvent, current_candidates, memory_events


@pytest.mark.parametrize(("statements", "expected"), [
    (["I prefer Monday for reminders.", "I don't want Monday reminders anymore."],
     [("set", "reminder_day", "Monday"), ("reject", "reminder_day", "Monday")]),
    (["I don't want Monday reminders anymore.", "I prefer Monday for reminders."],
     [("reject", "reminder_day", "Monday"), ("set", "reminder_day", "Monday")]),
    (["I moved reminders from Monday to Friday."], [("set", "reminder_day", "Friday")]),
    (["I moved reminders OFF Monday."], [("reject", "reminder_day", "Monday")]),
    (["I no longer prefer Monday for reminders; use Friday now."],
     [("reject", "reminder_day", "Monday"), ("set", "reminder_day", "Friday")]),
    (["I prefer Monday for reminders; I want a meeting; use Friday now."],
     [("set", "reminder_day", "Monday"), ("irrelevant", None, None), ("irrelevant", None, None)]),
    (["I use miles for distances.", "I don't know how many miles the route is."],
     [("set", "distance_unit", "miles"), ("irrelevant", None, None)]),
    (["I prefer Monday for reminders.", "I don't discuss reminders in my Monday meeting."],
     [("set", "reminder_day", "Monday"), ("irrelevant", None, None)]),
    (["I measure distances in nautical miles.", "I no longer use nautical miles."],
     [("set", "distance_unit", "nautical miles"), ("reject", "distance_unit", "nautical miles")]),
    (["My project is Orion.", "My project is not Orion anymore."],
     [("set", "project_name", "Orion"), ("reject", "project_name", "Orion")]),
    (["My project is no longer called Orion.", "My project is Orion."],
     [("reject", "project_name", "Orion"), ("set", "project_name", "Orion")]),
    (['My project is "No Limits".', "My project is no longer called May."],
     [("set", "project_name", "No Limits"), ("reject", "project_name", "May")]),
    (["I might call my project Orion."], [("irrelevant", "project_name", "Orion")]),
    (["I prefer Monday for reminders.", "I might move reminders off Monday."],
     [("set", "reminder_day", "Monday"), ("irrelevant", "reminder_day", "Monday")]),
])
def test_memory_events_classify_the_relation_before_resolving_candidates(statements, expected):
    events = list(memory_events(statements))
    assert [(event.action, event.key, event.value) for event in events] == expected
    assert all(event.evidence in statements for event in events)


@pytest.mark.parametrize("key", FACT_KEYS)
def test_candidate_reducer_uses_order_and_rejects_only_the_same_full_value(key):
    first = MemoryEvent("set", key, "Alpha", "first")
    other_rejection = MemoryEvent("reject", key, "Alpha Research", "unrelated full value")
    irrelevant = MemoryEvent("irrelevant", None, None, "unrelated clause")
    assert current_candidates([first, other_rejection, irrelevant]) == {key: first}

    rejection = MemoryEvent("reject", key, "ALPHA", "revoked candidate")
    assert current_candidates([first, rejection]) == {}
    assert current_candidates([rejection, first]) == {key: first}

    replacement = MemoryEvent("set", key, "Beta", "latest")
    assert current_candidates([first, replacement, rejection]) == {key: replacement}
