"""Parse the demo's limited English settings into ordered memory events."""

import re
from dataclasses import dataclass
from typing import Literal


FACT_KEYS = ("reminder_day", "distance_unit", "project_name")
_DAY = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_OWNER = r"(?:my|our)"
_REMINDERS = rf"(?:{_OWNER}\s+)?reminders?"
_MODE = r"now|no\s+longer|do\s+not|don['’]t|never|used\s+to|may|might|could|maybe"
_PREFIX = rf"(?:I|we)\s+(?:(?P<mode>{_MODE})\s+)?"
_TAIL = r"(?:\s+(?:now|instead|from\s+now\s+on|anymore|any\s+longer))?\s*[,.!]?\s*"
_CLAUSE_BREAK = re.compile(
    r"[;\n]|\bbut\b|\band\s+(?=(?:I|we|my|our)\b)|,\s*(?=(?:not|I|we|my|our)\b)", re.I,
)


def _grammar(pattern):
    return re.compile(r"\s*" + pattern + _TAIL, re.I | re.X)


_REMINDER_ASSIGNMENT = _grammar(rf"""
    (?:
        {_PREFIX}(?:
            (?:prefer|want|need|like|choose|use|set|move[d]?|switch(?:ed)?|change[d]?|(?:re)?schedule[d]?)\s+
            (?:
                {_REMINDERS}\s+(?P<relation>on|for|to|off)\s+(?P<day_after>{_DAY})
                |{_REMINDERS}\s+from\s+(?:{_DAY})\s+to\s+(?P<day_moved>{_DAY})
                |(?P<day_before>{_DAY})\s+(?:for\s+{_REMINDERS}|reminders?)
            )
            |(?:want|need)\s+(?:you\s+)?to\s+remind\s+(?:me|us)\s+on\s+(?P<day_remind>{_DAY})
        )
        |{_OWNER}\s+reminders?(?:\s+day)?\s+(?:is|are|should\s+be|will\s+be)\s+
        (?:(?P<decl_mode>not|no\s+longer)\s+)?(?P<day_decl>{_DAY})
    )
""")
_REMINDER_CONTINUATION = _grammar(rf"""
    (?:
        (?:please\s+)?(?:use|switch\s+to|make\s+it)
        |{_PREFIX}(?:prefer|want|choose)
        |(?P<denial>not)
    )\s+(?P<day>{_DAY})
""")
_REMINDER_CANCELLATION = _grammar(rf"""
    (?:I|we)\s+(?:(?:want|need)\s+to\s+)?(?:cancel|stop|remove)\s+
    (?:{_OWNER}\s+)?(?:
        (?P<day_before>{_DAY})\s+reminders?
        |reminders?\s+(?:on|for)\s+(?P<day_after>{_DAY})
    )
""")
_REMINDER_EXCEPTION = _grammar(rf"""
    (?:I|we)\s+(?:prefer|want|need|like|choose|use)\s+(?:
        {_REMINDERS}\s+on\s+any\s+(?:weekday|day)\s+(?:except|other\s+than)\s+(?P<day_after>{_DAY})
        |any\s+(?:weekday|day)\s+(?:except|other\s+than)\s+(?P<day_before>{_DAY})\s+for\s+{_REMINDERS}
    )
""")
_UNIT = r"[\w-]+(?:\s+[\w-]+)*?"
_DISTANCE_ASSIGNMENT = _grammar(rf"""
    {_PREFIX}(?:
        (?:use|prefer|want|choose)\s+(?P<preferred_unit>{_UNIT})\s+for\s+(?:{_OWNER}\s+)?distances?
        |measure\s+(?:{_OWNER}\s+)?distances?\s+in\s+(?P<measured_unit>{_UNIT})
    )
""")
_DISTANCE_REJECTION = _grammar(rf"{_PREFIX}(?:use|prefer|want|choose)\s+(?P<unit>{_UNIT})")
_NAME = r'''(?P<name>"[^"\r\n]+"|'[^'\r\n]+'|[^\r\n]+?)'''
_PROJECT_ASSIGNMENT = _grammar(rf"""
    (?:
        {_OWNER}\s+project(?:['’]s\s+name|\s+name)?\s+is\s+
        (?:(?P<decl_mode>now|not|no\s+longer)\s+)?(?:(?:called|named)\s+)?
        |{_PREFIX}(?:
            (?:call|named)\s+{_OWNER}\s+project\s+
            |renamed\s+{_OWNER}\s+project\s+(?:to\s+)?
        )
    ){_NAME}
""")
_PROJECT_WORK = _grammar(rf"{_PREFIX}work\s+on\s+{_NAME}")


@dataclass(frozen=True)
class MemoryEvent:
    action: Literal["set", "reject", "irrelevant"]
    key: str | None
    value: str | None
    evidence: str


def _action(match):
    groups = match.groupdict()
    mode = " ".join((groups.get("mode") or groups.get("decl_mode") or "").split()).casefold()
    if mode in {"may", "might", "could", "maybe"}:
        return "irrelevant"
    if mode in {"not", "no longer", "do not", "don't", "don’t", "never", "used to"} or groups.get("denial"):
        return "reject"
    return "set"


def _event(key, match, evidence, *values, action=None):
    value = next(match[name] for name in values if match[name] is not None)
    if key == "project_name" and value[:1] in {"'", '"'} and value[-1:] == value[:1]:
        value = value[1:-1]
    return MemoryEvent(action or _action(match), key, value, evidence)


def _reminder_event(clause, evidence, continuation):
    match = _REMINDER_ASSIGNMENT.fullmatch(clause)
    if match:
        action = _action(match)
        if action == "set" and (match["relation"] or "").casefold() == "off":
            action = "reject"
        return _event("reminder_day", match, evidence, "day_after", "day_moved", "day_before",
                      "day_remind", "day_decl", action=action)
    for grammar in (_REMINDER_CANCELLATION, _REMINDER_EXCEPTION):
        if match := grammar.fullmatch(clause):
            return _event("reminder_day", match, evidence, "day_before", "day_after", action="reject")
    if continuation and (match := _REMINDER_CONTINUATION.fullmatch(clause)):
        return _event("reminder_day", match, evidence, "day")
    return None


def _distance_event(clause, evidence):
    if match := _DISTANCE_ASSIGNMENT.fullmatch(clause):
        return _event("distance_unit", match, evidence, "preferred_unit", "measured_unit")
    if (match := _DISTANCE_REJECTION.fullmatch(clause)) and _action(match) == "reject":
        return _event("distance_unit", match, evidence, "unit", action="reject")
    return None


def _project_event(clause, evidence):
    if match := _PROJECT_ASSIGNMENT.fullmatch(clause):
        return _event("project_name", match, evidence, "name")
    if (match := _PROJECT_WORK.fullmatch(clause)) and _action(match) == "reject":
        return _event("project_name", match, evidence, "name", action="reject")
    return None


def memory_events(statements):
    """An unrelated clause resets the narrowly supported reminder continuation."""
    for statement in statements:
        reminder_context = False
        for clause in _CLAUSE_BREAK.split(statement):
            event = (_reminder_event(clause, statement, reminder_context)
                     or _distance_event(clause, statement) or _project_event(clause, statement)
                     or MemoryEvent("irrelevant", None, None, statement))
            reminder_context = event.key == "reminder_day" and event.action != "irrelevant"
            yield event


def current_candidates(events):
    """Resolve this extraction's candidates, without deleting facts in the store."""
    candidates = {}
    for event in events:
        if event.action == "set":
            candidates[event.key] = event
        elif event.action == "reject":
            current = candidates.get(event.key)
            if current is not None and (
                " ".join(current.value.split()).casefold() == " ".join(event.value.split()).casefold()
            ):
                del candidates[event.key]
    return candidates
