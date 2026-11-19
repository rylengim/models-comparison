"""Versioned final-answer eligibility, shared by measurement and offline replay."""

from .dataset import score_row

GRADING_POLICY_VERSION = "freshqa-final-eligibility-v1"
HISTORICAL_POLICY = "historical-recorded"
HARD_FAILURE_STATUSES = {"error", "cost_unknown", "budget_exceeded"}
FINAL_ANSWER_STATUSES = {"ok", "max_steps", "tool_limit", "repeated_tool_call"}


def ineligibility_reason(trace):
    """A loop limit may have a completed final; a failed/truncated call may not.

    Older traces may omit finish_reason or LLM envelopes. Their recorded status
    remains evidence. A known terminal finish_reason must be stop; any other
    reason rejects the final, including future provider-specific values.
    Earlier failed candidates do not invalidate a later completed final.
    """
    if trace.get("status") not in FINAL_ANSWER_STATUSES:
        return trace.get("status") or "missing_status"
    if trace.get("cost_usd") is None:
        return "cost_unknown"
    last_call = next((event for event in reversed(trace.get("events", []))
                      if event.get("type") == "llm"), None)
    for finish in (trace.get("finish_reason"),
                   last_call.get("finish_reason") if last_call is not None else None):
        if finish is not None and finish != "stop":
            return "output_truncated" if finish == "length" else "incomplete_finish_reason"
    if last_call is not None:
        if last_call.get("error"):
            return "llm_error"
        message = last_call.get("message")
        if message is not None:
            if message.get("tool_calls"):
                return "unfinished_tool_call"
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                return "empty_response"
    final = trace.get("final")
    if not isinstance(final, str) or not final.strip():
        return "empty_response"
    return None


def score_trace(row, trace):
    """Check whether the final is admissible before running the text metric."""
    return ineligibility_reason(trace) is None and score_row(row, trace["final"])
