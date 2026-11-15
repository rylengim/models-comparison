import copy

import pytest

from hw2.rag import validate_response
from hw2.reparse import reparse_record


def saved_record():
    source = {"id": "p1", "page": "Numbers", "section": "Result", "text": "The count is 42.",
              "url": "https://example.org/count"}
    raw = {"message": {"role": "assistant", "content": '```json\n{"answer":"42","quote":"The count is 42.","source_id":"p1"}\n```'},
           "finish_reason": "stop", "request_id": "receipt-1", "cost_usd": .01,
           "prompt_tokens": 10, "completion_tokens": 20}
    rag = validate_response(raw, [source])
    return {"config": "medium_rag", "id": "one", "status": "invalid_output", "final": "unknown",
            "retrieved_ids": ["p1"], "cost_usd": .01, "prompt_tokens": 10,
            "completion_tokens": 20, "steps": 1, "rag_result": rag}


def no_hit_record():
    return {"config": "cheap_rag", "id": "empty", "status": "no_hits", "steps": 0,
            "cost_usd": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "retrieved_ids": [], "rag_result": validate_response(None, [])}


def test_reparse_is_pure_and_recovers_a_saved_fenced_answer_without_any_client():
    record = saved_record()
    before = copy.deepcopy(record)
    replay = reparse_record(record)
    assert replay["replay_status"] == "reparsed"
    assert replay["previous_status"] == "invalid_output"
    assert replay["result"]["answer"] == "42"
    assert replay["result"]["usage"]["request_id"] == "receipt-1"
    assert record == before


@pytest.mark.parametrize("missing, expected", [
    ("raw_response", "unavailable_raw_response"),
    ("retrieved_context", "unavailable_context"),
])
def test_missing_inputs_cannot_be_reconstructed_from_validated_answer_or_ids(missing, expected):
    record = saved_record()
    record["rag_result"].pop(missing)
    replay = reparse_record(record)
    assert replay["replay_status"] == expected
    assert replay["result"] is None


@pytest.mark.parametrize("change", ["source_order", "missing_text", "duplicate_id", "receipt", "cost"])
def test_reparse_rejects_context_or_receipt_mismatch(change):
    record = saved_record()
    rag = record["rag_result"]
    if change == "source_order":
        record["retrieved_ids"] = ["not-the-saved-context"]
    elif change == "missing_text":
        rag["retrieved_context"][0].pop("text")
    elif change == "duplicate_id":
        rag["retrieved_context"].append(copy.deepcopy(rag["retrieved_context"][0]))
    elif change == "receipt":
        rag["raw_response"]["request_id"] = "different-call"
    else:
        record["cost_usd"] = .02
    replay = reparse_record(record)
    assert replay["replay_status"] == "inconsistent_saved_inputs"
    assert replay["result"] is None


def test_no_hit_record_does_not_require_a_nonexistent_model_response():
    record = no_hit_record()
    replay = reparse_record(record)
    assert replay["replay_status"] == "no_model_call"
    assert replay["result"]["status"] == "no_hits"


def test_empty_receipt_cannot_hide_disagreement_between_raw_and_record_cost():
    record = saved_record()
    record["rag_result"]["usage"] = {}
    record["cost_usd"] = .02
    assert reparse_record(record)["replay_status"] == "inconsistent_saved_inputs"


@pytest.mark.parametrize("location", ["usage", "raw_response"])
@pytest.mark.parametrize("field", ["request_id", "cost_usd", "prompt_tokens", "completion_tokens"])
def test_paid_replay_requires_every_billing_field(location, field):
    record = saved_record()
    record["rag_result"][location].pop(field)
    replay = reparse_record(record)
    assert replay["replay_status"] == "inconsistent_saved_inputs"
    assert replay["result"] is None


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("request_id", " "), ("request_id", None), ("request_id", 123),
    ("cost_usd", False), ("cost_usd", -0.01), ("cost_usd", "0.01"),
    ("cost_usd", float("inf")), ("cost_usd", float("nan")),
    ("prompt_tokens", True), ("prompt_tokens", -1), ("prompt_tokens", 10.0),
    ("completion_tokens", False), ("completion_tokens", "20"), ("completion_tokens", None),
])
def test_matching_billing_values_must_still_have_valid_types_and_values(field, value):
    record = saved_record()
    for location in ("usage", "raw_response"):
        record["rag_result"][location][field] = value
    if field != "request_id":
        record[field] = value
    assert reparse_record(record)["replay_status"] == "inconsistent_saved_inputs"


@pytest.mark.parametrize("field", ["cost_usd", "prompt_tokens", "completion_tokens", "steps"])
@pytest.mark.parametrize("change", ["missing", "wrong", "bool"])
def test_paid_replay_requires_consistent_top_level_totals(field, change):
    record = saved_record()
    if change == "missing":
        record.pop(field)
    else:
        record[field] = True if change == "bool" else record[field] + 1
    assert reparse_record(record)["replay_status"] == "inconsistent_saved_inputs"


@pytest.mark.parametrize("change", [
    "paid_raw", "empty_raw", "context", "receipt", "missing_usage", "null_usage",
    "top_receipt", "missing_tokens", "positive_tokens", "bool_steps", "bool_cost", "bool_tokens",
])
def test_no_hit_replay_cannot_hide_a_model_call_or_ambiguous_accounting(change):
    record = no_hit_record()
    rag = record["rag_result"]
    if change == "paid_raw":
        rag["raw_response"] = saved_record()["rag_result"]["raw_response"]
    elif change == "empty_raw":
        rag["raw_response"] = {}
    elif change == "context":
        rag["retrieved_context"] = saved_record()["rag_result"]["retrieved_context"]
    elif change == "receipt":
        rag["usage"] = saved_record()["rag_result"]["usage"]
    elif change == "missing_usage":
        rag.pop("usage")
    elif change == "null_usage":
        rag["usage"] = None
    elif change == "top_receipt":
        record["request_id"] = "paid-call"
    elif change == "missing_tokens":
        record.pop("prompt_tokens")
    elif change == "positive_tokens":
        record["completion_tokens"] = 20
    elif change == "bool_steps":
        record["steps"] = False
    elif change == "bool_cost":
        record["cost_usd"] = False
    else:
        record["prompt_tokens"] = False
    replay = reparse_record(record)
    assert replay["replay_status"] == "inconsistent_saved_inputs"
    assert replay["result"] is None


def test_legacy_no_hit_record_may_omit_raw_and_context_only():
    record = no_hit_record()
    record["rag_result"].pop("raw_response")
    record["rag_result"].pop("retrieved_context")
    assert reparse_record(record)["replay_status"] == "no_model_call"


def test_replay_accepts_the_future_public_snapshot_billing_shape():
    from hw2.benchmark_snapshot import public_record

    record = saved_record()
    extra = {"provider": "test-provider", "cost_basis": "provider_reported", "estimated_cost_usd": None}
    record["rag_result"]["usage"].update(extra)
    record["rag_result"]["raw_response"].update(extra)
    public = public_record(record)
    assert reparse_record(public)["replay_status"] == "reparsed"
