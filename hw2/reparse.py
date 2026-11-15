"""Revalidate saved RAG completions without creating a model or retrieval client."""

import math

from hw2.rag import PARSER_VERSION, SOURCE_KEYS, validate_response

BILLING = ("request_id", "cost_usd", "prompt_tokens", "completion_tokens")


def _valid_totals(record):
    cost = record.get("cost_usd")
    return ((type(cost) is int or type(cost) is float and math.isfinite(cost)) and cost >= 0
            and all(type(record.get(key)) is int and record[key] >= 0 for key in BILLING[2:]))


def _valid_receipt(receipt):
    return (isinstance(receipt, dict) and isinstance(receipt.get("request_id"), str)
            and bool(receipt["request_id"].strip()) and _valid_totals(receipt))


def _confirmed_no_call(record, rag):
    return (_valid_totals(record) and type(record.get("steps")) is int and record["steps"] == 0
            and all(record[key] == 0 for key in BILLING[1:]) and record.get("request_id") is None
            and record.get("retrieved_ids") == rag.get("retrieved_ids") == [] and rag.get("usage") == {}
            and rag.get("raw_response") is None and rag.get("retrieved_context", []) == [])


def reparse_record(record):
    """Return a separate assessment; missing raw text is never reconstructed.

    Legacy ``final``/``quote`` fields cannot recover rejected model output. They
    remain historical measurements when this function reports unavailable input.
    A saved call needs matching complete receipts and record totals; ``no_hits``
    needs explicit zero totals and no saved response, context or receipt.
    """
    if record.get("config") not in {"cheap_rag", "medium_rag"}:
        raise ValueError("Only mandatory RAG configurations support this replay.")
    assessment = {
        "config": record["config"], "id": record["id"], "parser_version": PARSER_VERSION,
        "previous_status": record["status"], "replay_status": "inconsistent_saved_inputs", "result": None,
    }
    rag = record.get("rag_result")
    if not isinstance(rag, dict):
        return assessment
    if record["status"] == "no_hits":
        if _confirmed_no_call(record, rag):
            assessment.update(replay_status="no_model_call", result=validate_response(None, []))
        return assessment
    raw = rag.get("raw_response")
    if not isinstance(raw, dict):
        assessment["replay_status"] = "unavailable_raw_response"
        return assessment
    sources = rag.get("retrieved_context")
    if sources is None:
        assessment["replay_status"] = "unavailable_context"
        return assessment
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(s, dict) or any(not isinstance(s.get(k), str) for k in SOURCE_KEYS)
                   or not s["id"] or not s["text"] for s in sources)):
        return assessment
    ids = [s["id"] for s in sources]
    if len(set(ids)) != len(ids) or ids != rag.get("retrieved_ids") or ids != record.get("retrieved_ids"):
        return assessment
    usage = rag.get("usage")
    if (not _valid_receipt(usage) or not _valid_receipt(raw) or not _valid_totals(record)
            or type(record.get("steps")) is not int or record["steps"] != 1
            or any(raw[key] != usage[key] for key in BILLING)
            or any(record[key] != usage[key] for key in BILLING[1:])
            or "request_id" in record and record["request_id"] != usage["request_id"]):
        return assessment
    assessment.update(replay_status="reparsed", result=validate_response(raw, sources))
    return assessment
