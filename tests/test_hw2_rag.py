import json

import pytest

from hw2.rag import PARSER_VERSION, rag_answer, response_snapshot, validate_response


PASSAGE = {
    "id": "canada:jan", "page": "2026 in Canada", "section": "Events / January",
    "url": "https://example.org/canada", "text": "January 13 – Statistics Canada announces 850 job cuts.",
}


class StubKnowledgeBase:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, page=None):
        self.calls.append((query, page))
        return self.hits


class StubClient:
    """Fixed local completion to test formatting and evidence checks, not model quality."""
    def __init__(self, content):
        self.content = content
        self.messages = []

    def complete(self, *, model, messages):
        self.messages.append(messages)
        return {"message": {"role": "assistant", "content": self.content},
                "finish_reason": "stop", "prompt_tokens": 12, "completion_tokens": 3}


def test_rag_retrieves_first_and_uses_source_metadata_for_a_supported_answer():
    question = "How many cuts did Statistics Canada announce?"
    kb = StubKnowledgeBase([PASSAGE])
    quote = "Statistics Canada announces 850 job cuts."
    client = StubClient(json.dumps({"answer": "850", "quote": quote, "source_id": PASSAGE["id"]}))
    result = rag_answer(question, page=PASSAGE["page"], knowledge_base=kb, client=client, model="stub")
    assert kb.calls == [(question, PASSAGE["page"])]
    messages, = client.messages
    prompt = json.loads(messages[-1]["content"])
    assert prompt["question"] == question
    assert prompt["sources"] == [{k: PASSAGE[k] for k in ("id", "page", "section", "text")}]
    assert result["status"] == "answered"
    assert result["answer"] == "850" and result["quote"] == quote
    assert result["source"] == {k: PASSAGE[k] for k in ("id", "page", "section", "url")}
    assert result["retrieved_ids"] == [PASSAGE["id"]]
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 3}


def test_rag_distinguishes_no_hits_from_an_intentional_model_abstention():
    client = StubClient('{"answer":"unknown"}')
    question = "How many cuts were in IT?"
    empty = rag_answer(question, knowledge_base=StubKnowledgeBase([]), client=client, model="stub")
    assert empty["status"] == "no_hits" and not client.messages
    answer = rag_answer(question, knowledge_base=StubKnowledgeBase([PASSAGE]), client=client, model="stub")
    assert answer["status"] == "insufficient_evidence"
    assert answer["answer"] == "unknown" and answer["source"] is None and answer["quote"] is None


def test_abstention_can_identify_the_insufficient_passage_without_citing_it_as_an_answer():
    client = StubClient(json.dumps({"answer": "unknown", "source_id": PASSAGE["id"]}))
    result = rag_answer("How many cuts were in IT?", knowledge_base=StubKnowledgeBase([PASSAGE]),
                        client=client, model="stub")
    assert result["status"] == "insufficient_evidence"
    assert result["answer"] == "unknown" and result["source"] is None and result["quote"] is None


@pytest.mark.parametrize("output", [
    "850", "not JSON",
    json.dumps({"answer": "unknown", "source_id": "not-retrieved"}),
    json.dumps({"answer": "850", "quote": "850 job cuts", "source_id": "other"}),
    json.dumps({"answer": "85", "quote": "85", "source_id": PASSAGE["id"]}),
    json.dumps({"answer": "850", "quote": "850 cuts in IT", "source_id": PASSAGE["id"]}),
    json.dumps({"answer": "850", "quote": "850 job cuts", "source_id": PASSAGE["id"], "url": "made-up"}),
])
def test_rag_fails_closed_for_malformed_or_unanchored_output(output):
    result = rag_answer("How many?", knowledge_base=StubKnowledgeBase([PASSAGE]),
                        client=StubClient(output), model="stub")
    assert result["status"] == "invalid_output"
    assert result["answer"] == "unknown" and result["quote"] is None and result["source"] is None


@pytest.mark.parametrize("language", ["json", "", "JSON"])
def test_rag_accepts_one_outer_markdown_fence_without_weakening_evidence_checks(language):
    payload = json.dumps({"answer": "850", "quote": "850 job cuts", "source_id": PASSAGE["id"]})
    result = rag_answer("How many?", knowledge_base=StubKnowledgeBase([PASSAGE]),
                        client=StubClient(f"```{language}\n{payload}\n```"), model="stub")
    assert result["status"] == "answered"
    assert result["answer"] == "850"


@pytest.mark.parametrize("output", [
    '```json\n{"answer":"unknown"}\n```\nAdditional explanation',
    'Explanation\n```json\n{"answer":"unknown"}\n```',
    '```json\n{"answer":"unknown"}\n```\n```json\n{"answer":"unknown"}\n```',
    '```javascript\n{"answer":"unknown"}\n```',
    '```json\n{"answer":"unknown"}',
    '{"answer":"wrong", "answer":"unknown"}',
    '{"answer":"unknown", "ignored":NaN}',
])
def test_rag_rejects_ambiguous_wrappers_and_non_strict_json(output):
    result = rag_answer("How many?", knowledge_base=StubKnowledgeBase([PASSAGE]),
                        client=StubClient(output), model="stub")
    assert result["status"] == "invalid_output"


def test_wrapping_an_invalid_quote_does_not_make_it_valid():
    payload = json.dumps({"answer": "850", "quote": "850 cuts in IT", "source_id": PASSAGE["id"]})
    result = rag_answer("How many?", knowledge_base=StubKnowledgeBase([PASSAGE]),
                        client=StubClient(f"```json\n{payload}\n```"), model="stub")
    assert result["status"] == "invalid_output"


def test_pure_validation_preserves_replay_inputs_and_separates_format_from_support():
    content = json.dumps({"answer": "850", "quote": "850 cuts in IT", "source_id": PASSAGE["id"]})
    response = {"message": {"role": "assistant", "content": content, "private_debug": "do not retain"},
                "finish_reason": "stop", "cost_usd": .01, "request_id": "receipt-1", "headers": {"key": "do not retain"}}
    source = {**PASSAGE, "private_debug": "do not retain"}
    result = validate_response(response, [source])
    assert result["parser_version"] == PARSER_VERSION
    assert result["format_valid"] is True
    assert result["validation_error"] == "unsupported_span"
    assert result["status"] == "invalid_output"
    assert result["retrieved_context"] == [PASSAGE]
    assert result["raw_response"] == {
        "message": {"role": "assistant", "content": content}, "finish_reason": "stop",
        "cost_usd": .01, "request_id": "receipt-1",
    }
    assert validate_response(result["raw_response"], result["retrieved_context"]) == result
    assert response["message"]["private_debug"] == "do not retain"


@pytest.mark.parametrize("envelope", [
    {"message": {"role": "user", "content": '{"answer":"unknown"}'}, "finish_reason": "stop"},
    {"message": {"role": "assistant", "content": '{"answer":"unknown"}'}, "finish_reason": "length"},
    {"message": {"role": "assistant", "content": '{"answer":"unknown"}', "tool_calls": [{"id": "x"}]},
     "finish_reason": "tool_calls"},
])
def test_pure_validation_rejects_invalid_completion_envelopes(envelope):
    result = validate_response(envelope, [PASSAGE])
    assert result["status"] == "invalid_output"
    assert result["format_valid"] is False
    assert result["validation_error"] == "response_envelope"


def test_raw_snapshot_does_not_retain_arbitrary_native_client_objects():
    response = {"message": {"role": "assistant", "content": object()}, "finish_reason": "stop"}
    assert response_snapshot(response) is None
