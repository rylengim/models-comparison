import json

import pytest

from hw2.rag import rag_answer


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
