import json
from copy import deepcopy

from hw4.mcp_ask import QuoteStub, run_stub
from hw4.spending import DailySpend, limit_micros


QUESTION = "What is the designation of the new comet?"
SOURCE = "https://example.invalid/release"
OTHER = "https://example.invalid/related-release"
QUOTE = "The new comet is designated C/2026 TEST."
LONG_SNIPPET = "Original search snippet for the new comet. " * 10
SEARCH_RESULT = {"note": "Full result metadata.", "results": [
    {"url": SOURCE, "title": "New comet", "snippet": LONG_SNIPPET, "note": "Original result metadata."},
    {"url": OTHER, "title": "Related release", "snippet": "Unused original search snippet."},
]}


class PublicationTools:
    def __init__(self, *, read_url=SOURCE):
        self.read_result = {
            "url": read_url, "found": True, "keywords_matched": True,
            "excerpts": [f"> {QUOTE}", "Additional publication context."],
        }

    def schemas(self, names):
        return [{"type": "function", "function": {"name": name}} for name in names]

    def run(self, name, arguments):
        if name == "web_search":
            assert arguments == {"query": QUESTION}
            return SEARCH_RESULT
        assert name == "page_find"
        assert arguments["url"] == SOURCE
        return self.read_result


def run_and_observe(monkeypatch, tmp_path, tools):
    contexts = []
    next_message = QuoteStub.next_message

    def observe(stub, messages):
        contexts.append(deepcopy(messages))
        return next_message(stub, messages)

    monkeypatch.setattr(QuoteStub, "next_message", observe)
    result = run_stub(QUESTION, tools, question_budget_micros=limit_micros("0.003"),
                      daily_spend=DailySpend(tmp_path / "spending.sqlite3"))
    return result, [[m for m in step if m["role"] == "tool"] for step in contexts]


def test_stub_previews_candidates_then_compacts_consumed_search(monkeypatch, tmp_path):
    tools = PublicationTools()
    result, (initial, search_only, final) = run_and_observe(monkeypatch, tmp_path, tools)

    assert initial == []
    preview, = [json.loads(m["content"]) for m in search_only]
    assert set(preview) == {"results"}
    assert [(hit["url"], hit["title"]) for hit in preview["results"]] == [
        (SOURCE, "New comet"), (OTHER, "Related release"),
    ]
    first_hit, second_hit = preview["results"]
    assert set(first_hit) == {"url", "title", "snippet"}
    assert LONG_SNIPPET.startswith(first_hit["snippet"].removesuffix("…"))
    assert 0 < len(first_hit["snippet"]) <= 200
    assert second_hit["snippet"] == SEARCH_RESULT["results"][1]["snippet"]
    search, page = final
    compacted = json.loads(search["content"])
    assert compacted["compacted"] is True
    assert compacted["results"] == [{"url": SOURCE}, second_hit]
    assert search["tool_call_id"] == search_only[0]["tool_call_id"]
    assert json.loads(page["content"]) == tools.read_result

    search_trace, page_trace = [event for event in result["events"] if event["type"] == "tool"]
    assert json.loads(search_trace["output"]) == SEARCH_RESULT
    assert page_trace["output"] == page["content"]
    assert result["context_steps"][1] == {
        "step": 2,
        "tool_chars_before": len(search_trace["output"]),
        "tool_chars_after": len(search_only[0]["content"]),
        "compacted_search_results": 0,
        "pre_read_compacted_search_results": 1,
    }
    assert result["context_steps"][-1] == {
        "step": 3,
        "tool_chars_before": len(search_trace["output"]) + len(page_trace["output"]),
        "tool_chars_after": len(search["content"]) + len(page["content"]),
        "compacted_search_results": 1,
        "pre_read_compacted_search_results": 0,
    }
    assert result["context_steps"][-1]["tool_chars_after"] < result["context_steps"][-1]["tool_chars_before"]
    assert (result["status"], result["final"], result["source_url"]) == ("ok", QUOTE, SOURCE)
    assert result["cost_usd"] == result["model_calls"] == 0


def test_stub_keeps_candidate_preview_when_page_reply_has_the_wrong_url(monkeypatch, tmp_path):
    result, (_, search_only, final) = run_and_observe(monkeypatch, tmp_path, PublicationTools(read_url=OTHER))

    assert final[0] == search_only[0]
    assert result["context_steps"][-1]["compacted_search_results"] == 0
    assert result["context_steps"][-1]["pre_read_compacted_search_results"] == 1
    assert (result["status"], result["final"], result["source_url"]) == ("mcp_error", "unknown", None)
