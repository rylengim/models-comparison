from copy import deepcopy

import pytest

from hw4.mcp_ask import QuoteStub, run_stub
from hw4.prompt_cache import SimulatedPromptCache
from hw4.spending import DailySpend, limit_micros


QUESTION = "What is the designation of the new comet?"
SOURCE = "https://example.invalid/release"
QUOTE = "The new comet is designated C/2026 TEST."


class PublicationTools:
    def schemas(self, names):
        return [{"type": "function", "function": {"name": name}} for name in names]

    def run(self, name, arguments):
        if name == "web_search":
            return {"results": [{"url": SOURCE, "snippet": "A new comet was designated."}]}
        assert name == "page_find" and arguments["url"] == SOURCE
        return {"url": SOURCE, "found": True, "keywords_matched": True, "excerpts": [f"> {QUOTE}"]}


def observe_ask(monkeypatch, tmp_path, budget):
    observed = []
    next_message = QuoteStub.next_message

    def record(stub, messages):
        observed.append(deepcopy(messages))
        return next_message(stub, messages)

    monkeypatch.setattr(QuoteStub, "next_message", record)
    trace = run_stub(QUESTION, PublicationTools(), question_budget_micros=limit_micros(budget),
                     daily_spend=DailySpend(tmp_path / "spending.sqlite3"))
    return trace, observed


def test_prompt_cache_reuses_only_the_stable_prefix(monkeypatch, tmp_path):
    trace, observed = observe_ask(monkeypatch, tmp_path, "0.003")
    cold, warm, final = observed
    assert cold[0] == warm[0] == final[0]
    block, = cold[0]["content"]
    assert block["type"] == "text" and block["cache_control"] == {"type": "ephemeral"}
    assert all(messages[1]["content"] == QUESTION for messages in observed)
    assert [sum(message["role"] == "tool" for message in messages) for messages in observed] == [0, 1, 2]

    cache = trace["prompt_cache"]
    assert cache["mode"] == "simulated" and cache["token_estimate"] == "utf8_bytes_div_4"
    steps = cache["steps"]
    assert [step["step"] for step in steps] == [1, 2, 3]
    assert len({step["prefix_sha256"] for step in steps}) == 1
    assert len({step["prefix_tokens"] for step in steps}) == 1
    assert steps[0]["input_tokens"] < steps[1]["input_tokens"]
    assert steps[0]["cached_tokens"] == 0
    assert all(0 < step["cached_tokens"] == step["prefix_tokens"] < step["input_tokens"] for step in steps[1:])
    assert all(step["cached_input_fraction"] == pytest.approx(step["cached_tokens"] / step["input_tokens"])
               for step in steps)
    assert cache["input_tokens"] == sum(step["input_tokens"] for step in steps)
    assert cache["cached_tokens"] == sum(step["cached_tokens"] for step in steps)
    assert cache["cached_input_fraction"] == pytest.approx(cache["cached_tokens"] / cache["input_tokens"])
    assert (trace["status"], trace["final"], trace["source_url"]) == ("ok", QUOTE, SOURCE)
    assert trace["model_calls"] == trace["cost_usd"] == 0


def test_blocked_stub_step_does_not_prime_or_hit_cache(monkeypatch, tmp_path):
    trace, observed = observe_ask(monkeypatch, tmp_path, "0.001")

    assert trace["status"] == "budget_exceeded"
    cache = trace["prompt_cache"]
    first, = cache["steps"]
    assert len(observed) == first["step"] == trace["steps"] == 1
    assert cache["input_tokens"] == first["input_tokens"] > 0
    assert cache["cached_tokens"] == first["cached_tokens"] == cache["cached_input_fraction"] == 0
    assert trace["model_calls"] == trace["cost_usd"] == 0


def test_tool_schema_change_creates_a_cold_prefix():
    cache = SimulatedPromptCache()
    system = {"role": "system", "content": "Read the publication."}
    user = {"role": "user", "content": QUESTION}
    tools = [{"type": "function", "function": {"name": "web_search", "description": "Search publications"}}]
    _, first = cache.prepare([system, user], tools)
    _, repeated = cache.prepare([system, {**user, "content": "What is the source?"}], tools)
    changed_tools = deepcopy(tools)
    changed_tools[0]["function"]["description"] = "Search sources"
    _, changed = cache.prepare([system, user], changed_tools)

    assert system["content"] == "Read the publication."
    assert first["prefix_sha256"] == repeated["prefix_sha256"] != changed["prefix_sha256"]
    assert first["cached_tokens"] == changed["cached_tokens"] == 0
    assert repeated["cached_tokens"] == repeated["prefix_tokens"] > 0
