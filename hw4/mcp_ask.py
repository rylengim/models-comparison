"""Exercise the agent loop and publication tools without a model provider."""

import json
import sqlite3
from functools import partial
from urllib.parse import urlsplit

from freshqa.agent import run_agent
from freshqa.client import BudgetExceeded
from hw2.prepare_corpus import normalize_evidence

from .context import compact_search_results
from .mcp_client import TOOL_NAMES, with_publication_tools
from .prompt_cache import SimulatedPromptCache
from .spending import DEFAULT_BUDGET_USD, DEFAULT_LEDGER, STUB_CALL_MICROS, DailySpend, as_usd, limit_micros


def tool_call(name, arguments):
    return {"role": "assistant", "tool_calls": [{
        "id": f"stub-{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }]}


class QuoteStub:
    """Search once, read the first hit, and return a quote from that page."""

    provider = "stub"
    spent_usd = 0.0

    def __init__(self, question, *, question_budget_micros, daily_spend):
        self.question = question
        self.keywords = [word for word in normalize_evidence(question).split() if len(word) > 2][:12]
        self.question_budget_micros = question_budget_micros
        self.daily_spend = daily_spend
        self.simulated_spent_micros = 0
        self.context_steps = []
        self.prompt_cache = SimulatedPromptCache()
        self.source_url = None
        self.mcp_error = False
        self.spending_error = False

    def next_message(self, messages):
        replies = [json.loads(msg["content"]) for msg in messages if msg["role"] == "tool"]
        if not replies:
            return tool_call("web_search", {"query": self.question})
        if any(not isinstance(reply, dict) or "error" in reply for reply in replies):
            raise ValueError("Invalid MCP result")
        hits = replies[0]["results"]
        if not isinstance(hits, list):
            raise ValueError("Invalid MCP search results")
        if not hits:
            return {"role": "assistant", "content": "FINAL: unknown"}
        url = hits[0]["url"]
        if not isinstance(url, str):
            raise ValueError("Invalid document URL")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Invalid document URL")
        if len(replies) == 1:
            return tool_call("page_find", {"url": url, "keywords": self.keywords})

        document = replies[1]
        if document["url"] != url or not isinstance(document["found"], bool):
            raise ValueError("MCP returned a different document")
        passages = document["excerpts"]
        if not isinstance(passages, list) or any(not isinstance(text, str) for text in passages):
            raise ValueError("Invalid MCP passages")
        if document["found"] and document["keywords_matched"] is True:
            quotes = [line[2:].strip() for excerpt in passages for line in excerpt.splitlines()
                      if line.startswith("> ") and line[2:].strip()]
            terms = set(self.keywords)
            best = max(quotes, key=lambda quote: len(terms.intersection(normalize_evidence(quote).split())), default="")
            if best:
                self.source_url = url
                return {"role": "assistant", "content": f"FINAL: {best}"}
        return {"role": "assistant", "content": "FINAL: unknown"}

    def complete(self, *, model, messages, tools=None):
        next_spend = self.simulated_spent_micros + STUB_CALL_MICROS
        if next_spend > self.question_budget_micros:
            raise BudgetExceeded("Лимит на вопрос исчерпан. Следующий вызов заглушки заблокирован.")
        try:
            self.daily_spend.record_stub_call(STUB_CALL_MICROS)
        except (OSError, sqlite3.Error):
            self.spending_error = True
            raise BudgetExceeded("Дневной счётчик недоступен. Вызов заглушки заблокирован.") from None
        self.simulated_spent_micros = next_spend
        try:
            prepared, metrics = compact_search_results(messages)
            self.context_steps.append({"step": len(self.context_steps) + 1, **metrics})
            prepared, _ = self.prompt_cache.prepare(prepared, tools)
            message = self.next_message(prepared)
        except (IndexError, KeyError, TypeError, ValueError):
            self.mcp_error = True
            message = {"role": "assistant", "content": "FINAL: unknown"}
        return {"message": message, "cost_usd": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "request_id": None}


def run_stub(question, registry, *, question_budget_micros, daily_spend):
    stub = QuoteStub(question, question_budget_micros=question_budget_micros, daily_spend=daily_spend)
    result = run_agent(question, model="stub", client=stub, registry=registry, tool_names=TOOL_NAMES, max_steps=4)
    if stub.mcp_error:
        result["status"] = "mcp_error"
    try:
        day = daily_spend.snapshot()
    except (OSError, sqlite3.Error):
        day = None
    if stub.spending_error or day is None:
        result.update(status="spending_error", final="unknown")
    for event in result["events"]:
        if event["type"] == "llm":
            event["type"] = "stub"
    result.update(
        model="stub", model_calls=0, cost_basis="stub",
        source_url=stub.source_url if result["status"] == "ok" else None,
        question_budget_usd=as_usd(question_budget_micros),
        simulated_cost_usd=as_usd(stub.simulated_spent_micros),
        simulated_remaining_usd=as_usd(question_budget_micros - stub.simulated_spent_micros),
        daily_spend=day,
        context_steps=stub.context_steps,
        prompt_cache=stub.prompt_cache.report(),
    )
    return result


async def ask_publications(question, *, budget_usd=None, ledger_path=DEFAULT_LEDGER):
    if not question.strip():
        raise ValueError("Введите вопрос для поиска по корпусу.")
    budget = limit_micros(DEFAULT_BUDGET_USD if budget_usd is None else budget_usd)
    return await with_publication_tools(partial(
        run_stub, question.strip(), question_budget_micros=budget, daily_spend=DailySpend(ledger_path),
    ))
