from datetime import datetime, timezone

from hw4.mcp_ask import QuoteStub, run_stub
from hw4.spending import DailySpend, limit_micros


class PublicationTools:
    url = "https://example.invalid/release"

    def __init__(self):
        self.calls = []

    def schemas(self, names):
        return [{"type": "function", "function": {"name": name}} for name in names]

    def run(self, name, arguments):
        self.calls.append(name)
        if name == "web_search":
            return {"results": [{"url": self.url}]}
        return {"url": self.url, "found": True, "keywords_matched": True, "excerpts": ["> The source quote."]}


def test_mcp_budget_blocks_before_next_stub_step(tmp_path, monkeypatch):
    steps = []
    next_message = QuoteStub.next_message

    def observe(stub, messages):
        steps.append(True)
        return next_message(stub, messages)

    monkeypatch.setattr(QuoteStub, "next_message", observe)
    tools = PublicationTools()
    daily = DailySpend(tmp_path / "spending.sqlite3")
    result = run_stub("What is the quote?", tools, question_budget_micros=limit_micros("0.001"), daily_spend=daily)

    assert result["status"] == "budget_exceeded"
    assert result["steps"] == len(steps) == 1
    assert tools.calls == ["web_search"]
    assert result["simulated_cost_usd"] == result["daily_spend"]["simulated_usd"] == 0.001
    assert result["daily_spend"]["simulated_calls"] == 1
    assert result["cost_usd"] == result["model_calls"] == 0


def test_mcp_daily_spend_persists_and_rolls_over_in_moscow(tmp_path):
    path = tmp_path / "spending.sqlite3"
    before = datetime(2026, 11, 19, 20, 59, 59, tzinfo=timezone.utc)
    after = datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)

    def ask_at(clock):
        return run_stub("What is the quote?", PublicationTools(), question_budget_micros=limit_micros("0.003"),
                        daily_spend=DailySpend(path, clock=clock))

    first = ask_at(lambda: before)
    second = ask_at(lambda: before)
    third = ask_at(lambda: after)
    assert [r["status"] for r in (first, second, third)] == ["ok"] * 3
    assert [r["daily_spend"]["simulated_usd"] for r in (first, second, third)] == [0.003, 0.006, 0.003]
    assert third["daily_spend"] == {"date": "2026-11-20", "timezone": "Europe/Moscow", "simulated_usd": 0.003,
                                    "simulated_calls": 3}
    assert DailySpend(path, clock=lambda: before).snapshot()["simulated_calls"] == 6
