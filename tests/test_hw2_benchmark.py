import json

import pytest

from hw2.benchmark import CONFIGS, benchmark
from freshqa.client import CostUnknownError


MODELS = dict.fromkeys(CONFIGS, "stub")
ROW = {"id": "case-1", "question": "How many jobs?", "answer": "42", "evidence": "42 jobs."}


class Client:
    def __init__(self, budget=10):
        self.spent_usd = 0.0
        self.budget_usd = budget
        self.calls = []

    def complete(self, *, model, messages, tools=None):
        self.calls.append(messages)
        self.spent_usd += .1
        if tools is None and "sources" in messages[-1]["content"]:
            content = json.dumps({"answer": "42", "quote": "42 jobs.", "source_id": "one"})
        else:
            content = "FINAL: 42"
        return {"message": {"role": "assistant", "content": content}, "cost_usd": .1,
                "finish_reason": "stop", "request_id": str(len(self.calls)),
                "prompt_tokens": 20, "completion_tokens": 5}


class KB:
    def search(self, query, page=None):
        assert query == ROW["question"]
        return [{"id": "one", "page": "Jobs", "section": "Statistics",
                 "text": "42 jobs.", "url": "https://example.org/Jobs"}]


def test_five_configs_and_no_repeat_on_resume(tmp_path):
    client = Client()
    assert benchmark([ROW], models=MODELS, out_dir=tmp_path, client=client, knowledge_base=KB())
    assert len(client.calls) == 5
    assert all("evidence" not in messages[-1]["content"] for messages in client.calls)
    for name in CONFIGS:
        record = json.loads((tmp_path / name / "case-1.json").read_text())
        assert record["correct"] and record["outcome"] == "correct_answer"
        assert record["cost_usd"] == pytest.approx(.1) and not record["measured"]
        assert record["retrieved_ids"] == (["one"] if name in {"cheap_rag", "medium_rag"} else [])
    fresh_client = Client()
    assert benchmark([ROW], models=MODELS, out_dir=tmp_path, client=fresh_client, resume=True)
    assert not fresh_client.calls
    with pytest.raises(ValueError, match="resume"):
        benchmark([dict(ROW, question="Changed?")], models=MODELS, out_dir=tmp_path, resume=True)


def test_legacy_costs_excluded_from_active_budget(tmp_path):
    prior = {"id": ROW["id"], "config": "strong_no_tools", "question": ROW["question"],
             "final": "42", "model": "stub", "measured": True, "origin": "hw1", "status": "ok",
             "provider": "openrouter", "cost_usd": 9.0, "cost_basis": "provider_reported",
             "final_message": {"role": "assistant", "content": "FINAL: 42"}}
    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, prior_records=[prior])
    assert (tmp_path / "strong_no_tools/case-1.json").is_file()
    client = Client(.1)
    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, client=client, knowledge_base=KB(), resume=True)
    assert client.spent_usd == pytest.approx(.1) and len(client.calls) == 1
    again = Client(.1)
    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, client=again, knowledge_base=KB(), resume=True)
    assert not again.calls and again.spent_usd == pytest.approx(.1)


def test_ambiguous_paid_call_prevents_automatic_retry(tmp_path):
    class Unknown(Client):
        def complete(self, **kwargs):
            self.accounting_uncertain = True
            raise CostUnknownError("uncertain")

    with pytest.raises(RuntimeError, match="сверка"):
        benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=["strong_no_tools"], client=Unknown())
    assert (tmp_path / "inflight.json").exists()
    failed = json.loads((tmp_path / "strong_no_tools/case-1.json").read_text())
    assert failed["cost_usd"] is None and not failed["refused"]
    client = Client()
    with pytest.raises(RuntimeError, match="сверка"):
        benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=["strong_no_tools"], client=client, resume=True)
    assert not client.calls


def test_interrupt_between_debit_and_result_keeps_marker(tmp_path):
    class Interrupted(Client):
        def complete(self, **kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=["cheap_rag"],
                  client=Interrupted(), knowledge_base=KB())
    assert (tmp_path / "inflight.json").exists()
    assert not (tmp_path / "cheap_rag/case-1.json").exists()


@pytest.mark.parametrize("name", ["strong_no_tools", "cheap_rag"])
def test_preflight_reserve_rejection_keeps_row_pending(tmp_path, name):
    from freshqa.client import BudgetExceeded

    class Reserve(Client):
        def complete(self, **kwargs):
            raise BudgetExceeded("insufficient headroom")

    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=[name],
                         client=Reserve(), knowledge_base=KB())
    assert not (tmp_path / "inflight.json").exists()
    assert not (tmp_path / name / "case-1.json").exists()
    assert benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=[name],
                     client=Client(), knowledge_base=KB(), resume=True)


def test_preflight_rejection_after_tool_use_preserves_actual_spending(tmp_path):
    from freshqa.client import BudgetExceeded

    class Reserve(Client):
        def complete(self, **kwargs):
            if self.calls:
                raise BudgetExceeded("out of headroom")
            response = super().complete(**kwargs)
            response["message"] = {"role": "assistant", "tool_calls": [
                {"id": "first", "type": "function", "function": {
                    "name": "knowledge_base", "arguments": json.dumps({"query": ROW["question"]})}}
            ]}
            return response

    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=["cheap_agent_kb"],
                         client=Reserve(), knowledge_base=KB())
    partial = json.loads((tmp_path / "cheap_agent_kb/case-1.json").read_text())
    assert partial["status"] == "budget_exceeded" and partial["cost_usd"] == .1
    assert partial["retrieved_ids"] == ["one"] and not partial["refused"]
    assert not (tmp_path / "inflight.json").exists()
    other = Client()
    assert not benchmark([ROW], out_dir=tmp_path, models=MODELS, config_names=["cheap_agent_kb"],
                         client=other, resume=True)
    assert not other.calls
