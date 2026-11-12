import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from freshqa.client import BudgetExceeded, CostUnknownError, OpenRouterError
from hw3.client import MeteredClient

RATES = {"test": {"prompt": 0, "completion": 0}}


def response(cost=0.001):
    usage = {"prompt_tokens": 12, "completion_tokens": 3}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "test-call", "choices": [{"message": {"role": "assistant", "content": "FINAL: 4"},
            "finish_reason": "stop"}], "usage": usage}


def test_http_hook_records_actual_cost_and_scope(tmp_path):
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", budget_usd=2, rates=RATES,
                           transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response())), interval=0)
    client.set_scope("bare", "q1", max_calls=2)
    result = client.complete(model="test", messages=[{"role": "user", "content": "2+2"}])
    assert result["cost_usd"] == 0.001
    assert client.spent_usd == 0.001
    record = json.loads((tmp_path / "calls.jsonl").read_text())
    assert (record["config"], record["question_id"], record["prompt_tokens"]) == ("bare", "q1", 12)
    assert "test-key" not in (tmp_path / "calls.jsonl").read_text()
    client.close()


def test_question_limit_blocks_before_network(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", rates=RATES, transport=httpx.MockTransport(serve), interval=0)
    client.set_scope("critic", "q1", max_calls=1)
    client.complete(model="test", messages=[])
    with pytest.raises(BudgetExceeded):
        client.complete(model="test", messages=[])
    assert len(sent) == 1
    client.close()


def test_missing_cost_is_not_zero_and_stops_next_request(tmp_path):
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", interval=0, rates=RATES,
                           transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response(None))))
    client.set_scope("bare", "q1")
    with pytest.raises(CostUnknownError):
        client.complete(model="test", messages=[])
    assert client.accounting_uncertain
    assert client.records[-1]["cost_usd"] is None
    with pytest.raises(CostUnknownError):
        client.complete(model="test", messages=[])
    assert len(client.records) == 1
    client.close()


@pytest.mark.parametrize("choices", [["bad"], 123, {}, None, [], [{"message": None}]])
def test_malformed_paid_choices_keep_cost_in_ledger(tmp_path, choices):
    payload = response()
    payload["choices"] = choices
    ledger = tmp_path / "calls.jsonl"
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    client.set_scope("bare", "q1")
    with pytest.raises(OpenRouterError, match="Malformed paid response"):
        client.complete(model="test", messages=[])
    assert client.spent_usd == 0.001
    assert not client.accounting_uncertain
    assert json.loads(ledger.read_text())["cost_usd"] == 0.001
    assert not client.inflight.exists()
    client.close()


def test_resumed_scope_keeps_call_limit_and_total_spend(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response())
    ledger = tmp_path / "calls.jsonl"
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(serve))
    client.set_scope("bare", "q1", max_calls=1)
    client.complete(model="test", messages=[])
    client.close()

    resumed = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                            transport=httpx.MockTransport(serve))
    resumed.set_scope("bare", "q1", max_calls=1)
    assert resumed.spent_usd == 0.001
    with pytest.raises(BudgetExceeded, match="Question model-call limit"):
        resumed.complete(model="test", messages=[])
    assert len(sent) == 1
    # A different question has its own call allowance while sharing the money cap.
    resumed.set_scope("bare", "q2", max_calls=1)
    resumed.complete(model="test", messages=[])
    assert len(sent) == 2
    assert resumed.spent_usd == 0.002
    resumed.close()


def test_missing_model_rates_block_before_network(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", budget_usd=0.0001,
                           rates=RATES, interval=0, transport=httpx.MockTransport(serve))
    with pytest.raises(ValueError, match="rates are required"):
        client.complete(model="unknown-model", messages=[])
    assert sent == []
    assert client.spent_usd == 0
    assert not client.inflight.exists()
    client.close()


@pytest.mark.parametrize("rate", [
    {},
    {"prompt": -1, "completion": 0},
    {"prompt": 0, "completion": float("nan")},
    {"prompt": float("inf"), "completion": 0},
    {"prompt": True, "completion": 0},
    {"prompt": 0, "completion": 0, "request": -1},
])
def test_invalid_model_rates_block_before_network(tmp_path, rate):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", rates={"test": rate}, interval=0,
                           transport=httpx.MockTransport(serve))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        client.complete(model="test", messages=[])
    assert sent == []
    assert client.records == []
    client.close()


@pytest.mark.parametrize("max_tokens", [0, -1, True, 2.5])
def test_invalid_output_limit_rejected(tmp_path, max_tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        MeteredClient("test-key", tmp_path / "calls.jsonl", rates=RATES, max_tokens=max_tokens)


def test_budget_reserve_blocks_before_network(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", budget_usd=0.001, interval=0,
                           rates={"test": {"prompt": 0.000001, "completion": 0}},
                           transport=httpx.MockTransport(serve))
    with pytest.raises(BudgetExceeded, match="budget reserve"):
        client.complete(model="test", messages=[])
    assert sent == []
    assert client.scope_calls == 0
    assert not client.inflight.exists()
    client.close()


def test_provider_price_ceiling_uses_dollars_per_million_tokens(tmp_path):
    bodies = []
    def serve(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", interval=0, max_tokens=17,
                           rates={"test": {"prompt": 1.5e-7, "completion": 6e-7, "request": 0.0002}},
                           transport=httpx.MockTransport(serve))
    client.complete(model="test", messages=[])
    assert bodies[0]["provider"]["max_price"] == pytest.approx(
        {"prompt": 0.15, "completion": 0.6, "request": 0.0002})
    assert bodies[0]["max_tokens"] == 17
    client.close()


def test_restart_after_interruption_blocks_unreconciled_request(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    marker = ledger.with_suffix(".inflight.json")
    sent = []
    def interrupt_after_send(request):
        sent.append(request)
        pending = json.loads(marker.read_text())
        assert (pending["config"], pending["question_id"]) == ("bare", "q1")
        assert "test-key" not in marker.read_text()
        raise KeyboardInterrupt
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(interrupt_after_send))
    client.set_scope("bare", "q1")
    with pytest.raises(KeyboardInterrupt):
        client.complete(model="test", messages=[])
    client.close()

    resumed = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                            transport=httpx.MockTransport(interrupt_after_send))
    resumed.set_scope("bare", "q1")
    with pytest.raises(CostUnknownError, match="reconcile"):
        resumed.complete(model="test", messages=[])
    assert len(sent) == 1
    assert resumed.accounting_uncertain
    resumed.close()


def test_network_timeout_is_not_retried_and_remains_uncertain_on_restart(tmp_path):
    sent = []
    def timeout(request):
        sent.append(request)
        raise httpx.ReadTimeout("mock timeout", request=request)
    ledger = tmp_path / "calls.jsonl"
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(timeout))
    with pytest.raises(CostUnknownError, match="Network failure"):
        client.complete(model="test", messages=[])
    assert client.records[0]["cost_usd"] is None
    with pytest.raises(CostUnknownError):
        client.complete(model="test", messages=[])
    client.close()

    resumed = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                            transport=httpx.MockTransport(timeout))
    with pytest.raises(CostUnknownError):
        resumed.complete(model="test", messages=[])
    assert len(sent) == 1
    assert len(resumed.records) == 1
    resumed.close()


def test_same_client_after_interruption_cannot_overwrite_pending_request(tmp_path):
    sent = []
    def interrupt_once(request):
        sent.append(request)
        if len(sent) == 1:
            raise KeyboardInterrupt
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", tmp_path / "calls.jsonl", rates=RATES, interval=0,
                           transport=httpx.MockTransport(interrupt_once))
    client.set_scope("bare", "q1")
    with pytest.raises(KeyboardInterrupt):
        client.complete(model="test", messages=[])
    pending = client.inflight.read_text()
    with pytest.raises(CostUnknownError, match="reconcile"):
        client.complete(model="test", messages=[])
    assert len(sent) == 1
    assert client.inflight.read_text() == pending
    assert client.accounting_uncertain
    client.close()


def test_inflight_marker_and_directory_are_synced_before_request(tmp_path, monkeypatch):
    synced = set()
    real_fsync = os.fsync
    def track_fsync(fd):
        stat = os.fstat(fd)
        synced.add((stat.st_dev, stat.st_ino))
        real_fsync(fd)
    monkeypatch.setattr(os, "fsync", track_fsync)
    ledger = tmp_path / "calls.jsonl"
    marker = ledger.with_suffix(".inflight.json")
    synced_before_send = []
    def serve(request):
        for path in (marker, marker.parent):
            stat = path.stat()
            synced_before_send.append((stat.st_dev, stat.st_ino) in synced)
        return httpx.Response(200, json=response())
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(serve))
    client.complete(model="test", messages=[])
    assert synced_before_send == [True, True]
    client.close()


def test_total_budget_is_shared_across_questions_and_restart(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response(cost=0.6))
    rates = {"test": {"prompt": 0, "completion": 0, "request": 0.6}}
    ledger = tmp_path / "calls.jsonl"
    client = MeteredClient("test-key", ledger, budget_usd=1, rates=rates, interval=0,
                           transport=httpx.MockTransport(serve))
    client.set_scope("bare", "q1")
    client.complete(model="test", messages=[])
    client.set_scope("plan", "q2")
    with pytest.raises(BudgetExceeded, match="budget reserve"):
        client.complete(model="test", messages=[])
    client.close()

    resumed = MeteredClient("test-key", ledger, budget_usd=1, rates=rates, interval=0,
                            transport=httpx.MockTransport(serve))
    resumed.set_scope("plan", "q2")
    with pytest.raises(BudgetExceeded, match="budget reserve"):
        resumed.complete(model="test", messages=[])
    assert resumed.spent_usd == 0.6
    assert len(sent) == 1
    resumed.close()


def test_ledger_has_one_owner_and_reopening_restores_limits(tmp_path):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response(cost=0.6))
    ledger = tmp_path / "calls.jsonl"
    kwargs = dict(budget_usd=1, interval=0,
                  rates={"test": {"prompt": 0, "completion": 0, "request": 0.6}},
                  transport=httpx.MockTransport(serve))
    owner = MeteredClient("test-key", ledger, **kwargs)
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            contender = MeteredClient("test-key", ledger, **kwargs)
            contender.close()
        assert sent == []
        owner.set_scope("bare", "q1", max_calls=1)
        owner.complete(model="test", messages=[])
    finally:
        owner.close()
    owner.close()

    resumed = MeteredClient("test-key", ledger, **kwargs)
    try:
        assert resumed.spent_usd == 0.6
        resumed.set_scope("bare", "q1", max_calls=1)
        with pytest.raises(BudgetExceeded, match="Question model-call limit"):
            resumed.complete(model="test", messages=[])
        resumed.set_scope("bare", "q2", max_calls=1)
        with pytest.raises(BudgetExceeded, match="budget reserve"):
            resumed.complete(model="test", messages=[])
        # Repeated cleanup of the previous owner must not release the new lock.
        owner.close()
        with pytest.raises(RuntimeError, match="already in use"):
            contender = MeteredClient("test-key", ledger, **kwargs)
            contender.close()
        assert len(sent) == 1
    finally:
        resumed.close()


def test_ledger_ownership_is_enforced_across_processes(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    owner = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                          transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP request")))
    script = """
import sys
import httpx
from hw3.client import MeteredClient
def no_network(request):
    raise AssertionError("Unexpected HTTP request")
try:
    client = MeteredClient("test-key", sys.argv[1], rates={"test": {"prompt": 0, "completion": 0}},
                           transport=httpx.MockTransport(no_network))
except RuntimeError as error:
    assert "already in use" in str(error)
    print("blocked")
else:
    client.close()
    print("acquired")
"""
    def attempt():
        result = subprocess.run([sys.executable, "-B", "-c", script, str(ledger)],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    try:
        assert attempt() == "blocked"
    finally:
        owner.close()
    assert attempt() == "acquired"


def test_interrupt_after_inflight_removal_blocks_until_reopened(tmp_path, monkeypatch):
    sent = []
    def serve(request):
        sent.append(request)
        return httpx.Response(200, json=response(cost=0.6))
    ledger = tmp_path / "calls.jsonl"
    kwargs = dict(budget_usd=1, interval=0,
                  rates={"test": {"prompt": 0, "completion": 0, "request": 0.6}},
                  transport=httpx.MockTransport(serve))
    client = MeteredClient("test-key", ledger, **kwargs)
    real_sync = client._sync_directory
    interrupted = False
    def interrupt_after_unlink():
        nonlocal interrupted
        real_sync()
        if ledger.exists() and not client.inflight.exists() and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
    monkeypatch.setattr(client, "_sync_directory", interrupt_after_unlink)
    try:
        client.set_scope("bare", "q1")
        with pytest.raises(KeyboardInterrupt):
            client.complete(model="test", messages=[])
        assert interrupted
        assert json.loads(ledger.read_text())["cost_usd"] == 0.6
        assert not client.inflight.exists()
        with pytest.raises(CostUnknownError, match="reconcile"):
            client.complete(model="test", messages=[])
        assert len(sent) == 1
    finally:
        client.close()

    resumed = MeteredClient("test-key", ledger, **kwargs)
    try:
        assert resumed.spent_usd == 0.6
        assert not resumed.accounting_uncertain
        resumed.set_scope("bare", "q2")
        with pytest.raises(BudgetExceeded, match="budget reserve"):
            resumed.complete(model="test", messages=[])
        assert len(sent) == 1
    finally:
        resumed.close()


def test_http_initialization_failure_releases_ledger_ownership(tmp_path, monkeypatch):
    ledger = tmp_path / "calls.jsonl"
    def fail_initialization(**kwargs):
        raise RuntimeError("mock HTTP initialization failure")
    with monkeypatch.context() as patch:
        patch.setattr(httpx, "Client", fail_initialization)
        with pytest.raises(RuntimeError, match="mock HTTP initialization failure"):
            MeteredClient("test-key", ledger, rates=RATES)
    client = MeteredClient("test-key", ledger, rates=RATES,
                           transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP request")))
    client.close()


def test_http_close_failure_still_releases_ledger_ownership(tmp_path, monkeypatch):
    ledger = tmp_path / "calls.jsonl"
    kwargs = dict(rates=RATES,
                  transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP request")))
    client = MeteredClient("test-key", ledger, **kwargs)
    real_close = client.http.close
    def fail_close():
        real_close()
        raise RuntimeError("mock HTTP close failure")
    monkeypatch.setattr(client.http, "close", fail_close)
    with pytest.raises(RuntimeError, match="mock HTTP close failure"):
        client.close()
    replacement = MeteredClient("test-key", ledger, **kwargs)
    replacement.close()


def test_closed_client_rejects_requests_without_creating_inflight(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    client = MeteredClient("test-key", ledger, rates=RATES, interval=0,
                           transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP request")))
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.complete(model="test", messages=[])
    assert not client.inflight.exists()
    assert not ledger.exists()
    assert client.records == []
