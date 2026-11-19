import gzip
import json
import sys
from collections import Counter

import pytest

from hw2 import run_benchmark
from hw2.benchmark import CONFIGS, RAG_CONFIGS, _signature
from hw2.benchmark_snapshot import complete_records, load_snapshot, save_snapshot
from hw2.rag import validate_response


@pytest.fixture
def measurements():
    rows = [{"id": "one", "question": "Value?", "answer": "42"}]
    signature = _signature(rows, run_benchmark.MODELS, 6)
    prior, measured = [], []
    for config in CONFIGS:
        row = dict(id="one", question="Value?", config=config, provider="openrouter", measured=True,
                   model=signature["models"][config], final="42", status="ok", steps=1,
                   cost_usd=.01, prompt_tokens=12, completion_tokens=3, cost_basis="provider_reported",
                   retrieved_ids=[])
        receipt = dict(request_id=f"gen-{config}", cost_usd=.01, prompt_tokens=12, completion_tokens=3)
        if config in {"strong_no_tools", "cheap_live_search"}:
            prior.append({**row, "origin": "hw1", "final_message": {"role": "assistant", "content": "FINAL: 42"}})
        elif config in RAG_CONFIGS:
            rag = dict(answer="42", quote="42", status="answered", source={"id": "x", "page": "test"},
                       usage={**receipt, "debug_header": "sensitive-test-value"})
            measured.append({**row, "origin": "hw2", "status": "answered", "rag_result": rag})
        else:
            event = dict(type="llm", step=1, finish_reason="stop", **receipt,
                         message={"role": "assistant", "content": "FINAL: 42", "private_debug": "sensitive-test-value"})
            measured.append({**row, "origin": "hw2", "events": [event], "private_header": "sensitive-test-value"})
    return rows, signature, prior, measured


def test_snapshot_exports_only_new_measurements_and_checks_completeness(tmp_path, measurements):
    rows, signature, prior, measured = measurements
    output = tmp_path / "new.json.gz"
    save_snapshot(output, {"signature": signature}, signature, rows, prior, prior + measured)
    payload = gzip.decompress(output.read_bytes()).decode()
    assert "sensitive-test-value" not in payload
    assert len(json.loads(payload)["records"]) == 3
    records = load_snapshot(output, signature, rows, prior)
    assert len(records) == 5 and {r["config"] for r in records} == set(CONFIGS)
    for broken in (measured[:-1], measured + measured[:1], [dict(measured[0], config="unknown"), *measured[1:]],
                   [dict(measured[0], model="wrong"), *measured[1:]],
                   [dict(measured[0], cost_usd=.5), *measured[1:]]):
        with pytest.raises(ValueError):
            complete_records(broken, prior, rows, signature)
    with pytest.raises(ValueError, match="протокол"):
        load_snapshot(output, {**signature, "max_steps": 1}, rows, prior)


def test_historical_snapshot_still_requires_the_same_inputs(tmp_path, measurements):
    rows, signature, prior, measured = measurements
    output = tmp_path / "measured.json.gz"
    save_snapshot(output, {"signature": signature}, signature, rows, prior, prior + measured)
    hashes = {**signature["protocol_sha256"], "freshqa/agent.py": "0" * 64}
    updated = {**signature, "protocol_sha256": hashes}

    assert len(load_snapshot(output, updated, rows, prior, historical=True)) == len(CONFIGS)
    with pytest.raises(ValueError, match="протокол"):
        load_snapshot(output, updated, rows, prior)

    changed_fields = (
        {"dataset_sha256": "0" * 64},
        {"models": {**signature["models"], "cheap_rag": "different-model"}},
        {"max_steps": 1},
        {"tools": {}},
        {"protocol_sha256": {**hashes, "hw2/data/corpus.jsonl": "0" * 64}},
        {"protocol_sha256": {**hashes, "hw2/chunk_corpus.py": "0" * 64}},
    )
    for change in changed_fields:
        with pytest.raises(ValueError, match="протокол"):
            load_snapshot(output, {**updated, **change}, rows, prior, historical=True)


def test_fresh_clone_live_flag_replays_without_opening_a_key_or_creating_ledger(tmp_path, monkeypatch, measurements):
    rows, signature, prior, measured = measurements
    data = tmp_path / "data"
    data.mkdir()
    (data / "questions.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (data / "unanswerable.jsonl").write_text("")
    (data / "corpus.jsonl").write_text("")
    out = tmp_path / "report"
    ledger = tmp_path / "ledger"
    reference = tmp_path / "reference-report"
    save_snapshot(reference / "measured.json.gz", {"signature": signature}, signature, rows, prior, prior + measured)
    monkeypatch.setattr(run_benchmark, "DATA", data)
    monkeypatch.setattr(run_benchmark, "REPORT", reference)
    monkeypatch.setattr(run_benchmark, "load_prior_runs", lambda _: prior)
    monkeypatch.setattr(run_benchmark, "project_key", lambda *args: pytest.fail("unexpected credential access"))
    monkeypatch.setattr(run_benchmark, "benchmark", lambda *args, **kw: pytest.fail("unexpected new run"))
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--live",
                                      "--traces", str(ledger), "--out", str(out)])
    assert run_benchmark.main() == 0
    assert (out / "report.md").is_file() and not ledger.exists()


@pytest.mark.parametrize("stale_ledger", [False, True], ids=["clean-clone", "stale-ledger"])
def test_cli_replays_tracked_hw2_snapshot_offline(tmp_path, monkeypatch, stale_ledger):
    out = tmp_path / "report"
    ledger = tmp_path / "ledger"
    manifest = ledger / "manifest.json"
    stale_manifest = b'{"signature":{"format":0}}\n'
    if stale_ledger:
        ledger.mkdir()
        manifest.write_bytes(stale_manifest)
    monkeypatch.setattr(run_benchmark, "project_key", lambda *args: pytest.fail("unexpected credential access"))
    monkeypatch.setattr(run_benchmark, "benchmark", lambda *args, **kwargs: pytest.fail("unexpected new run"))
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--traces", str(ledger), "--out", str(out)])

    assert run_benchmark.main() == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report["questions"]) == set(CONFIGS)
    measured = [(config, record) for config, records in report["questions"].items() for record in records]
    assert len(measured) == len({(config, record["id"]) for config, record in measured}) == 570
    assert Counter(record["origin"] for _, record in measured) == {"hw1": 208, "hw2": 362}
    assert all(config["n"] == 114 and config["complete"] for config in report["configs"].values())
    assert (out / "report.md").is_file()
    assert ledger.exists() is stale_ledger
    if stale_ledger:
        assert list(ledger.iterdir()) == [manifest]
        assert manifest.read_bytes() == stale_manifest


def test_snapshot_retains_future_rag_replay_inputs_but_not_transport_metadata(tmp_path, measurements):
    rows, signature, prior, measured = measurements
    source = {"id": "x", "page": "test", "section": "results", "url": "https://example.org/result", "text": "Value: 42"}
    raw = {"message": {"role": "assistant", "content": '```json\n{"answer":"42","quote":"Value: 42","source_id":"x"}\n```'},
           "finish_reason": "stop"}
    for record in measured:
        if record["config"] in RAG_CONFIGS:
            receipt = record["rag_result"]["usage"]
            record["rag_result"] = validate_response({**raw, **receipt}, [source])
            record["rag_result"]["raw_response"]["private_header"] = "sensitive-test-value"
            record["rag_result"]["raw_response"]["message"]["private_debug"] = "sensitive-test-value"
            record["rag_result"]["retrieved_context"][0]["private_debug"] = "sensitive-test-value"
    output = tmp_path / "new.json.gz"
    save_snapshot(output, {"signature": signature}, signature, rows, prior, prior + measured)
    payload = gzip.decompress(output.read_bytes()).decode()
    assert "sensitive-test-value" not in payload
    records = load_snapshot(output, signature, rows, prior)
    for record in records:
        if record["config"] in RAG_CONFIGS:
            saved = record["rag_result"]
            assert saved["raw_response"]["message"] == raw["message"]
            assert saved["retrieved_context"] == [source]
            replayed = validate_response(saved["raw_response"], saved["retrieved_context"])
            assert replayed["status"] == "answered" and replayed["answer"] == "42"
