import json
from types import SimpleNamespace

import pytest

from freshqa.groq_check import run_check


def test_diagnostic_records_estimate_without_claiming_sonnet_freshness(tmp_path, monkeypatch):
    client = SimpleNamespace(budget_used_usd=0)

    def run(question, **kwargs):
        assert question == "Who won?"  # No gold or evidence is sent.
        client.budget_used_usd += .01
        return dict(final="Alice", status="ok", steps=1, elapsed_s=.1,
                    cost_usd=None, estimated_cost_usd=.01, known_estimated_cost_usd=.01,
                    events=[], provider="groq", cost_basis="list_price_estimate")

    monkeypatch.setattr("freshqa.groq_check.run_agent", run)
    rows = [{"id": "q", "question": "Who won?", "answer": "Alice", "evidence": "not to model"}]
    results = run_check(rows, client=client, out_dir=tmp_path)
    assert len(results) == 3
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["cost_usd"] is None
    assert summary["sonnet_freshness_checked"] is False
    assert summary["known_estimated_cost_usd"] == pytest.approx(.03)
    assert not (tmp_path / "inflight.json").exists()
    with pytest.raises(ValueError, match="не пуст"):
        run_check(rows, client=client, out_dir=tmp_path)


def test_interruption_leaves_marker_and_blocks_repetition(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("freshqa.groq_check.run_agent", stop)
    client = SimpleNamespace(budget_used_usd=0)
    rows = [{"id": "q", "question": "Who won?", "answer": "Alice"}]
    with pytest.raises(KeyboardInterrupt):
        run_check(rows, client=client, out_dir=tmp_path)
    assert (tmp_path / "inflight.json").exists()
    with pytest.raises(ValueError, match="не пуст"):
        run_check(rows, client=client, out_dir=tmp_path)
