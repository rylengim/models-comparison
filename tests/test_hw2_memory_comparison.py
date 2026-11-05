import json

import pytest

from hw2.memory_comparison import write_comparison


def test_counts_answers_and_session_extraction_once(tmp_path):
    full = {"prompt_tokens": 700, "completion_tokens": 12, "cost_usd": 0.0008}
    window = {"prompt_tokens": 300, "completion_tokens": 14, "cost_usd": 0.0004}
    turns = [{"session_id": s, "question": "Which city?", "full": full, "window": window}
             for s in ("a", "b")]
    extractions = [{"session_id": s, "prompt_tokens": 250, "completion_tokens": 20, "cost_usd": 0.0003}
                   for s in ("a", "b")]
    result = write_comparison(turns, extractions, tmp_path, model="test", price_basis="test",
                              reference_rates=(0.15, 0.6))
    totals = result["totals"]
    assert totals["full"] == {"prompt_tokens": 1400, "completion_tokens": 24, "cost_usd": 0.0016}
    assert totals["window_answers"]["prompt_tokens"] == 600
    assert totals["extractions"]["prompt_tokens"] == 500
    assert totals["window"]["prompt_tokens"] == 1100
    assert totals["window"]["completion_tokens"] == 68
    assert totals["window"]["cost_usd"] == pytest.approx(0.0014)
    assert result["hypothetical_pricing"]["estimated_usd"]["window"] == pytest.approx(0.0002058)
    assert json.loads((tmp_path / "comparison.json").read_text()) == result
    assert (tmp_path / "input_tokens.png").read_bytes().startswith(b"\x89PNG")


def test_missing_extraction_cannot_make_window_appear_cheaper(tmp_path):
    use = {"prompt_tokens": 20, "completion_tokens": 4, "cost_usd": 0}
    turns = [{"session_id": "a", "question": "What time?", "full": use, "window": use}]
    with pytest.raises(ValueError, match="every session"):
        write_comparison(turns, [], tmp_path, model="test", price_basis="test")
