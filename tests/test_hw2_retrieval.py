import pytest

from hw2.evaluate_retrieval import evaluate_from_rankings, reciprocal_rank_fusion


def test_rrf_combines_ranks_and_breaks_ties_by_chunk_id():
    result = reciprocal_rank_fusion(["z", "common"], ["a", "common"])
    assert result == ["common", "a", "z"]
    assert reciprocal_rank_fusion(["z", "a"], []) == ["z", "a"]


def test_recall_uses_all_questions_and_only_full_quotes_at_source():
    chunks = [
        {"id": "a", "url": "other", "text": "Flood in May"},
        {"id": "b", "url": "source", "text": "Flood in May"},
    ]
    questions = [
        {"id": "hit", "url": "source", "question": "Flood?", "evidence": "flood, IN MAY!"},
        {"id": "split", "url": "source", "question": "Flood?", "evidence": "Flood in May 2026"},
        {"id": "miss", "url": "source", "question": "Elsewhere?", "evidence": "Flood in May"},
    ]
    result = evaluate_from_rankings(questions, chunks, [["a", "b"], ["b"], []])
    assert result["gold_coverage"] == 2
    for metric in result["metrics"].values():
        assert metric["hits"] == {"1": 0, "3": 1, "5": 1, "10": 1}
        assert metric["recall"]["5"] == 1 / 3
    details = result["questions"]
    assert details[0]["results"]["vector"] == {"rank": 2, "first_gold_id": "b", "top_id": "a"}
    assert details[1]["gold_ids"] == []
    assert details[2]["results"]["rrf"]["rank"] is None
    with pytest.raises(ValueError, match="посторонний ID"):
        evaluate_from_rankings(questions[:1], chunks, [["outdated"]])
