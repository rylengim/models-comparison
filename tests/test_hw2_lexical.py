from hw2.lexical_baseline import evaluate


def test_bm25_counts_missing_gold_and_checks_whole_quote_and_url():
    chunks = [
        {"id": "other-page", "url": "u2", "text": "Flood! In May."},
        {"id": "correct", "url": "u1", "text": "Flood! In May."},
    ]
    questions = [
        {"id": "q1", "url": "u1", "question": "Flood?", "evidence": "FLOOD in May"},
        {"id": "q2", "url": "u1", "question": "Flood?", "evidence": "Flood in May 2025"},
        {"id": "q3", "url": "u1", "question": "Unrelated", "evidence": "Flood in May"},
    ]
    covered, hits, example = evaluate(questions, chunks)
    assert covered == 2
    assert hits == {1: 0, 3: 1, 5: 1, 10: 1}
    assert example == ("q1", "correct", 2)
