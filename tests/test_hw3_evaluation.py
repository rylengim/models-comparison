import pytest

from hw3.evaluation import score


@pytest.mark.parametrize(('expected', 'answer', 'correct'), [
    ('6.3', 'FINAL: 6.3', True),
    ('6.3', '3.6', False),
    ('43', 'Not 43; the answer is 46.', False),
    ('43', 'FINAL: 43', True),
    ('43', '143', False),
    ('15 March', 'March 15, 2026', True),
    ('15 March', 'March 15, 2025', False),
    ('15 March', 'not March 15, 2026', False),
])
def test_score_preserves_dates_numbers_and_rejects_negation(expected, answer, correct):
    row = {'id': 'q', 'answer': expected, 'page': '2026 events'}
    assert score(row, answer) is correct


def test_declared_aliases_are_supported_without_changing_the_gold():
    row = {'answer': 'United States', 'answer_aliases': ['USA']}
    assert score(row, 'FINAL: USA')
    assert row['answer'] == 'United States'


def test_non_text_and_empty_answers_are_not_correct():
    assert not score({'answer': '43'}, None)
    assert not score({'answer': '43'}, '')
