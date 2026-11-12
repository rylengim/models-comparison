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


@pytest.mark.parametrize(('expected', 'answer'), [
    ('Twelve', '12'), ('nine', '9'), ('9', 'nine'), ('twenty', '20'),
    ('Miguel Díaz-Canel', 'Miguel Diaz-Canel'),
    ('Deniz Göktaş', 'Deniz Goktas'),
    ('Markus Hinterhäuser', 'Markus Hinterhauser'),
    ('Dina Levačić', 'Dina Levacic'),
])
def test_normalizes_single_number_words_and_diacritics(expected, answer):
    assert score({'answer': expected}, answer)


def test_count_question_does_not_require_echoing_people_unit():
    row = {'answer': 'two people', 'question': 'How many people were killed?'}
    assert score(row, '2')
    assert score(row, 'two')
    assert score(row, '2 people')
    assert not score(row, '22')


def test_does_not_treat_english_composite_numbers_as_single_numbers():
    assert not score({'answer': '20'}, 'twenty one')
    assert not score({'answer': '20'}, 'twenty-one')
    assert not score({'answer': '1'}, 'one hundred')


@pytest.mark.parametrize(('expected', 'answer'), [
    ('five', 'one hundred and five'),
    ('nine', 'one hundred and nine'),
    ('Twelve', 'one hundred and twelve'),
    ('1', 'two hundred and one'),
    ('12', 'one thousand and twelve people'),
    ('5', 'ONE HUNDRED AND FIVE'),
    ('5', 'one-hundred-and-five'),
    ('5', 'one hundred\nand\nfive'),
])
def test_does_not_extract_single_numbers_from_composites_with_and(expected, answer):
    assert not score({'answer': expected}, answer)


@pytest.mark.parametrize('answer', [
    'And five people died.',
    'Five people died and nine were injured.',
    'Five, and the search continued.',
])
def test_and_in_prose_does_not_block_standalone_number_normalization(answer):
    assert score({'answer': '5'}, answer)


def test_rejects_multiple_short_alternatives():
    assert not score({'answer': 'Alice'}, 'Alice or Bob')
    assert not score({'answer': '43'}, '43 or 46')
    assert not score({'answer': '43'}, '43 / 46')
    assert not score({'answer': 'Alice'}, 'Alice либо Bob')
    assert score({'answer': 'Alice'}, 'FINAL: Alice')


def test_normalization_does_not_change_explicit_year_rule():
    row = {'answer': '15 March', 'page': '2026 events'}
    assert not score(row, '15 March 2025')
    assert score(row, 'fifteen March 2026')
