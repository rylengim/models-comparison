from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize(('expected', 'answer'), [
    ('5', 'one point five'),
    ('5', 'five and a half'),
    ('1', 'one point five'),
    ('1', 'one half'),
    ('3', 'three quarters'),
])
@pytest.mark.parametrize('variant', ['plain', 'uppercase', 'hyphenated', 'multiline'])
def test_does_not_extract_integer_parts_from_word_fractions(expected, answer, variant):
    if variant == 'uppercase':
        answer = answer.upper()
    elif variant == 'hyphenated':
        answer = answer.replace(' ', '-')
    elif variant == 'multiline':
        answer = answer.replace(' ', '\n')
    assert not score({'answer': expected}, answer)


@pytest.mark.parametrize(('expected', 'answer'), [
    ('5', '5 and a half'),
    ('1', '1 point five'),
    ('5', '1 point five'),
    ('1', '1 point 5'),
    ('5', '1 point 5'),
])
def test_does_not_extract_integer_parts_from_mixed_digit_fractions(expected, answer):
    assert not score({'answer': expected}, answer)


@pytest.mark.parametrize(('expected', 'answer'), [
    ('5', 'five'),
    ('1', 'one'),
    ('3', 'three'),
    ('5', 'Five people walked one and a half miles.'),
    ('5', 'Half the team left; five stayed.'),
    ('5', 'At one point, five people arrived.'),
    ('one point five', 'one point five'),
    ('five and a half', 'five and a half'),
    ('one half', 'one half'),
    ('three quarters', 'three quarters'),
])
def test_fraction_guard_preserves_standalone_counts_and_complete_phrase_matches(expected, answer):
    assert score({'answer': expected}, answer)


@pytest.mark.parametrize(('expected', 'answer', 'correct'), [
    ('1.5', 'FINAL: 1.5', True),
    ('1', '1.5', False),
    ('5', '1.5', False),
    ('5.5', 'The distance was 5.5 miles.', True),
    ('5', '5.5', False),
    ('0.5', '0.5', True),
    ('0.75', '0.75', True),
    ('3', '0.75', False),
])
def test_fraction_guard_preserves_explicit_decimal_comparisons(expected, answer, correct):
    assert score({'answer': expected}, answer) is correct


@pytest.mark.parametrize('answer', [
    'A five point plan.',
    'It took one point five hours; five people attended.',
    'It took 1 point 5 hours; 5 people attended.',
    'Five people ran a half marathon.',
])
def test_fraction_guard_preserves_numbers_outside_fraction_spans(answer):
    assert score({'answer': '5'}, answer)


def test_explicit_fraction_alias_is_still_accepted():
    row = {'answer': '1.5', 'answer_aliases': ['one point five']}
    assert score(row, 'FINAL: one point five')


@pytest.mark.parametrize(('phrase', 'answer'), [
    ('one point five', 'twenty one point five'),
    ('one half', 'two and one half'),
    ('one point five miles', 'twenty one point five miles'),
    ('one half mile', 'two and one half mile'),
    ('one point five', 'one point five six'),
])
@pytest.mark.parametrize('as_alias', [False, True])
def test_text_fractions_cannot_match_part_of_a_larger_fraction(phrase, answer, as_alias):
    row = {'answer': '1.5', 'answer_aliases': [phrase]} if as_alias else {'answer': phrase}
    assert not score(row, answer)


@pytest.mark.parametrize('phrase', [
    'one point five', 'one half', 'one point five miles', 'one half mile',
])
@pytest.mark.parametrize('as_alias', [False, True])
def test_complete_text_fractions_still_match_inside_prose(phrase, as_alias):
    row = {'answer': '1.5', 'answer_aliases': [phrase]} if as_alias else {'answer': phrase}
    assert score(row, f'The measured distance was {phrase} yesterday.')


@pytest.mark.parametrize(('expected', 'answer'), [
    ('5', '1 point 5, 5 people'),
    ('10', 'The ratio was one point five and 10 people attended.'),
    ('5', '1 point 5; 5 people'),
    ('10', 'The ratio was one point five, and 10 people attended.'),
    ('9', 'The ratio was one point five and nine people attended.'),
    ('12', 'one point 5, 12 people'),
    ('12', 'one point 5 and 12 people'),
])
def test_fraction_spans_stop_before_separate_counts(expected, answer):
    assert score({'answer': expected}, answer)


def test_a_complete_fraction_match_can_follow_a_partial_match():
    assert score({'answer': 'one half'}, 'It was two and one half; then one half.')


def test_fraction_guard_preserves_reordered_dates_in_prose():
    assert score({'answer': '15 March', 'page': '2026 events'},
                 'One point five miles on March 15, 2026.')


def test_fraction_detector_finishes_on_repeated_digit_hyphen_input():
    # A separate process bounds a regex regression that could hang the grader.
    script = (
        "from hw3.evaluation import canonical; "
        "text = '1--' * 40 + 'items'; "
        "assert canonical(text) == text"
    )
    subprocess.run([sys.executable, '-c', script], check=True, timeout=3,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True)


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
