import json

import pytest

from freshqa.dataset import is_correct, load_rows, score_row, validate_rows


def row(**changes):
    return {"id": "one", "source": "fresh", "question": "What happened?",
            "answer": "NGC 123", "evidence": "Observed NGC 123 in January 2026.",
            "url": "https://example.org/news", **changes}


def test_load_rejects_broken_json_instead_of_silently_dropping(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row()) + "\ninvalid\n")
    with pytest.raises(ValueError, match="2"):
        load_rows(path)


def test_validate_requires_literal_quote_unique_ids_and_nonempty_data():
    assert not validate_rows([row()])
    assert validate_rows([])
    assert validate_rows([row(), row()])
    assert validate_rows([row(answer="ngc 123")])
    assert validate_rows([row(event_date="2027-01-01")], today="2026-10-29")
    assert validate_rows([row(url="ftp://example.org")])
    assert validate_rows([row(url="https://[")])


@pytest.mark.parametrize("answer,output,expected", [
    ("NGC 123", "FINAL: NGC 123.", True),
    ("15 March", "FINAL: March 15, 2026", True),
    ("15 March 2026", "FINAL: 2026-03-15", True),
    ("15 March 2026", "FINAL: 15 March 2025", False),
    ("12", "FINAL: 112", False),
    ("1,200", "FINAL: 1200", True),
    ("12", "FINAL: 12.5", False),
    ("12", "FINAL: 12.", True),
    ("NGC 123", "I considered NGC 123. FINAL: unknown", False),
    ("", "FINAL: anything", False),
    ("Congo", "FINAL: Democratic Republic of the Congo", True),
    ("25 000 km/s", "FINAL: 25,000 km/s", True),
])
def test_scoring(answer, output, expected):
    assert is_correct(answer, output) is expected


def test_score_row_rejects_explicit_wrong_year_and_accepts_declared_alias():
    dated = row(answer="22 March", page="2026 in spaceflight")
    assert score_row(dated, "March 22")
    assert score_row(dated, "March 22, 2026")
    assert not score_row(dated, "March 22, 2025")
    assert score_row(row(answer="two and a half", answer_aliases=["2.5 days"]), "2.5 days")


@pytest.mark.parametrize(('expected', 'actual'), [
    ('Twelve', '12 people were killed.'), ('12', 'Twelve people were killed.'),
    ('nine', '9'), ('21', 'twenty-one'), ('105', 'one hundred and five'),
    ('1200', 'twelve hundred'), ('30,000', 'thirty thousand'),
    ('thirty thousand people', '30,000 people'), ('two people', '2 people'),
    ('1.4 billion years', '1,400,000,000 years'), ('13 years', 'thirteen years'),
])
def test_metric_v2_numeric_equivalences(expected, actual):
    assert is_correct(expected, actual)


@pytest.mark.parametrize(('expected', 'actual'), [
    ('12', '112'), ('12', '12.5'), ('12', '-12'), ('12', 'twelve hundred'),
    ('5', 'one hundred and five'), ('20', 'twenty-one'),
    ('5', 'one point five'), ('5', 'five and a half'),
    ('1', 'one half'), ('3', 'three quarters'), ('5', '1 point 5'),
    ('12', '12-15'), ('12', '12/15'), ('12', '12–15'),
    ('13 years', '13 months'), ('25 000 km/s', '25,000 m/s'),
    ('two people', 'two cars'), ('Twelve Angry Men', '12'),
])
def test_metric_v2_does_not_match_partial_quantities(expected, actual):
    assert not is_correct(expected, actual)


def test_count_question_accepts_value_without_repeated_people_unit():
    gold = row(answer='two people', question='How many people were killed?')
    assert score_row(gold, '2')
    assert score_row(gold, 'two')
    assert not score_row(gold, '2 cars')
    assert not score_row(gold, '22')
    assert not score_row(row(answer='two people', question='Which phrase was used?'), '2')


@pytest.mark.parametrize('answer', [
    'unknown\n\nThe article gives 519 for a different date.',
    'FINAL: Unknown. The number 519 is from a later report.',
    'неизвестно: упоминается 519, но на другую дату.',
])
def test_explicit_abstention_cannot_pass_by_explanatory_number(answer):
    assert not is_correct('519', answer)


def test_unknown_word_inside_an_answer_is_not_an_abstention():
    assert is_correct('species', 'An unknown species was discovered.')
    assert is_correct('Unknown Pleasures', 'Unknown Pleasures')


@pytest.mark.parametrize(('expected', 'actual'), [
    ('12', '12 to 15'), ('12', '12 through 15'), ('12', '12—15'),
    ('12', '−12'), ('12', '- 12'),
    ('1', 'one-third'), ('2', 'two-thirds'), ('1', 'one hundredth'),
    ('1', 'one trillion'), ('12', '12,5'), ('12', '12,34'),
    ('25 km', '25 km/h'), ('25 km', '25 km per hour'),
    ('25 m', '25 m^2'), ('25 m', '25 m / s'),
    ('two and a half', 'two and a half million'),
    ('1.5 cars', '1 5 cars'), ('8.15m', '8 15m'),
    ('2', 'two‑thirds'), ('1', 'one‑third'),
    ('12', '12×10^3'), ('12', '12:30'), ('1', 'one fifteenth'),
    ('12', 'twelve quadrillion'), ('£30.4bn', '£30 4bn'),
])
def test_metric_rejects_whole_expression_and_unit_boundary_bypasses(expected, actual):
    assert not is_correct(expected, actual)


@pytest.mark.parametrize('separator', [' - ', ', ', '; ', ' – ', ' / '])
def test_explicit_abstention_accepts_punctuation_variants(separator):
    assert not is_correct('519', 'unknown' + separator + 'the source mentions 519 for another date.')


@pytest.mark.parametrize(('expected', 'actual'), [
    ('Twelve Angry Men', 'The film is Twelve Angry Men.'),
    ('two and a half', 'It took two and a half days.'),
    ('8.15m', 'The jump was 8.15m.'),
    ('1.5 cars', 'The estimate is 1.5 cars.'),
    ('25 km/s', 'It travels 25 km/s.'),
    ('£30.4bn', 'The cost was £30.4bn.'),
    ('Unknown: Cosmic Time Machine', 'Unknown: Cosmic Time Machine'),
])
def test_literal_unsupported_quantities_and_names_still_match(expected, actual):
    assert is_correct(expected, actual)


def test_numeric_parser_bounds_do_not_recurse_through_long_compounds():
    from freshqa.quantities import number_value

    assert number_value('one hundred ' * 2000) is None
    assert not is_correct('1', 'one hundred ' * 2000)


def test_named_abstention_exception_requires_exact_gold_not_an_explanation():
    assert not is_correct('519', 'Unknown: 519 appears in a later source.')
    assert not is_correct('Unknown: Cosmic Time Machine',
                          'Unknown: I found Unknown: Cosmic Time Machine, but cannot confirm it.')


def test_scaled_decimal_arithmetic_does_not_round_distinct_large_integers_together():
    assert is_correct('1000000000000000000000000000001000',
                      '1000000000000000000000000000001 thousand')
    assert not is_correct('1000000000000000000000000000000000',
                          '1000000000000000000000000000001 thousand')
