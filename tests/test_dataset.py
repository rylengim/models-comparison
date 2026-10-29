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
