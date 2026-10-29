"""Dataset validation and a documented, deterministic answer metric."""
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

REQUIRED = ("id", "source", "question", "answer", "evidence", "url")
MONTHS = {name.lower(): i for i, name in enumerate(
    "January February March April May June July August September October November December".split(), 1)}
MONTH_PATTERN = "|".join(MONTHS)


def load_rows(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Строка {n}: неверный JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Строка {n}: нужен JSON-объект")
            rows.append(row)
    return rows


def validate_rows(rows, today=None):
    end = date.fromisoformat(today) if today else date.today()
    errors, ids, questions = [], set(), set()
    if not rows:
        return ["Датасет пуст"]
    for n, row in enumerate(rows, 1):
        prefix = str(row.get("id", n))
        missing = [k for k in REQUIRED if not isinstance(row.get(k), str) or not row[k].strip()]
        if missing:
            errors.append(f"{prefix}: нет непустых строковых полей {missing}")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            errors.append(f"{prefix}: id должен быть безопасным именем файла")
        if row["id"] in ids:
            errors.append(f"{prefix}: повтор id")
        ids.add(row["id"])
        question = " ".join(row["question"].lower().split())
        if question in questions:
            errors.append(f"{prefix}: повтор вопроса")
        questions.add(question)
        if row["answer"] not in row["evidence"]:
            errors.append(f"{prefix}: ответ не содержится в evidence дословно")
        if len(row["answer"]) > 80:
            errors.append(f"{prefix}: ответ длиннее 80 символов")
        try:
            parsed = urlsplit(row["url"])
            valid_url = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        except ValueError:
            valid_url = False
        if not valid_url:
            errors.append(f"{prefix}: неверный URL")
        if "event_date" in row:
            try:
                event = date.fromisoformat(row["event_date"])
                if not date(2025, 1, 1) < event <= end:
                    errors.append(f"{prefix}: дата вне допустимого интервала")
            except (TypeError, ValueError):
                errors.append(f"{prefix}: event_date должна быть YYYY-MM-DD")
    return errors


def normalize(text):
    text = re.sub(r"(?<=\d)[ \u00a0](?=\d{3}(?:\D|$))", "", str(text))
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def final_text(text):
    return re.split(r"\bFINAL\s*:", str(text), flags=re.I)[-1].strip()


def dates_in(text):
    found = []
    text = text.lower()
    for m in re.finditer(rf"\b(\d{{1,2}})\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?\b", text):
        found.append((int(m[1]), MONTHS[m[2]], int(m[3]) if m[3] else None))
    for m in re.finditer(rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", text):
        found.append((int(m[2]), MONTHS[m[1]], int(m[3]) if m[3] else None))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        found.append((int(m[3]), int(m[2]), int(m[1])))
    return found


def is_correct(answer, output, *, event_year=None):
    """Seminar containment, with FINAL isolation and date/numeric safeguards."""
    actual = final_text(output)
    expected = str(answer).strip()
    if not expected or normalize(actual) in {"unknown", "неизвестно", ""}:
        return False
    expected_dates = dates_in(expected)
    if expected_dates:
        d, m, y = expected_dates[0]
        return any(d == ad and m == am and (y is None or y == ay)
                   and (y is not None or event_year is None or ay is None or ay == event_year)
                   for ad, am, ay in dates_in(actual))
    number = expected.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", number):
        candidates = re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)", actual)
        return any(Decimal(number) == Decimal(v.replace(",", "")) for v in candidates)
    needle = normalize(expected)
    return bool(needle) and f" {needle} " in f" {normalize(actual)} "


def score_row(row, output):
    """Use only declared aliases and the source's known year, never an LLM judge."""
    year_match = re.search(r"\b(20\d{2})\b", row.get("event_date", row.get("page", "")))
    year = int(year_match[1]) if year_match else None
    return any(is_correct(answer, output, event_year=year)
               for answer in [row["answer"], *row.get("answer_aliases", [])])
