"""HW1 answer metric with documented short-answer normalization.

Canonicalization removes combining diacritics and converts standalone English
number words zero..twenty. Composite quantities are left intact, preventing
"one hundred" from being silently interpreted as 1. Explicit negation and
unresolved alternatives are conservatively rejected and remain auditable.
"""
import re
import unicodedata

from freshqa.dataset import final_text, score_row

_NEGATION = re.compile(r"\b(?:not|never|neither|nor|не|нет)\b|n['’]t\b", re.IGNORECASE)
_ALTERNATIVE = re.compile(r'\b(?:or|либо|или)\b|\s+/\s+', re.IGNORECASE)
_NUMBER_WORDS = {word: str(value) for value, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve '
    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_COMPOSITE_WORDS = set(_NUMBER_WORDS) | set(
    'thirty forty fifty sixty seventy eighty ninety hundred thousand million billion'.split())


def canonical(text: str) -> str:
    """Normalize spelling variants without reordering words or erasing decimals."""
    text = ''.join(char for char in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(char))
    words = list(re.finditer(r'[A-Za-z]+', text))
    replacements = []
    for index, word in enumerate(words):
        replacement = _NUMBER_WORDS.get(word[0].lower())
        if replacement is None:
            continue
        adjacent = []
        if index:
            previous = words[index - 1]
            if re.fullmatch(r'[\s-]+', text[previous.end():word.start()]):
                adjacent.append(previous[0].lower())
        if index + 1 < len(words):
            following = words[index + 1]
            if re.fullmatch(r'[\s-]+', text[word.end():following.start()]):
                adjacent.append(following[0].lower())
        if not any(neighbor in _COMPOSITE_WORDS for neighbor in adjacent):
            replacements.append((word.start(), word.end(), replacement))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def score(row: dict, answer: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    actual = canonical(final_text(answer))
    expected = canonical(str(row['answer']))
    aliases = [canonical(alias) for alias in row.get('answer_aliases', [])]
    if _NEGATION.search(actual):
        return False
    if _ALTERNATIVE.search(actual) and not any(_ALTERNATIVE.search(a) for a in [expected, *aliases]):
        return False
    # A question asking how many people expects the cardinality, not the word
    # "people". Limit this equivalence to the explicit count-question shape.
    count = re.fullmatch(r'(\d+)\s+(?:people|persons?)', expected, re.IGNORECASE)
    if count and re.match(r'^how many\b', row.get('question', ''), re.IGNORECASE):
        aliases.append(count[1])
    normalized_row = {**row, 'answer': expected, 'answer_aliases': aliases}
    return score_row(normalized_row, actual)
