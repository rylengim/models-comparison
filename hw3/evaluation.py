"""HW1 answer metric with documented short-answer normalization.

Canonicalization removes combining diacritics and converts standalone English
number words zero..twenty. Composite quantities are left intact, preventing
"one hundred and five" from being silently interpreted as 1 or 5. Word decimals
and half/quarter fractions are kept whole and excluded from numeric containment.
Explicit negation and unresolved alternatives remain conservatively rejected.
"""
import re
import unicodedata

from freshqa.dataset import dates_in, final_text, normalize, score_row

_NEGATION = re.compile(r"\b(?:not|never|neither|nor|не|нет)\b|n['’]t\b", re.IGNORECASE)
_ALTERNATIVE = re.compile(r'\b(?:or|либо|или)\b|\s+/\s+', re.IGNORECASE)
_NUMBER_WORDS = {word: str(value) for value, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve '
    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_COMPOSITE_WORDS = set(_NUMBER_WORDS) | set(
    'thirty forty fifty sixty seventy eighty ninety hundred thousand million billion'.split())
_NUMBER_GAP = r'[\s-]+'
# Signs stay outside detected spans; otherwise '-' also matches NUMBER_GAP and
# repeated digit/hyphen input can cause exponential regex backtracking.
_NUMBER_TOKEN = r'(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|' + '|'.join(sorted(_COMPOSITE_WORDS)) + ')'
_NUMBER_PHRASE = rf'{_NUMBER_TOKEN}(?:{_NUMBER_GAP}(?:and{_NUMBER_GAP})?{_NUMBER_TOKEN})*'
# "and" after decimal digits starts a separate phrase, not more decimal digits.
_DECIMAL_DIGITS = rf'{_NUMBER_TOKEN}(?:{_NUMBER_GAP}{_NUMBER_TOKEN})*'
# Recognize whole expressions without guessing their numeric value. Reuse the
# same spans for word normalization and numeric matching, including mixed digits.
_FRACTION = re.compile(
    rf'(?<![\w.])(?:{_NUMBER_PHRASE}{_NUMBER_GAP}point{_NUMBER_GAP}{_DECIMAL_DIGITS}'
    rf'|(?:{_NUMBER_PHRASE}{_NUMBER_GAP}and{_NUMBER_GAP})?'
    rf'(?:{_NUMBER_PHRASE}|a){_NUMBER_GAP}(?:half|halves|quarters?))(?!\w|\.\d)',
    re.IGNORECASE,
)


def canonical(text: str) -> str:
    """Normalize spelling variants without reordering words or erasing decimals."""
    text = ''.join(char for char in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(char))
    fractions = [match.span() for match in _FRACTION.finditer(text)]
    words = list(re.finditer(r'[A-Za-z]+', text))
    protected = {word.start() for word in words
                 if any(start <= word.start() < end for start, end in fractions)}
    replacements = []
    for index, word in enumerate(words):
        replacement = _NUMBER_WORDS.get(word[0].lower())
        if replacement is None or word.start() in protected:
            continue
        adjacent = []
        if index:
            previous = words[index - 1]
            if previous[0].lower() == 'and' and index > 1:
                previous = words[index - 2]
            if previous.start() not in protected and re.fullmatch(
                    r'[\s-]+(?:and[\s-]+)?', text[previous.end():word.start()], re.IGNORECASE):
                adjacent.append(previous[0].lower())
        if index + 1 < len(words):
            following = words[index + 1]
            if following[0].lower() == 'and' and index + 2 < len(words):
                following = words[index + 2]
            if following.start() not in protected and re.fullmatch(
                    r'[\s-]+(?:and[\s-]+)?', text[word.end():following.start()], re.IGNORECASE):
                adjacent.append(following[0].lower())
        if not any(neighbor in _COMPOSITE_WORDS for neighbor in adjacent):
            replacements.append((word.start(), word.end(), replacement))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def _matches_whole_fractions(candidate: str, actual: str) -> bool:
    """Text containment may include a fraction, but cannot cut through one."""
    fractions = []
    for fraction in _FRACTION.finditer(actual):
        # Use the shared normalization, including its article/punctuation rules.
        end = len(normalize(actual[:fraction.end()]))
        fractions.append((end - len(normalize(fraction[0])), end))
    if not fractions:
        return True
    needle = normalize(candidate)
    if not needle:
        return False
    for match in re.finditer(rf'(?<!\S){re.escape(needle)}(?!\S)', normalize(actual)):
        if not any(start < boundary < end for start, end in fractions
                   for boundary in match.span()):
            return True
    return False


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
    numeric_actual = _FRACTION.sub(' ', actual)
    for candidate in [expected, *aliases]:
        numeric = re.fullmatch(r'[+-]?\d+(?:\.\d+)?', candidate.strip().replace(',', ''))
        normalized_row = {**row, 'answer': candidate, 'answer_aliases': []}
        if score_row(normalized_row, numeric_actual if numeric else actual) and (
                numeric or dates_in(candidate) or _matches_whole_fractions(candidate, actual)):
            return True
    return False
