"""Bounded English cardinal quantities for deterministic offline evaluation.

Parse whole expressions before punctuation normalization. Fractions, ranges,
scientific notation, and unrecognised units have no inferred equivalences: only
their complete literal spelling can match (or an explicitly declared alias).
This is a bounded lexical metric, not a judge of negation or factual relevance.
"""
import re
from decimal import Decimal, localcontext

SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve '
                 'thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
SCALES = {'thousand': 1000, 'million': 10**6, 'billion': 10**9}
DIGITS = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
# Consume unsupported numeric forms whole, rather than retrying a smaller digit
# token inside them. Attached units are literal-only (e.g. 8.15m).
NUMERIC_TOKEN = r'(?:[+−-]\s*)?\d+(?:[,.]\d+)*(?:[eE][+−-]?\d+)?(?:[a-zA-Z]+(?:[/^][a-zA-Z0-9]+)*)?'
FRACTIONS = {'half', 'halves', 'quarter', 'quarters', 'third', 'fourth', 'fifth',
             'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'hundredth',
             'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
             'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth',
             'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth',
             'eightieth', 'ninetieth', 'thousandth', 'millionth', 'billionth'}
WORDS = (set(SMALL) | set(TENS) | set(SCALES) | FRACTIONS
         | {word + 's' for word in FRACTIONS} | {'hundred', 'minus', 'trillion'})
TOKEN = re.compile(r'(?<![\w.])(?:' + NUMERIC_TOKEN + '|[a-z]+illion(?:th)?s?|'
                   + '|'.join(sorted(WORDS, key=len, reverse=True))
                   + r')(?!\w|\.\d)', re.I)
GAP = re.compile(r'(?:[\s/:*×^+-]+|[\s-]+(?:and[\s-]+)?(?:a|point)[\s-]+'
                 r'|[\s-]+(?:and|to|through|plus|times|over)[\s-]+)', re.I)
UNITS = {
    'person': 'people', 'persons': 'people', 'people': 'people',
    **{word + suffix: word for word in ('game', 'year', 'month', 'day', 'hour', 'minute', 'second', 'mile')
       for suffix in ('', 's')},
    'km/s': 'km/s', 'm/s': 'm/s', 'km': 'km', 'm': 'm',
    '°c': '°c', '°f': '°f', '%': '%', 'percent': '%', 'lira': 'lira',
}
UNIT_END = r'(?!\w|\s*[/^²³]|\s+per\b)'
UNIT = re.compile(r'\s*(' + '|'.join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + ')' + UNIT_END, re.I)
MAX_NUMBER_WORDS = 64
MAX_NUMBER_CHARS = 256
DASHES = str.maketrans({char: '-' for char in '‐‑‒–—−'})
CURRENCY = re.compile(r'^([$£€₹¥])\s*')


def _group(words):
    if not words:
        return None
    if len(words) == 1:
        word = words[0]
        if re.fullmatch(DIGITS, word):
            return Decimal(word.replace(',', ''))
        value = SMALL.get(word, TENS.get(word))
        return Decimal(value) if value is not None else None
    if 'hundred' in words:
        index = words.index('hundred')
        prefix = _group(words[:index])
        rest = words[index + 1:]
        if rest[:1] == ['and']:
            rest = rest[1:]
            if not rest:
                return None
        suffix = _group(rest) if rest else Decimal(0)
        if prefix is not None and suffix is not None and 1 <= prefix < 100 and prefix == int(prefix) and 0 <= suffix < 100:
            return prefix * 100 + suffix
        return None
    if len(words) == 2 and words[0] in TENS and words[1] in SMALL and 1 <= SMALL[words[1]] <= 9:
        return Decimal(TENS[words[0]] + SMALL[words[1]])
    return None


def number_value(expression):
    expression = expression.strip().lower().translate(DASHES)
    if len(expression) > MAX_NUMBER_CHARS or len(expression.split()) > MAX_NUMBER_WORDS:
        return None
    expression = re.sub(r'^([+-])\s+', r'\1', expression)
    # More precision than this bounded input plus the largest supported scale:
    # Decimal's default 28 digits must not round two distinct integers together.
    with localcontext() as context:
        context.prec = MAX_NUMBER_CHARS + 12
        return _number_value(expression)


def _number_value(expression):
    if re.fullmatch(DIGITS, expression):
        return Decimal(expression.replace(',', ''))
    # Hyphens join English number words, never digit ranges.
    if re.search(r'\d\s*[-–/]\s*\d', expression):
        return None
    words = re.split(r'[\s-]+', expression)
    sign = -1 if words[:1] == ['minus'] else 1
    if sign == -1:
        words = words[1:]
    total, group, previous_scale = Decimal(0), [], 10**12
    for word in words:
        if word not in SCALES:
            group.append(word)
            continue
        value = _group(group)
        scale = SCALES[word]
        if value is None or value < 0 or scale >= previous_scale:
            return None
        total += value * scale
        previous_scale, group = scale, []
    if total and group[:1] == ['and']:
        group = group[1:]
        if not group:
            return None
    remainder = _group(group) if group else Decimal(0)
    if remainder is None or (total and not 0 <= remainder < previous_scale):
        return None
    return sign * (total + remainder) if words else None


def _spans(text):
    """Maximal expressions protect compound/fraction components from matching."""
    tokens = iter(TOKEN.finditer(text))
    first = next(tokens, None)
    if first is None:
        return
    start, end = first.span()
    for token in tokens:
        if GAP.fullmatch(text[end:token.start()]):
            end = token.end()
        else:
            yield start, end, number_value(text[start:end])
            start, end = token.span()
    yield start, end, number_value(text[start:end])


def _spaced_thousands(text):
    return re.sub(r'(?<=\d)[ \u00a0](?=\d{3}(?:\D|$))', '', text.translate(DASHES))


def _literal(text):
    return ' '.join(text.lower().split())


def _literal_quantity(expected, actual, expected_end, currency=None):
    """No punctuation erasure or component matching for unsupported syntax."""
    prefix = _literal(expected[:expected_end])
    suffix = expected[expected_end:].strip()
    tail = re.compile(r'\s*' + r'\s+'.join(map(re.escape, suffix.split())) + UNIT_END, re.I) if suffix else None
    for start, end, _ in _spans(actual):
        if currency and not actual[:start].rstrip().endswith(currency):
            continue
        if _literal(actual[start:end]) == prefix and (tail is None or tail.match(actual[end:])):
            return True
    return False


def quantity_matches(expected, actual, *, allow_count_omission=False):
    """None means no leading numeric expression; other gold gets a guarded match.

    Unknown quantity syntax/units and names starting with a number retain only
    literal matching. They cannot bypass numeric guards via punctuation removal.
    """
    expected, actual = _spaced_thousands(expected), _spaced_thousands(actual)
    currency = CURRENCY.match(expected)
    if currency:
        expected = expected[currency.end():]
    first = next(_spans(expected), None)
    if first is None or first[0] != 0:
        return None
    _, end, value = first
    suffix = expected[end:].strip().lower()
    if currency or value is None or (suffix and suffix not in UNITS):
        return _literal_quantity(expected, actual, end, currency=currency[1] if currency else None)
    unit = UNITS.get(suffix)
    for start, end, candidate in _spans(actual):
        if candidate != value:
            continue
        if unit is None:
            return True
        found = UNIT.match(actual[end:])
        if found and UNITS[found[1].lower()] == unit:
            return True
        if allow_count_omission and unit == 'people' and not actual[:start].strip() and not actual[end:].strip(' .!'):
            return True
    return False
