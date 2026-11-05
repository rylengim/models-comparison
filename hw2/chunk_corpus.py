"""Build character and section-aware JSONL chunks: python3 hw2/chunk_corpus.py."""

import argparse
import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path
from tempfile import TemporaryDirectory


DEFAULT_CORPUS = Path(__file__).resolve().parent / "data" / "corpus.jsonl"
CHUNK_SIZE = 600
OVERLAP = 100
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def sections_and_lines(text):
    """Keep source offsets and the heading hierarchy for every article."""
    sections = [(0, "")]
    lines = []
    headings = []
    for match in re.finditer(r"[^\r\n]+", text):
        line = match.group()
        stripped = line.strip()
        if not stripped:
            continue
        heading = HEADING.fullmatch(stripped)
        if heading:
            level = len(heading[1])
            headings = [(n, title) for n, title in headings if n < level]
            headings.append((level, heading[2].strip()))
            sections.append((match.start(), " / ".join(title for _, title in headings)))
        else:
            start = match.start() + len(line) - len(line.lstrip())
            lines.append((start, start + len(stripped)))
    return sections, lines


def character_spans(text):
    for start in range(0, len(text), CHUNK_SIZE - OVERLAP):
        end = min(start + CHUNK_SIZE, len(text))
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break


def structured_spans(text, lines):
    """A source line is usually a timeline entry. Split long entries at spaces."""
    for start, end in lines:
        while end - start > CHUNK_SIZE:
            limit = start + CHUNK_SIZE
            boundary = text.rfind(" ", start + 1, limit + 1)
            stop = boundary if boundary != -1 else limit
            yield start, stop
            start = stop
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            yield start, end


def chunks(document, method):
    text = document["text"]
    sections, lines = sections_and_lines(text)
    offsets = [offset for offset, _ in sections]
    doc_id = hashlib.sha256(document["url"].encode("utf-8")).hexdigest()[:20]
    if method == "char":
        spans = character_spans(text)
    elif method == "structure":
        spans = structured_spans(text, lines)
    else:
        raise ValueError(f"Unknown chunking method: {method}")
    for start, end in spans:
        section = sections[bisect_right(offsets, start) - 1][1]
        yield {
            "id": f"{doc_id}:{method}:{start}",
            "page": document["page"],
            "url": document["url"],
            "section": section,
            "start": start,
            "end": end,
            "text": text[start:end],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    corpus = parser.parse_args().corpus
    methods = ("char", "structure")
    counts = dict.fromkeys(methods, 0)
    # Parse the complete input before replacing any previous result.
    with TemporaryDirectory(dir=corpus.parent) as temp_dir:
        paths = {method: Path(temp_dir) / f"chunks_{method}.jsonl" for method in methods}
        with (corpus.open(encoding="utf-8") as source,
              paths["char"].open("w", encoding="utf-8") as char_file,
              paths["structure"].open("w", encoding="utf-8") as structure_file):
            for line in source:
                if line.strip():
                    document = json.loads(line)
                    for method, output in zip(methods, (char_file, structure_file)):
                        for chunk in chunks(document, method):
                            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            counts[method] += 1
        for method, path in paths.items():
            path.replace(corpus.parent / path.name)
            print(f"{path.name}: {counts[method]} фрагментов")


if __name__ == "__main__":
    main()
