"""Measure BM25 retrieval without API calls: python3 -m hw2.lexical_baseline."""
import math
from collections import Counter, defaultdict

from freshqa.dataset import load_rows
from hw2.prepare_corpus import DATA, normalize_evidence


KS = (1, 3, 5, 10)


def bm25_index(chunks, k1=1.2, b=0.75):
    """Precompute BM25 term weights for this chunk collection."""
    postings = defaultdict(list)
    lengths = []
    for index, chunk in enumerate(chunks):
        tokens = normalize_evidence(chunk["text"]).split()
        lengths.append(len(tokens))
        for word, frequency in Counter(tokens).items():
            postings[word].append((index, frequency))

    count = len(lengths)
    if not postings:
        return {}
    average = sum(lengths) / count
    for word, entries in postings.items():
        idf = math.log1p((count - len(entries) + 0.5) / (len(entries) + 0.5))
        weighted = []
        for index, frequency in entries:
            denominator = frequency + k1 * (1 - b + b * lengths[index] / average)
            weighted.append((index, idf * frequency * (k1 + 1) / denominator))
        postings[word] = weighted
    return postings


def rank(question, postings):
    scores = Counter()
    for word in dict.fromkeys(normalize_evidence(question).split()):
        for index, weight in postings.get(word, ()):
            scores[index] += weight
    # The source order gives exact score ties a deterministic rank.
    return sorted(scores, key=lambda index: (-scores[index], index))


def evaluate(questions, chunks):
    postings = bm25_index(chunks)
    by_url = defaultdict(list)
    for index, chunk in enumerate(chunks):
        body = " " + normalize_evidence(chunk["text"]) + " "
        by_url[chunk["url"]].append((index, body))

    hits = dict.fromkeys(KS, 0)
    covered = 0
    example = None
    for question in questions:
        quote = normalize_evidence(question["evidence"])
        needle = f" {quote} "
        gold = {i for i, body in by_url[question["url"]] if quote and needle in body}
        if not gold:
            continue
        covered += 1
        for position, index in enumerate(rank(question["question"], postings), 1):
            if index in gold:
                for k in KS:
                    hits[k] += position <= k
                if example is None:
                    example = (question["id"], chunks[index]["id"], position)
                break
    return covered, hits, example


def main():
    questions = load_rows(DATA / "questions.jsonl")
    if not questions:
        raise SystemExit("Файл вопросов пуст.")
    for method in ("char", "structure"):
        chunks = load_rows(DATA / f"chunks_{method}.jsonl")
        covered, hits, example = evaluate(questions, chunks)
        total = len(questions)
        print(f"BM25 {method}: полная цитата есть во фрагменте для {covered}/{total} вопросов.")
        for k in KS:
            print(f"  recall@{k:<2} = {hits[k]}/{total} = {hits[k] / total:.3f}")
        if example is not None:
            question_id, chunk_id, position = example
            print(f"  {question_id}: первый верный фрагмент {chunk_id}, позиция {position}.")


if __name__ == "__main__":
    main()
