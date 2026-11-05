"""Build two Milvus collections and compare vector, BM25 and RRF retrieval."""
import argparse
import hashlib
from pathlib import Path

from freshqa.dataset import load_rows
from hw2.embeddings import MODEL_ID, MODEL_REVISION, encode_texts
from hw2.evaluate_retrieval import CANDIDATE_DEPTH, evaluate_from_rankings, save_report
from hw2.prepare_corpus import DATA
from hw2.vector_store import DEFAULT_URI, index_chunks, open_store, search_many


RESULTS = DATA.parent / "results"
DEMO_PAGE = "2026 in Canada"
DEMO_QUERY = "In the Canada timeline, what did Statistics Canada announce about jobs?"


def filter_demo(client, chunks):
    vectors = encode_texts([DEMO_QUERY], role="query")
    ids, = search_many(client, "structure", vectors, limit=3, page_filter=DEMO_PAGE)
    by_id = {chunk["id"]: chunk for chunk in chunks}
    hits = [by_id[chunk_id] for chunk_id in ids]
    if len(hits) != 3 or any(row["page"] != DEMO_PAGE for row in hits):
        raise RuntimeError("Фильтр Milvus по статье вернул неожиданный результат.")
    fields = ("id", "page", "section", "url", "text")
    return {"query": DEMO_QUERY, "page_filter": DEMO_PAGE,
            "hits": [{field: row[field] for field in fields} for row in hits]}


def run(uri, output_dir=RESULTS):
    questions = load_rows(DATA / "questions.jsonl")
    queries = encode_texts([row["question"] for row in questions], role="query")
    report = {
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "dimension": queries.shape[1],
        "backend": "Milvus Lite" if uri.endswith(".db") else "Milvus Standalone",
        "index": "FLAT", "metric": "COSINE", "query_count": len(questions),
        "input_sha256": {}, "methods": {},
    }
    for filename in ("corpus.jsonl", "questions.jsonl", "chunks_char.jsonl", "chunks_structure.jsonl"):
        report["input_sha256"][filename] = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
    client = open_store(uri)
    try:
        for method in ("char", "structure"):
            chunks = load_rows(DATA / f"chunks_{method}.jsonl")
            print(f"Индекс {method}, {len(chunks)} фрагментов", flush=True)
            vectors = encode_texts([row["text"] for row in chunks], role="passage")
            index_chunks(client, method, chunks, vectors)
            rankings = search_many(client, method, queries, limit=CANDIDATE_DEPTH)
            result = evaluate_from_rankings(questions, chunks, rankings)
            report["methods"][method] = result
            for name, metrics in result["metrics"].items():
                print(f"  {name}: {metrics['hits']} из {len(questions)}", flush=True)
            if method == "structure":
                report["filter_demo"] = filter_demo(client, chunks)
    finally:
        client.close()
    save_report(report, Path(output_dir))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help="Локальный файл .db (Milvus Lite) или адрес Milvus Standalone")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    run(args.uri, args.output_dir)


if __name__ == "__main__":
    main()
