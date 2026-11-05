"""Score Milvus, BM25 and reciprocal rank fusion against full source quotes."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from hw2.lexical_baseline import KS, bm25_index, rank
from hw2.prepare_corpus import normalize_evidence


RRF_CONSTANT = 60
CANDIDATE_DEPTH = 100


def reciprocal_rank_fusion(*rankings):
    """Use 1-based ranks, with stable document-ID ordering for tied totals."""
    scores = Counter()
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1 / (RRF_CONSTANT + position)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


def evaluate_from_rankings(questions, chunks, vector_rankings, *, depth=CANDIDATE_DEPTH):
    """Evaluate one chunking scheme; the query order must match the Milvus result."""
    if not questions or not chunks:
        raise ValueError("Нужны непустые выборки вопросов и фрагментов.")
    if len(vector_rankings) != len(questions):
        raise ValueError("Для каждого вопроса требуется отдельная векторная выдача.")
    if depth < max(KS):
        raise ValueError(f"Глубина поиска должна быть не меньше {max(KS)}.")

    by_url = defaultdict(list)
    chunk_ids = {chunk["id"] for chunk in chunks}
    for chunk in chunks:
        body = " " + normalize_evidence(chunk["text"]) + " "
        by_url[chunk["url"]].append((chunk["id"], body))
    postings = bm25_index(chunks)
    hits = {name: dict.fromkeys(KS, 0) for name in ("vector", "bm25", "rrf")}
    details = []
    for question, vector in zip(questions, vector_rankings, strict=True):
        vector = vector[:depth]
        if len(vector) != len(set(vector)) or not chunk_ids.issuperset(vector):
            raise ValueError(f"{question['id']}: векторная выдача содержит повтор или посторонний ID.")
        indices = rank(question["question"], postings)
        bm25 = [chunks[i]["id"] for i in indices[:depth]]
        fused = reciprocal_rank_fusion(vector, bm25)
        quote = normalize_evidence(question["evidence"])
        gold = {chunk_id for chunk_id, body in by_url[question["url"]]
                if quote and f" {quote} " in body}
        results = {}
        for method, ranking in (("vector", vector), ("bm25", bm25), ("rrf", fused)):
            position, match = next(
                ((n, chunk_id) for n, chunk_id in enumerate(ranking, 1) if chunk_id in gold),
                (None, None),
            )
            if position is not None:
                for k in KS:
                    hits[method][k] += position <= k
            results[method] = {"rank": position, "first_gold_id": match,
                               "top_id": ranking[0] if ranking else None}
        details.append({"id": question["id"], "gold_ids": sorted(gold), "results": results})

    total = len(questions)
    metrics = {method: {"hits": {str(k): count for k, count in counts.items()},
                        "recall": {str(k): count / total for k, count in counts.items()}}
               for method, counts in hits.items()}
    return {
        "question_count": total, "chunk_count": len(chunks),
        "gold_coverage": sum(bool(row["gold_ids"]) for row in details),
        "depth": depth, "rrf_constant": RRF_CONSTANT, "metrics": metrics, "questions": details,
    }


def save_report(report, output_dir):
    """Save the measurements and plot both vector recall curves on the same axes."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = report["methods"]
    total = methods["structure"]["question_count"]
    fig = Figure(figsize=(8, 4.8), layout="constrained")
    ax = fig.subplots()
    ax.set_title("Как часто векторный поиск находит цитату", loc="left", weight="bold", pad=14)
    for method, label, color, marker in (
        ("char", "По символам", "#694c97", "s"),
        ("structure", "По структуре", "#007e77", "o"),
    ):
        scores = methods[method]["metrics"]["vector"]["recall"]
        values = [scores[str(k)] for k in KS]
        ax.plot(KS, values, color=color, marker=marker, linewidth=2.5, markersize=7, label=label)
    ax.set(xlabel="Число первых фрагментов (k)",
           ylabel=f"Доля вопросов с найденной цитатой (из {total})",
           xticks=KS, ylim=(0, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", color="#d5dce0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(output_dir / "recall_at_k.png", dpi=160)
    (output_dir / "retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
