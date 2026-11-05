"""Repeat two recorded local Qwen completions against the live homework retriever."""
import argparse
import json

from freshqa.dataset import load_rows
from hw2.knowledge_base import KnowledgeBase
from hw2.prepare_corpus import DATA
from hw2.rag import rag_answer
from hw2.vector_store import DEFAULT_URI


class RecordedClient:
    """Replay only the recorded question and its original ordered retrieval IDs."""

    def __init__(self, cases):
        self.cases = {case["question"]: case for case in cases}

    def complete(self, *, model, messages):
        prompt = json.loads(messages[-1]["content"])
        case = self.cases[prompt["question"]]
        if [s["id"] for s in prompt["sources"]] != case["retrieved_ids"]:
            raise ValueError("Выдача изменилась с момента записи ответа модели.")
        return {"message": {"role": "assistant", "content": json.dumps(case["completion"])},
                "finish_reason": "stop", **case["usage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI, help="Локальный адрес Milvus или файл Milvus Lite")
    args = parser.parse_args()
    total = next(r for r in load_rows(DATA / "questions.jsonl") if r["id"] == "fresh-113")
    division = next(r for r in load_rows(DATA / "unanswerable.jsonl") if r["id"] == "unknown-7")
    recorded = json.loads((DATA.parent / "results" / "rag_smoke.json").read_text(encoding="utf-8"))
    client = RecordedClient(recorded["cases"])
    print("Поиск выполняется в локальном Milvus. Используются ответы ранее запущенной локальной Qwen.")
    with KnowledgeBase(args.uri) as kb:
        answered = rag_answer(total["question"], knowledge_base=kb, client=client, model=recorded["model"])
        if (answered["status"] != "answered" or answered["answer"] != total["answer"]
                or answered["source"]["url"] != total["url"] or answered["quote"] not in total["evidence"]):
            raise RuntimeError(f"Ответ не совпал с эталоном {total['id']}. Получен {answered!r}")
        print(f"\n{total['question']}\n{answered['answer']}\n{answered['quote']}\n{answered['source']['url']}")

        refused = rag_answer(division["question"], knowledge_base=kb, client=client, model=recorded["model"])
        if (refused["status"] != "insufficient_evidence" or refused["answer"] != division["answer"]
                or answered["source"]["id"] not in refused["retrieved_ids"]):
            raise RuntimeError(f"Не пройдена проверка отказа с тематическим контекстом: {refused!r}")
        print(f"\n{division['question']}\n{refused['answer']} "
              "— найден тот же фрагмент с общей численностью, разбивка по ИТ в нём отсутствует.")

        page = total["page"]
        hits = kb.search(total["question"], page=page)
        if len(hits) != 3 or any(row["page"] != page for row in hits):
            raise RuntimeError(f"Не пройдена проверка фильтра по статье {page}.")
        print(f"\nФильтр по статье {page}")
        for position, row in enumerate(hits, 1):
            print(f"{position}. {row['page']} — {row['section'] or '(введение)'} — {row['id']}")


if __name__ == "__main__":
    main()
