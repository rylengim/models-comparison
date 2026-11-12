"""Regrade frozen inference records offline using explicit, source-backed gold fixes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .report import _load_records, _summarize, render_report

ROOT = Path(__file__).resolve().parents[1]


def apply_overrides(rows, overrides):
    """Copy gold rows and replace only the four fields approved for offline grading."""
    ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if not ids or any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Gold rows must have nonempty unique IDs")
    if not isinstance(overrides, list):
        raise ValueError("Overrides must be a list")
    corrected = deepcopy(rows)
    by_id, seen = {row["id"]: row for row in corrected}, set()
    for patch in overrides:
        if not isinstance(patch, dict) or not isinstance(patch.get("id"), str):
            raise ValueError("Override requires a string id")
        qid = patch["id"]
        if qid in seen:
            raise ValueError(f"Duplicate override: {qid}")
        seen.add(qid)
        if qid not in by_id:
            raise ValueError(f"Unknown override id: {qid}")
        if "old_answer" not in patch or patch["old_answer"] != by_id[qid].get("answer"):
            raise ValueError(f"old_answer mismatch: {qid}")
        if any(not isinstance(patch.get(key), str) or not patch[key].strip()
               for key in ("answer", "evidence", "source_url")):
            raise ValueError(f"Override requires answer, evidence and source_url: {qid}")
        aliases = patch.get("answer_aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError(f"Invalid answer aliases: {qid}")
        by_id[qid].update(answer=patch["answer"], answer_aliases=deepcopy(aliases),
                         evidence=patch["evidence"], url=patch["source_url"])
    return corrected


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_report(result_dir, rows):
    root = Path(result_dir)
    overrides = json.loads((root / "grading_overrides.json").read_text(encoding="utf-8"))
    corrected = apply_overrides(rows, overrides)
    out = root / "audited"
    out.mkdir(parents=True, exist_ok=True)
    if (root / "manual_audit.md").is_file():
        (out / "manual_audit.md").write_text(
            "# Проверка проверки\n\n[Полный аудит с ответами, источниками и решениями](../manual_audit.md).\n",
            encoding="utf-8")
    else:
        (out / "manual_audit.md").unlink(missing_ok=True)
    summary = render_report(root, corrected, out)
    review_path = root / "manual_reviews.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else []
    reviews = {(r["config"], r["id"]): r for r in reviews}
    candidates = json.loads((out / "audit_candidates.json").read_text(encoding="utf-8"))
    for candidate in candidates:
        review = reviews.get((candidate["config"], candidate["id"]))
        if review is not None:
            candidate.update(audit_status=review["judgement"], audit_note=review.get("note", ""))
    _write_json(out / "audit_candidates.json", candidates)
    note = ["## Аудированная оценка", "",
            "Пересчёт выполнен локально без новых вызовов модели. [Исходный отчёт](../report.md) сохранён. "
            "Замороженные вопросы, ответы и HTTP-учёт не изменены. Исходный и окончательный ответы критика "
            "переоценены по одному и тому же исправленному эталону во всех конфигурациях.", "",
            "Исправления заданы в [grading_overrides.json](../grading_overrides.json):"]
    note += [f"- `{p['id']}`: {p['old_answer']} → {p['answer']}; [подтверждающий источник]({p['source_url']})."
             for p in overrides]
    report = out / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n" + "\n".join(note) + "\n", encoding="utf-8")
    original_gold = {row["id"]: row for row in rows}
    original = [_summarize(config, records, original_gold)
                for config, records in _load_records(root, original_gold).items()]
    _write_json(out / "grading_summary.json", {"original": original, "audited": summary,
                                               "applied_overrides": overrides})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hw3/results/main")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "hw3/data/questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    build_report(args.out, rows)
    print(args.out / "audited/report.md")


if __name__ == "__main__":
    main()
