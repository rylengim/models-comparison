"""Compatible entry point; format checking never imports the paid collector."""
import argparse
import json
from pathlib import Path

from freshqa.__main__ import main


def check(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/fresh.jsonl")
    parser.add_argument("--sonnet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=5)
    args = parser.parse_args(argv)
    code = main(["validate", args.path])
    if code or not args.sonnet:
        return code
    command = ["benchmark", "--data", args.path, "--configs", "strong_no_tools",
               "--budget-usd", str(args.budget_usd)]
    if args.resume:
        command.append("--resume")
    code = main(command)
    if code:
        return code
    result = json.loads(Path("reports/freshness.json").read_text(encoding="utf-8"))
    print(f"Sonnet без инструментов: {result['correct']}/{result['total']}, {result['accuracy']:.1%}")
    print("Критерий свежести " + ("пройден" if result["passed"] is True else "НЕ пройден или использована другая модель"))
    print("Угаданные id:", ", ".join(result["guessed_ids"]) or "нет")
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(check())
