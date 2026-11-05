"""Compare measured model usage for the same dialogue with two context policies."""

import json
import math
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def _usage(record):
    """Require an actual token count and a known charge for every measurement."""
    for field in ("prompt_tokens", "completion_tokens"):
        if type(record[field]) is not int or record[field] < 0:
            raise ValueError(f"Invalid {field} in measured model usage")
    cost = record["cost_usd"]
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("Unknown or invalid model cost")
    return {field: record[field] for field in ("prompt_tokens", "completion_tokens", "cost_usd")}


def _total(records):
    return {
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "completion_tokens": sum(record["completion_tokens"] for record in records),
        "cost_usd": math.fsum(record["cost_usd"] for record in records),
    }


def _plot(turns, path):
    fig = Figure(figsize=(9, 4.4), layout="constrained")
    ax = fig.subplots()
    numbers = list(range(1, len(turns) + 1))
    for key, label, color, marker in [
        ("full", "Вся история", "#355C9A", "o"),
        ("window", "Окно и найденные факты", "#AD4C14", "s"),
    ]:
        ax.plot(numbers, [r[key]["prompt_tokens"] for r in turns], label=label,
                color=color, marker=marker, markersize=4.5, linewidth=2)
    boundaries = [i for i in range(1, len(turns)) if turns[i - 1]["session_id"] != turns[i]["session_id"]]
    for i in boundaries:
        ax.axvline(i + 0.5, color="#939CA5", linestyle=":", linewidth=1.5,
                   label="Смена сессии" if i == boundaries[0] else None)
    ax.set_title("Сколько токенов агент получает при каждом вопросе", loc="left", pad=13)
    ax.set(xlabel="Вопрос по порядку", ylabel="Входные токены", ylim=(0, None))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(len(turns), 15)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)


def write_comparison(turns, extractions, out_dir, *, model, price_basis, reference_rates=None):
    """Save per-turn usage, an end-to-end cost breakdown, and the input-token chart.

    full and window refer to the replies to the same ordered questions. All
    model calls needed to answer a question count toward that turn. Extracting
    semantic facts at session end is a separate cost of the window policy.
    Optional reference rates are hypothetical USD per million input/output tokens.
    """
    turns = [
        {"session_id": t["session_id"], "question": t["question"],
         "full": _usage(t["full"]), "window": _usage(t["window"])}
        for t in turns
    ]
    if not turns:
        raise ValueError("A comparison needs a measured dialogue")
    extractions = [{"session_id": e["session_id"], **_usage(e)} for e in extractions]
    sessions = {t["session_id"] for t in turns}
    if len(extractions) != len(sessions) or {e["session_id"] for e in extractions} != sessions:
        raise ValueError("Expected measured extraction usage for every session")

    answers = _total([t["window"] for t in turns])
    extraction = _total(extractions)
    totals = {
        "full": _total([t["full"] for t in turns]),
        "window_answers": answers,
        "extractions": extraction,
        "window": _total([answers, extraction]),
    }
    report = {"model": model, "price_basis": price_basis, "turns": turns,
              "extractions": extractions, "totals": totals}
    if reference_rates is not None:
        input_rate, output_rate = reference_rates
        if any(type(rate) not in (float, int) or not math.isfinite(rate) or rate < 0
               for rate in reference_rates):
            raise ValueError("Expected non-negative reference rates")
        report["hypothetical_pricing"] = {
            "usd_per_million_input_tokens": input_rate, "usd_per_million_output_tokens": output_rate,
            "estimated_usd": {
                key: (row["prompt_tokens"] * input_rate + row["completion_tokens"] * output_rate) / 1_000_000
                for key, row in totals.items()
            },
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(turns, out_dir / "input_tokens.png")
    (out_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
