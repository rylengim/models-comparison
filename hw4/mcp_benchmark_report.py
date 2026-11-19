"""Evaluate and report FreshQA MCP benchmark traces."""

import json
import math
import re

from freshqa.dataset import normalize
from hw3.evaluation import score

from .quote_evidence import quote_id_evidence_matches_trace


_TELEMETRY = (
    "stub_steps", "tool_calls", "input_tokens_estimate", "cached_tokens_estimate",
    "model_calls", "cost_usd", "simulated_cost_usd",
)
_PAID_USAGE = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
_PAID_TELEMETRY = ("tool_calls", "model_calls", "cost_usd", "known_cost_usd", *_PAID_USAGE)
_CHECKS = ("answer_correct", "source_matches", "evidence_found", "verified_correct")
_SKIPPED = {"run_budget_exceeded", "accounting_uncertain"}
_TERMINAL_CITATION = re.compile(
    r'(?:\s+[—–-]\s+|(?<=[.!?])\s+)'
    r'(?:(?P<attribution>the (?:announcement|article|press release|report|source) (?:states|says|notes)'
    r'|as (?:stated|reported|noted) in the (?:announcement|article|press release|report|source)):\s*)?'
    r'(?P<emphasis>\*{1,2}|_{1,2})?(?P<quoted>"[^"\r\n]+"|“[^”\r\n]+”)'
    r'(?(emphasis)(?P=emphasis))\s*[.!]?\s*\Z',
    re.IGNORECASE,
)
_EXPLANATORY_ASIDE = re.compile(
    r'\s+\((?:a|an|the)\s+[^()\r\n]*\bor\b[^()\r\n]*\)'
    r'(?=\s+(?:is|are|was|were|has|have|had|would|could|will|can|may|might|must)\b)',
    re.IGNORECASE,
)


def _nonnegative_int(value):
    return type(value) is int and 0 <= value <= 2**63 - 1


def _nonnegative_amount(value):
    return _nonnegative_int(value) or (type(value) is float and 0 <= value <= 2**63 - 1)


def _page_contains_evidence(events, url, evidence):
    expected = " ".join(evidence.split())
    for event in events:
        if event.get("type") != "tool" or event.get("name") != "page_find":
            continue
        args = event.get("arguments")
        if not isinstance(args, dict) or args.get("url") != url:
            continue
        try:
            page = json.loads(event.get("output"))
        except (TypeError, ValueError, RecursionError):
            continue
        if (not isinstance(page, dict) or "error" in page
                or page.get("found") is not True or page.get("url") != url):
            continue
        excerpts = page.get("excerpts")
        if isinstance(excerpts, list) and any(
            isinstance(excerpt, str) and expected in " ".join(excerpt.split()) for excerpt in excerpts
        ):
            return True
    return False


def _paid_answer(row, final):
    """Extract the answer from a paid response with a citation or an explanatory aside."""
    quote = _TERMINAL_CITATION.search(final)
    if quote and not quote["attribution"]:
        evidence = normalize(row["evidence"])
        if not evidence or f" {evidence} " not in f" {normalize(quote['quoted'])} ":
            quote = None
    if quote is None:
        escaped = r"\s+".join(re.escape(word) for word in row["evidence"].split())
        quote = re.search(rf'(?:"{escaped}"|“{escaped}”)\s*[.!]?\s*\Z', final)
    if quote:
        answer = final[:quote.start()].rstrip(" \t\n\r:;.,—–-")
        if any(char.isalnum() for char in answer):
            final = answer
    for aside in reversed(list(_EXPLANATORY_ASIDE.finditer(final))):
        before, after = final[:aside.start()], final[aside.end():]
        # Keep alternatives adjacent to an already-stated answer in scoring scope.
        if score(row, after) and not score(row, before):
            final = before + after
    return final


def _stub_telemetry(trace, cache, model_calls, cost):
    if (any(trace.get(key, "stub") != "stub" for key in ("provider", "cost_basis"))
            or model_calls != 0 or not _nonnegative_amount(cost) or cost != 0
            or cache.get("mode") != "simulated"):
        return None
    stub_steps, simulated_cost = trace.get("steps"), trace.get("simulated_cost_usd")
    prompt, cached = cache.get("input_tokens"), cache.get("cached_tokens")
    if (not _nonnegative_int(stub_steps) or not _nonnegative_amount(simulated_cost)
            or not _nonnegative_int(prompt) or not _nonnegative_int(cached) or cached > prompt):
        return None
    return {
        "stub_steps": stub_steps, "simulated_cost_usd": simulated_cost,
        "input_tokens_estimate": prompt, "cached_tokens_estimate": cached,
    }


def _paid_telemetry(trace, cache, cost):
    if (trace.get("provider") != "openrouter" or trace.get("cost_basis") != "provider_reported"
            or cache.get("mode") != "provider_reported"):
        return None
    known_cost = trace.get("known_cost_usd", cost)
    if ((cost is not None and not _nonnegative_amount(cost))
            or (known_cost is not None and not _nonnegative_amount(known_cost))
            or (cost is not None and cost != known_cost)):
        return None
    counts = {key: cache.get(key) for key in _PAID_USAGE}
    if any(value is not None and not _nonnegative_int(value) for value in counts.values()):
        return None
    prompt, cached = counts["prompt_tokens"], counts["cached_tokens"]
    if prompt is not None and cached is not None and cached > prompt:
        return None
    return {"known_cost_usd": known_cost, **counts}


def evaluate_case(row, trace, *, returncode, duration_s, trace_path=None, diagnostic=None, model="stub",
                  use_mcp=True):
    """Keep failed questions in the denominator and count only verified telemetry."""
    telemetry_fields = _TELEMETRY if model == "stub" else _PAID_TELEMETRY
    case = {
        "id": row["id"], "status": "missing_trace" if trace is None else "invalid_trace",
        "returncode": returncode, "duration_s": duration_s, "final": None, "source_url": None,
        "trace_path": str(trace_path) if trace_path is not None else None, "diagnostic": diagnostic,
        **dict.fromkeys(_CHECKS, False), **dict.fromkeys(telemetry_fields),
    }
    if not isinstance(trace, dict) or trace.get("model") != model:
        return case
    status, final, source = trace.get("status"), trace.get("final"), trace.get("source_url")
    events, cache = trace.get("events"), trace.get("prompt_cache")
    model_calls, cost = trace.get("model_calls"), trace.get("cost_usd")
    if (not isinstance(status, str) or not status or (returncode == 0) != (status == "ok")
            or not isinstance(final, str) or not final.strip() or not (source is None or isinstance(source, str))
            or not isinstance(events, list) or not all(isinstance(event, dict) for event in events)
            or not _nonnegative_int(model_calls) or not isinstance(cache, dict)):
        return case

    tool_calls = sum(event.get("type") == "tool" for event in events)
    if not use_mcp and (source is not None or tool_calls or model_calls > 1
                        or any(event.get("tools_enabled") for event in events)):
        return case

    telemetry = (_stub_telemetry(trace, cache, model_calls, cost) if model == "stub"
                 else _paid_telemetry(trace, cache, cost))
    if telemetry is None:
        return case

    case.update(
        status=status, final=final, source_url=source,
        tool_calls=tool_calls,
        model_calls=model_calls, cost_usd=cost, **telemetry,
    )
    if status == "ok":
        answer = _paid_answer(row, final) if model != "stub" and use_mcp else final
        case["answer_correct"] = score(row, answer)
        if use_mcp:
            case["source_matches"] = source == row["url"]
            case["evidence_found"] = case["source_matches"] and _page_contains_evidence(events, source, row["evidence"])
            case["verified_correct"] = all(case[key] for key in _CHECKS[:-1])
            if case["verified_correct"] and model != "stub":
                case["verified_correct"] = quote_id_evidence_matches_trace(trace)
    return case


def _summarize_paid(cases, totals, rates):
    attempted = [case for case in cases if case["status"] not in _SKIPPED]
    totals.update(attempted=len(attempted), skipped=len(cases) - len(attempted))
    means = {"duration_s": math.fsum(case["duration_s"] for case in attempted) / len(attempted) if attempted else None}
    coverage, known_totals = {}, {}
    for key in _PAID_TELEMETRY:
        available = [case[key] for case in attempted if case[key] is not None]
        coverage[key] = len(available)
        subtotal = math.fsum(available) if key.endswith("_usd") else sum(available)
        known_totals[key] = subtotal if available else None
        totals[key] = known_totals[key] if len(available) == len(attempted) else None
        means[key] = subtotal / len(available) if available else None
    # This subtotal remains useful when a different charge in the same trace is unknown.
    totals["known_cost_usd"] = known_totals["known_cost_usd"]
    comparable = [case for case in attempted if case["prompt_tokens"] is not None and case["cached_tokens"] is not None]
    prompt = sum(case["prompt_tokens"] for case in comparable)
    coverage["cached_input_fraction"] = len(comparable)
    totals["cached_input_fraction"] = sum(case["cached_tokens"] for case in comparable) / prompt if prompt else None
    return {
        "mode": "openrouter", "totals": totals, "means": means, "rates": rates, "cases": cases,
        "telemetry_coverage": coverage, "known_totals": known_totals,
    }


def summarize(cases, *, mode="stub"):
    cases = list(cases)
    questions = len(cases)
    completed = sum(case["status"] == "ok" for case in cases)
    traces = [case for case in cases if case["model_calls"] is not None]
    totals = {
        "questions": questions, "completed": completed, "errors": questions - completed,
        "unknown": sum(case["status"] == "ok" and case["final"].strip().casefold() == "unknown" for case in cases),
        "telemetry_questions": len(traces),
        **{key: sum(case[key] for case in cases) for key in _CHECKS},
    }
    rates = {key: totals[key] / questions if questions else None for key in _CHECKS}
    if mode == "openrouter":
        return _summarize_paid(cases, totals, rates)
    means = {"duration_s": math.fsum(case["duration_s"] for case in cases) / questions if questions else None}
    totals.update({key: sum(case[key] for case in traces) if traces else None for key in _TELEMETRY})
    if traces:
        totals["simulated_cost_usd"] = math.fsum(case["simulated_cost_usd"] for case in traces)
    input_tokens = totals["input_tokens_estimate"]
    totals["cached_input_fraction"] = totals["cached_tokens_estimate"] / input_tokens if input_tokens else None
    means.update({key: totals[key] / len(traces) if traces else None for key in _TELEMETRY})
    return {"mode": "stub", "totals": totals, "means": means, "rates": rates, "cases": cases}


def _display(value, format_spec=""):
    return "—" if value is None else format(value, format_spec)


def _paid_rows(summary):
    totals = summary["totals"]
    rows = [
        ("Модель", _display(summary.get("model"))),
        ("Лимит на вопрос, USD", _display(summary.get("question_budget_usd"), ".6f")),
        ("Бюджет прогона, USD", _display(summary.get("run_budget_usd"), ".6f")),
        ("Расход прогона по журналу, USD", _display(summary.get("run_cost_usd"), ".8f")),
        ("Остаток бюджета, USD", _display(summary.get("run_remaining_usd"), ".8f")),
    ]
    if summary.get("run_cost_usd") is None:
        rows.append(("Подтверждённая часть расходов по журналу, USD",
                     _display(summary.get("run_known_cost_usd"), ".8f")))
    if summary.get("stop_reason"):
        rows.append(("Причина остановки", f"`{summary['stop_reason']}`"))

    coverage = summary["telemetry_coverage"]
    for key, label in (
        ("model_calls", "Вызовы модели по трейсам"),
        ("tool_calls", "Вызовы инструментов по трейсам"),
        ("prompt_tokens", "Входные токены"),
        ("completion_tokens", "Выходные токены"),
        ("cached_tokens", "Токены чтения кэша"),
        ("cache_write_tokens", "Токены записи кэша"),
        ("cached_input_fraction", "Доля входных токенов из кэша"),
    ):
        value = totals[key]
        display = _display(value, ".1%" if key == "cached_input_fraction" else "")
        subtotal = summary["known_totals"].get(key)
        if value is None and subtotal is not None:
            display = f"{subtotal} (частично)"
        rows.append((label, f"{display} · {coverage[key]}/{totals['attempted']} вопросов"))
    return rows


def render_report(summary):
    totals, means, rates = summary["totals"], summary["means"], summary["rates"]
    questions = totals["questions"]
    paid = summary["mode"] == "openrouter"
    use_mcp = summary.get("use_mcp", True)
    skipped = totals["skipped"] if paid else 0
    rows = [
        ("Вопросы с ответом `ok`", f"{totals['completed']}/{questions}"),
        ("Технические сбои", totals["errors"] - skipped),
        ("Штатный `unknown`", totals["unknown"]),
    ]
    if paid:
        rows.append(("Пропущено после остановки", skipped))
    for key, label in zip(_CHECKS, ("Совпадение с эталоном", "Эталонный URL", "Эталонная цитата в `page_find`",
                                    "Ответ, URL и цитата совпали")):
        value = f"{totals[key]}/{questions} ({_display(rates[key], '.1%')})"
        rows.append((label, value if use_mcp or key == "answer_correct" else "—"))
    rows.extend([
        ("Среднее время команды с запуском MCP, с" if use_mcp else "Среднее время команды, с",
         _display(means["duration_s"], ".3f")),
        ("Вопросы с телеметрией", f"{totals['telemetry_questions']}/{questions}"),
    ])
    if paid:
        rows.extend(_paid_rows(summary))
        intro = ("Режим OpenRouter. В долях качества учитываются все вопросы набора. "
                 "Токены и вызовы берутся из трейсов, покрытие указано для вопросов, "
                 "которые скрипт пытался запустить. Расход прогона берётся из журнала.")
    else:
        rows += [
            ("Среднее число шагов заглушки", _display(means["stub_steps"], ".2f")),
            ("Среднее число вызовов инструментов", _display(means["tool_calls"], ".2f")),
            ("Оценка входных токенов, всего / в среднем", f"{_display(totals['input_tokens_estimate'])} / "
             f"{_display(means['input_tokens_estimate'], '.1f')}"),
            ("Оценка токенов из кэша, всего / доля", f"{_display(totals['cached_tokens_estimate'])} / "
             f"{_display(totals['cached_input_fraction'], '.1%')}"),
            ("Условные списания по трейсам, USD", _display(totals["simulated_cost_usd"], ".6f")),
            ("Реальные вызовы модели по трейсам", _display(totals["model_calls"])),
            ("Реальные списания по трейсам, USD", _display(totals["cost_usd"], ".6f")),
        ]
        intro = ("Заглушка выбирает цитату из прочитанной публикации. Замер показывает работу CLI и оценщика с этой "
                 "заглушкой. Все вопросы, включая технические сбои, входят в знаменатель долей. "
                 "Средние шаги, вызовы и токены считаются по доступным трейсам. Токены и условные списания смоделированы.")
    command = "ask --mcp" if use_mcp else "ask --no-tools"
    lines = [
        f"# FreshQA · прогон `{command}`", "",
        intro,
        "", "| Показатель | Значение |", "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    header = "| Вопрос | Статус | Ответ | URL | Цитата | Время, с |"
    divider = "| --- | --- | --- | --- | --- | ---: |"
    if paid:
        header += " Расход, USD | Вход | Выход | Из кэша |"
        divider += " ---: | ---: | ---: | ---: |"
    lines.extend(["", header, divider])
    for case in summary["cases"]:
        checks = " | ".join("да" if case[key] else "—" for key in _CHECKS[:-1])
        status = case["status"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        duration = "—" if case["status"] in _SKIPPED else _display(case["duration_s"], ".3f")
        line = f"| `{case['id']}` | {status} | {checks} | {duration} |"
        if paid:
            costs = _display(case["cost_usd"], ".8f")
            if case["cost_usd"] is None and case["known_cost_usd"] is not None:
                costs = f"{case['known_cost_usd']:.8f} (частично)"
            usage = " | ".join(_display(case[key]) for key in _PAID_USAGE[:-1])
            line += f" {costs} | {usage} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _per_question(value, questions):
    return value / questions if value is not None and questions else None


def render_comparison(summary):
    """Render a common answer metric plus MCP source checks for each configuration."""
    lines = [
        "# FreshQA · сравнение трёх конфигураций", "",
        "Точность, вызовы, токены и цена на вопрос рассчитаны по всем вопросам набора. "
        "Задержка усреднена по запущенным командам. Цена верного ответа включает все списания "
        "конфигурации. Подтверждённые ответы с источником учитывают совпадение ответа, URL и цитаты.",
        "", "| Показатель | Значение |", "| --- | ---: |",
    ]
    for key, label in (
        ("run_budget_usd", "Общий бюджет, USD"),
        ("run_cost_usd", "Общий расход по журналам, USD"),
        ("run_remaining_usd", "Остаток бюджета, USD"),
    ):
        lines.append(f"| {label} | {_display(summary[key], '.8f')} |")
    if summary["run_cost_usd"] is None:
        lines.append(f"| Подтверждённая часть расходов, USD | {_display(summary['run_known_cost_usd'], '.8f')} |")
    if summary.get("stop_reason"):
        lines.append(f"| Причина остановки | `{summary['stop_reason']}` |")
    columns = (
        "Конфигурация", "Точность", "Подтверждено источником", "Завершено", "Пропущено",
        "Вызовы на вопрос", "Токены на вопрос", "Центы на вопрос", "Центы за верный ответ",
        "Задержка, с", "Токены из кэша",
    )
    lines.extend(["", f"| {' | '.join(columns)} |", "| --- |" + " ---: |" * (len(columns) - 1)])
    for config in summary["configurations"]:
        totals, means, rates = config["totals"], config["means"], config["rates"]
        questions, correct = totals["questions"], totals["answer_correct"]
        cost = config["run_cost_usd"]
        cents = cost * 100 if cost is not None else None
        prompt, completion = totals["prompt_tokens"], totals["completion_tokens"]
        tokens = prompt + completion if prompt is not None and completion is not None else None
        cached, fraction = totals["cached_tokens"], totals["cached_input_fraction"]
        cache = _display(cached)
        if fraction is not None:
            partial = " по доступным трейсам" if cached is None or prompt is None else ""
            cache += f" ({fraction:.1%}{partial})"
        cells = [
            config["label"], f"{correct}/{questions} ({_display(rates['answer_correct'], '.1%')})",
            f"{totals['verified_correct']}/{questions}" if config["use_mcp"] else "—",
            f"{totals['completed']}/{questions}", str(totals["skipped"]),
            _display(_per_question(totals["model_calls"], questions), ".2f"),
            _display(_per_question(tokens, questions), ".1f"), _display(_per_question(cents, questions), ".5f"),
            _display(_per_question(cents, correct), ".5f"), _display(means["duration_s"], ".3f"), cache,
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"
