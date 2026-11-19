"""Resolve evidence from saved FreshQA MCP reads."""

import json
import re
from urllib.parse import urlsplit

from freshqa.dataset import final_text


QUOTE_ID_FORMAT = "QUOTE_ID"
QUOTE_BLOCK_BREAK = re.compile(r"\n[ \t\r]*\n|\n(?=[ \t]*>)")
_QUOTE_ID_LINE = re.compile(r"^[ \t]*QUOTE_ID[ \t]*:", re.MULTILINE | re.IGNORECASE)


def json_object(content):
    try:
        output = json.loads(content)
    except (TypeError, ValueError, RecursionError):
        return {}
    return output if isinstance(output, dict) else {}


def quote_blocks(read, read_number):
    block_number = 0
    for excerpt in read["excerpts"]:
        if not isinstance(excerpt, str):
            continue
        for part in QUOTE_BLOCK_BREAK.split(excerpt):
            text = part.lstrip().removeprefix(">").strip()
            if text:
                block_number += 1
                yield {"id": f"r{read_number}-b{block_number}", "text": text}, excerpt


def is_https_url(url):
    if not isinstance(url, str) or not url.startswith("https://") or any(char.isspace() for char in url):
        return False
    try:
        return bool(urlsplit(url).hostname)
    except ValueError:
        return False


def _read_evidence(event):
    arguments = event.get("arguments")
    if event.get("type") != "tool" or event.get("name") != "page_find" or not isinstance(arguments, dict):
        return None
    url = arguments.get("url")
    if not is_https_url(url):
        return None
    page = json_object(event.get("output"))
    if "error" in page or page.get("found") is not True or page.get("url") != url:
        return None
    excerpts = page.get("excerpts")
    if not isinstance(excerpts, list):
        return None
    passages = [text for text in excerpts if isinstance(text, str) and text.strip()]
    return {"url": url, "excerpts": passages} if passages else None


def _successful_reads(events, *, step=None):
    tool_events = (event for event in events if event.get("type") == "tool")
    for read_number, event in enumerate(tool_events, start=1):
        if step is None or step == event.get("step"):
            read = _read_evidence(event)
            if read is not None:
                yield read_number, read


def recent_evidence(events):
    for event in reversed(events):
        if event.get("type") == "llm" and not event.get("early_final"):
            step = event.get("step")
            if type(step) is not int or step < 1:
                return None
            reads = list(_successful_reads(events, step=step))
            return reads[-1] if reads else None
    return None


def resolve_quote(final, reads):
    lines = final.splitlines()
    if len(lines) != 2 or not lines[1].startswith("QUOTE_ID:"):
        return None
    answer = lines[0].strip()
    quote_id = lines[1][len("QUOTE_ID:"):].strip()
    if not answer or not quote_id or answer.casefold() == "unknown":
        return None
    for evidence in reads:
        if evidence is None:
            continue
        read_number, read = evidence
        for block, excerpt in quote_blocks(read, read_number):
            if block["id"] == quote_id:
                return answer, block["text"], read["url"], excerpt
    return None


def resolve_final_quote(final, events):
    if events and events[-1].get("early_final_accepted") is True:
        reads = (recent_evidence(events),)
    else:
        reads = _successful_reads(events)
    return resolve_quote(final, reads)


def _llm_content(event):
    message = event.get("message")
    if event.get("type") != "llm" or not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def supported_verdict(event):
    return (event.get("finish_reason") == "stop"
            and _llm_content(event).strip() == "SUPPORTED"
            and event["message"].get("role") == "assistant"
            and not event["message"].get("tool_calls"))


def quote_id_evidence_matches_trace(trace):
    """Check the selected block and verdict. Legacy traces keep their original scoring rules."""
    events = trace["events"]
    uses_quote_id = trace.get("quote_format") == QUOTE_ID_FORMAT or any(
        _QUOTE_ID_LINE.search(_llm_content(event)) for event in events
    )
    if not uses_quote_id:
        return True
    if len(events) < 2:
        return False

    candidate, judge = events[-2:]
    if (judge.get("evidence_check") is not True or judge.get("evidence_supported") is not True
            or judge.get("tools_enabled") is not False or not supported_verdict(judge)):
        return False

    content = _llm_content(candidate)
    if (not content or candidate["message"].get("role") != "assistant"
            or candidate["message"].get("tool_calls") or candidate.get("finish_reason") == "length"):
        return False
    if candidate.get("early_final") is True and candidate.get("early_final_accepted") is not True:
        return False
    verified = resolve_final_quote(final_text(content), events[:-1])
    return verified is not None and verified[:3] == (trace["final"], trace.get("quote"), trace.get("source_url"))
