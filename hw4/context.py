"""Shorten search results sent to the model while preserving the agent's history."""

import json


SEARCH_SNIPPET_CHARS = 200


def _json_object(value):
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _confirmed_url(function, content):
    requested = _json_object(function.get("arguments")).get("url")
    if not isinstance(requested, str) or not requested.strip():
        return None
    result = _json_object(content)
    if "error" in result or result.get("found") is not True or result.get("url") != requested:
        return None
    excerpts = result.get("excerpts")
    if (isinstance(excerpts, list) and all(isinstance(text, str) for text in excerpts)
            and any(text.strip() for text in excerpts)):
        return requested
    return None


def _preview_hit(hit):
    preview = {"url": hit["url"]}
    title, snippet = hit.get("title"), hit.get("snippet")
    if isinstance(title, str):
        preview["title"] = title
    if isinstance(snippet, str):
        if len(snippet) > SEARCH_SNIPPET_CHARS:
            snippet = snippet[:SEARCH_SNIPPET_CHARS - 1].rstrip() + "…"
        preview["snippet"] = snippet
    return preview


def _compact_search(content, index, reads):
    search = _json_object(content)
    hits = search.get("results")
    if "error" in search or not isinstance(hits, list) or not hits:
        return content, False
    urls = [hit.get("url") if isinstance(hit, dict) else None for hit in hits]
    if any(not isinstance(url, str) or not url.strip() for url in urls):
        return content, False
    read_hits = [{"url": url} for url in urls if reads.get(url, -1) > index]
    if read_hits:
        payload = {"results": read_hits, "compacted": True}
    else:
        payload = {"results": [_preview_hit(hit) for hit in hits]}
    compacted = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (compacted, not read_hits) if len(compacted) < len(content) else (content, False)


def compact_search_results(messages):
    """Return a model-facing copy and character counts for tool results."""
    pending = {}
    searches = []
    reads = {}
    before = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                call_id, function = call.get("id"), call.get("function")
                if isinstance(call_id, str) and call_id and isinstance(function, dict):
                    pending[call_id] = None if call_id in pending else (index, function)
        elif role == "tool":
            content = message.get("content")
            if isinstance(content, str):
                before += len(content)
            call_id = message.get("tool_call_id")
            request = pending.pop(call_id, None) if isinstance(call_id, str) else None
            if request is None or not isinstance(content, str):
                continue
            request_index, function = request
            if function.get("name") == "web_search":
                searches.append((index, content))
            elif function.get("name") == "page_find":
                url = _confirmed_url(function, content)
                if url is not None:
                    reads[url] = max(reads.get(url, -1), request_index)

    prepared = list(messages)
    after = before
    shortened = 0
    previews = 0
    for index, content in searches:
        compacted, is_preview = _compact_search(content, index, reads)
        if compacted == content:
            continue
        prepared[index] = {**prepared[index], "content": compacted}
        after -= len(content) - len(compacted)
        if is_preview:
            previews += 1
        else:
            shortened += 1
    return prepared, {
        "tool_chars_before": before,
        "tool_chars_after": after,
        "compacted_search_results": shortened,
        "pre_read_compacted_search_results": previews,
    }
