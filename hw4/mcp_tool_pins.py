"""Verify MCP tool metadata against pins recorded from the reviewed server."""

import hashlib
import json
import re
from pathlib import Path


TOOL_NAMES = ("web_search", "page_find")
TOOL_PINS_PATH = Path(__file__).with_name("mcp_tool_pins.json")


class ToolPinError(ValueError):
    """The tool contract does not match the locally pinned version."""


def tool_fingerprint(tool):
    payload = {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
        "outputSchema": getattr(tool, "outputSchema", None),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate keys")
    return result


def verify_tool_pins(tools, *, pin_path=TOOL_PINS_PATH):
    try:
        pins = json.loads(Path(pin_path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except OSError:
        raise ToolPinError("Не удалось прочитать файл закреплённых отпечатков MCP.") from None
    except (ValueError, RecursionError):
        raise ToolPinError("Неверный формат файла закреплённых отпечатков MCP.") from None

    if (not isinstance(pins, dict) or set(pins) != set(TOOL_NAMES)
            or any(not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None
                   for pin in pins.values())):
        raise ToolPinError("Неверный формат файла закреплённых отпечатков MCP.")
    if set(tools) != set(TOOL_NAMES):
        raise ToolPinError("Неверный набор MCP-инструментов. Ожидались web_search и page_find.")

    changed = []
    for name in TOOL_NAMES:
        try:
            fingerprint = tool_fingerprint(tools[name])
        except (AttributeError, TypeError, ValueError, RecursionError):
            fingerprint = None
        if fingerprint != pins[name]:
            changed.append(name)
    if changed:
        raise ToolPinError(f"Изменилась схема или описание MCP-инструмента: {', '.join(changed)}.")
