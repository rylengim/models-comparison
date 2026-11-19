import json
from types import SimpleNamespace

import pytest

from hw4.mcp_tool_pins import ToolPinError, tool_fingerprint, verify_tool_pins


def publication_tools():
    return {
        "web_search": SimpleNamespace(
            name="web_search", description="Поиск публикаций",
            inputSchema={"type": "object", "properties": {"query": {"type": "string", "maxLength": 500}}},
            outputSchema=None,
        ),
        "page_find": SimpleNamespace(
            name="page_find", description="Чтение публикации",
            inputSchema={"type": "object", "properties": {"url": {"type": "string"}}},
            outputSchema=None,
        ),
    }


def write_pins(path, tools):
    path.write_text(json.dumps({name: tool_fingerprint(tool) for name, tool in tools.items()}), encoding="utf-8")


def test_tools_match_pins_with_reordered_schema_keys(tmp_path):
    tools = publication_tools()
    path = tmp_path / "pins.json"
    write_pins(path, tools)

    tools["web_search"].inputSchema = {
        "properties": {"query": {"maxLength": 500, "type": "string"}}, "type": "object",
    }

    verify_tool_pins(tools, pin_path=path)


@pytest.mark.parametrize(("name", "field", "value"), [
    ("web_search", "description", "Подменённое описание"),
    ("page_find", "inputSchema", {"type": "object", "properties": {"url": {"type": "integer"}}}),
    ("page_find", "outputSchema", {"type": "object"}),
])
def test_changed_tool_metadata_rejected(tmp_path, name, field, value):
    tools = publication_tools()
    path = tmp_path / "pins.json"
    write_pins(path, tools)
    setattr(tools[name], field, value)

    with pytest.raises(ToolPinError, match=name) as error:
        verify_tool_pins(tools, pin_path=path)
    assert "Подменённое описание" not in str(error.value)


@pytest.mark.parametrize("pin_file", [None, "not JSON", "{}", '{"web_search": "bad", "page_find": "bad"}'])
def test_unavailable_or_invalid_pin_file_rejected(tmp_path, pin_file):
    path = tmp_path / "pins.json"
    if pin_file is not None:
        path.write_text(pin_file, encoding="utf-8")

    with pytest.raises(ToolPinError):
        verify_tool_pins(publication_tools(), pin_path=path)
