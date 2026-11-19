"""Run a synchronous FreshQA operation against the local MCP publications server."""

import re
from datetime import timedelta
from pathlib import Path

from .mcp_tool_pins import TOOL_NAMES, ToolPinError, verify_tool_pins


REPO = Path(__file__).resolve().parents[1]
TIMEOUT = 15


class MCPUnavailable(ValueError):
    """The local MCP session could not be opened or maintained."""


class _PublicationTools:
    def __init__(self, tools, call):
        self._tools = tools
        self._call = call

    def schemas(self, names):
        return [
            {"type": "function", "function": {
                "name": name,
                "description": self._tools[name].description or "",
                "parameters": self._tools[name].inputSchema,
            }}
            for name in names
        ]

    def run(self, name, arguments):
        if name not in self._tools:
            return {"error": "tool_not_allowed"}
        if name == "page_find" and isinstance(arguments, dict):
            keywords = arguments.get("keywords")
            if isinstance(keywords, str):
                arguments = {**arguments, "keywords": re.findall(r"[^\s,;]+", keywords)}
        try:
            result = self._call(name, arguments)
        except Exception:
            return {"error": "mcp_transport_error"}
        if result.isError:
            return {"error": "mcp_tool_error"}
        if not isinstance(result.structuredContent, dict):
            return {"error": "mcp_invalid_result"}
        return result.structuredContent


async def with_publication_tools(operation):
    try:
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.shared.exceptions import McpError
    except ImportError:
        raise MCPUnavailable("Для режима MCP установите зависимости из hw4/requirements.txt.") from None

    params = StdioServerParameters(
        command="docker",
        args=["--host", "unix:///var/run/docker.sock", "compose", "--env-file", "/dev/null",
              "-f", str(REPO / "hw4/docker-compose.yml"), "run", "--rm", "--no-deps", "-T",
              "--build=false", "--pull", "never", "mcp"],
        cwd=REPO,
    )
    pin_error = None
    try:
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=TIMEOUT)) as session:
                with anyio.fail_after(TIMEOUT):
                    hello = await session.initialize()
                    listed = await session.list_tools()
                if hello.serverInfo.name != "freshqa-publications":
                    raise MCPUnavailable("Локальный MCP-сервер не предоставил инструменты FreshQA.")

                selected = [tool for tool in listed.tools if tool.name in TOOL_NAMES]
                tools = {tool.name: tool for tool in selected}
                try:
                    if len(tools) != len(selected):
                        raise ToolPinError("MCP-сервер вернул дубликаты закреплённых инструментов.")
                    verify_tool_pins(tools)
                except ToolPinError as error:
                    # Exit the MCP task groups normally so they preserve the specific diagnostic.
                    pin_error = error
                else:
                    # The agent runs synchronously; MCP I/O stays on the session's event loop.
                    registry = _PublicationTools(tools, lambda name, args: anyio.from_thread.run(session.call_tool, name, args))
                    return await anyio.to_thread.run_sync(operation, registry)
    except (OSError, TimeoutError, McpError, ExceptionGroup):
        if pin_error is None:
            raise MCPUnavailable("Не удалось открыть MCP FreshQA в локальном Docker. Проверьте Docker и образ сервиса mcp.") from None
    raise pin_error
