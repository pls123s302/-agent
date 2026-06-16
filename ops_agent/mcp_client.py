from typing import Any


def create_mcp_client(mcp_servers: dict[str, Any]):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(mcp_servers)


def find_tool(tools, tool_name: str):
    for tool in tools:
        if tool.name == tool_name or tool.name.endswith(tool_name):
            return tool
    available = ", ".join(tool.name for tool in tools)
    raise RuntimeError(f"Cannot find MCP tool: {tool_name}. Available tools: {available}")
