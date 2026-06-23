from typing import Any

from ops_agent.capability_tools import create_capability_tools
from ops_agent.config import load_mcp_servers
from ops_agent.mcp_client import create_mcp_client


async def create_tool_groups(config: dict[str, Any]) -> dict[str, list[Any]]:
    adapter_tools = []
    if config.get("mcp", {}).get("enabled", False):
        try:
            mcp_servers = load_mcp_servers(config)
            mcp_client = create_mcp_client(mcp_servers)
            adapter_tools.extend(await mcp_client.get_tools())
        except ModuleNotFoundError as error:
            config.setdefault("runtime_warnings", []).append(
                f"Docker MCP tools are disabled because dependency is missing: {error.name}"
            )

    return create_capability_tools(config, adapter_tools)


async def create_tools(config: dict[str, Any]) -> list[Any]:
    groups = await create_tool_groups(config)
    return groups["all"]
