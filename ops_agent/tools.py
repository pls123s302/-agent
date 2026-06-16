from typing import Any

from langchain_core.tools import StructuredTool

from ops_agent.cadvisor import collect_cadvisor_summary
from ops_agent.config import load_mcp_servers
from ops_agent.mcp_client import create_mcp_client


class CAdvisorTools:
    def __init__(self, cadvisor_config: dict[str, Any]):
        self.cadvisor_config = cadvisor_config

    async def query_cadvisor_metrics(self) -> str:
        return await collect_cadvisor_summary(self.cadvisor_config)


async def create_tool_groups(config: dict[str, Any]) -> dict[str, list[Any]]:
    cadvisor_tools = CAdvisorTools(config["cadvisor"])
    tool_config = config["tools"]["query_cadvisor_metrics"]

    metrics_tools = [
        StructuredTool.from_function(
            coroutine=cadvisor_tools.query_cadvisor_metrics,
            name=tool_config["name"],
            description=tool_config["description"],
        )
    ]

    docker_tools = []
    if config.get("mcp", {}).get("enabled", False):
        try:
            mcp_servers = load_mcp_servers(config)
            mcp_client = create_mcp_client(mcp_servers)
            docker_tools.extend(await mcp_client.get_tools())
        except ModuleNotFoundError as error:
            config.setdefault("runtime_warnings", []).append(
                f"Docker MCP tools are disabled because dependency is missing: {error.name}"
            )

    return {
        "metrics": metrics_tools,
        "docker": docker_tools,
        "all": [*metrics_tools, *docker_tools],
    }


async def create_tools(config: dict[str, Any]) -> list[Any]:
    groups = await create_tool_groups(config)
    return groups["all"]
