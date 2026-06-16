import json
from typing import Any

from ops_agent.state import AgentState


class OpsAgentNodes:
    def __init__(
        self,
        llm: Any,
        prompts: dict[str, str],
        metrics_tools: list[Any],
        docker_tools: list[Any],
        max_tool_steps: int,
        runtime_warnings: list[str] | None = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.metrics_llm = llm.bind_tools(metrics_tools)
        self.docker_tools = docker_tools
        self.docker_llm = llm.bind_tools(docker_tools) if docker_tools else llm
        self.max_tool_steps = max_tool_steps
        self.runtime_warnings = runtime_warnings or []

    async def router_agent(self, state: AgentState) -> AgentState:
        heuristic_route = route_by_heuristic(state)
        if heuristic_route:
            return {
                "route": heuristic_route,
                "next_action": "",
            }

        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["router"]),
                *state["messages"],
            ]
        )
        route = parse_route(response.content)
        return {
            "route": route,
            "next_action": "",
        }

    async def chat_agent(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["chat"]),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    async def log_agent(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["log"]),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    async def metrics_agent(self, state: AgentState) -> AgentState:
        response = await self.metrics_llm.ainvoke(
            [
                ("system", self.prompts["metrics"]),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    async def docker_agent(self, state: AgentState) -> AgentState:
        if not self.docker_tools:
            warning = "\n".join(self.runtime_warnings) or "Docker MCP tools are not available."
            return {
                "messages": [
                    (
                        "assistant",
                        "Docker 查询工具当前不可用，无法读取容器列表、日志或 stats。\n"
                        f"原因：{warning}\n"
                        "请确认当前 Python 环境已安装 mcp 和 langchain-mcp-adapters，并且 Docker CLI 可用。",
                    )
                ]
            }

        response = await self.docker_llm.ainvoke(
            [
                ("system", self.prompts["docker"]),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    async def reflect_agent(self, state: AgentState) -> AgentState:
        if state.get("tool_steps", 0) >= self.max_tool_steps:
            return {"next_action": "final"}

        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["reflect"]),
                *state["messages"],
            ]
        )
        return {"next_action": parse_next_action(response.content)}

    async def final_agent(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["final"]),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    def count_tool_step(self, state: AgentState) -> AgentState:
        return {"tool_steps": state.get("tool_steps", 0) + 1}


def route_by_heuristic(state: AgentState) -> str:
    user_text = latest_user_text(state)
    lowered = user_text.lower()

    if looks_like_pasted_log(user_text):
        return "log"

    docker_keywords = (
        "docker",
        "容器",
        "container",
        "日志",
        "log",
        "logs",
        "inspect",
        "stats",
    )
    recent_log_keywords = ("最近", "latest", "last", "tail", "条日志")
    if any(keyword in lowered for keyword in docker_keywords) and any(
        keyword in lowered for keyword in recent_log_keywords
    ):
        return "docker"

    if "日志" in lowered and not looks_like_pasted_log(user_text):
        return "docker"

    metrics_keywords = ("cpu", "内存", "memory", "资源", "监控", "性能", "占用")
    if any(keyword in lowered for keyword in metrics_keywords):
        return "metrics"

    return ""


def latest_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, tuple) and len(message) >= 2 and message[0] == "human":
            return str(message[1])
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return ""


def looks_like_pasted_log(text: str) -> bool:
    lowered = text.lower()
    log_markers = (
        "error",
        "warn",
        "exception",
        "traceback",
        "timeout",
        "failed",
        "stack",
        " at ",
    )
    if "分析日志" in lowered and any(marker in lowered for marker in log_markers):
        return True
    return "\n" in text and any(marker in lowered for marker in log_markers)


def parse_route(content: str) -> str:
    allowed = {"chat", "log", "metrics", "docker"}
    parsed = parse_json_object(content)
    route = str(parsed.get("route", "")).lower()
    if route in allowed:
        return route

    lowered = content.lower()
    for candidate in allowed:
        if candidate in lowered:
            return candidate
    return "chat"


def parse_next_action(content: str) -> str:
    allowed = {"continue", "ask_user", "final"}
    parsed = parse_json_object(content)
    action = str(parsed.get("next_action", "")).lower()
    if action in allowed:
        return action

    lowered = content.lower()
    for candidate in allowed:
        if candidate in lowered:
            return candidate
    return "final"


def parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if isinstance(value, dict):
        return value
    return {}
