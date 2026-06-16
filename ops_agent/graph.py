from typing import Any

from ops_agent.llm import create_llm
from ops_agent.nodes import OpsAgentNodes
from ops_agent.state import AgentState
from ops_agent.tools import create_tool_groups


async def build_graph(config: dict[str, Any]):
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    llm = create_llm(config["llm"])
    tool_groups = await create_tool_groups(config)
    nodes = OpsAgentNodes(
        llm=llm,
        prompts=config["prompts"],
        metrics_tools=tool_groups["metrics"],
        docker_tools=tool_groups["docker"],
        max_tool_steps=int(config["agent"]["max_tool_steps"]),
        runtime_warnings=config.get("runtime_warnings", []),
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", nodes.router_agent)
    graph.add_node("chat_agent", nodes.chat_agent)
    graph.add_node("log_agent", nodes.log_agent)
    graph.add_node("metrics_agent", nodes.metrics_agent)
    graph.add_node("metrics_tools", ToolNode(tool_groups["metrics"]))
    graph.add_node("docker_agent", nodes.docker_agent)
    if tool_groups["docker"]:
        graph.add_node("docker_tools", ToolNode(tool_groups["docker"]))
    graph.add_node("count_docker_tool_step", nodes.count_tool_step)
    graph.add_node("reflect_agent", nodes.reflect_agent)
    graph.add_node("final_agent", nodes.final_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {
            "chat": "chat_agent",
            "log": "log_agent",
            "metrics": "metrics_agent",
            "docker": "docker_agent",
        },
    )

    graph.add_edge("chat_agent", END)
    graph.add_edge("log_agent", END)

    graph.add_conditional_edges(
        "metrics_agent",
        route_by_tool_calls,
        {
            "tools": "metrics_tools",
            "end": END,
        },
    )
    graph.add_edge("metrics_tools", "final_agent")

    if tool_groups["docker"]:
        graph.add_conditional_edges(
            "docker_agent",
            route_by_tool_calls,
            {
                "tools": "docker_tools",
                "end": END,
            },
        )
        graph.add_edge("docker_tools", "count_docker_tool_step")
    else:
        graph.add_edge("docker_agent", END)
    graph.add_edge("count_docker_tool_step", "reflect_agent")
    graph.add_conditional_edges(
        "reflect_agent",
        route_from_reflect,
        {
            "continue": "docker_agent",
            "ask_user": "final_agent",
            "final": "final_agent",
        },
    )
    graph.add_edge("final_agent", END)
    return graph.compile()


def route_from_router(state: AgentState) -> str:
    return state.get("route", "chat")


def route_by_tool_calls(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


def route_from_reflect(state: AgentState) -> str:
    return state.get("next_action", "final")
