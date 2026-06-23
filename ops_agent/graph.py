from typing import Any

from ops_agent.llm import create_llm
from ops_agent.nodes import OpsAgentNodes
from ops_agent.state import AgentState
from ops_agent.tools import create_tool_groups


async def build_graph(config: dict[str, Any]):
    from langgraph.graph import END, StateGraph

    llm = create_llm(config["llm"])
    tool_groups = await create_tool_groups(config)
    nodes = OpsAgentNodes(
        llm=llm,
        prompts=config["prompts"],
        tools=tool_groups["all"],
        max_tool_steps=int(config["agent"]["max_tool_steps"]),
        max_replans=int(config["agent"].get("max_replans", 2)),
        runtime_warnings=config.get("runtime_warnings", []),
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", nodes.router_agent)
    graph.add_node("plan_agent", nodes.plan_agent)
    graph.add_node("replan_agent", nodes.replan_agent)
    graph.add_node("execute_step", nodes.execute_step)
    graph.add_node("reflect_agent", nodes.reflect_agent)
    graph.add_node("chat_agent", nodes.chat_agent)
    graph.add_node("log_agent", nodes.log_agent)
    graph.add_node("final_agent", nodes.final_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chat": "chat_agent",
            "log": "log_agent",
            "plan": "plan_agent",
        },
    )
    graph.add_conditional_edges(
        "plan_agent",
        route_from_plan,
        {
            "execute": "execute_step",
            "chat": "chat_agent",
            "log": "log_agent",
            "final": "final_agent",
        },
    )
    graph.add_edge("execute_step", "reflect_agent")
    graph.add_conditional_edges(
        "reflect_agent",
        route_from_reflect,
        {
            "continue": "execute_step",
            "replan": "replan_agent",
            "ask_user": "final_agent",
            "final": "final_agent",
        },
    )
    graph.add_conditional_edges(
        "replan_agent",
        route_after_replan,
        {
            "execute": "execute_step",
            "final": "final_agent",
        },
    )
    graph.add_edge("chat_agent", END)
    graph.add_edge("log_agent", END)
    graph.add_edge("final_agent", END)
    return graph.compile()


def route_after_router(state: AgentState) -> str:
    route = state.get("route", "chat")
    if route in {"ops", "metrics", "docker"}:
        return "plan"
    return route


def route_from_plan(state: AgentState) -> str:
    route = state.get("route", "chat")
    if route in {"chat", "log"}:
        return route
    if state.get("plan", {}).get("steps"):
        return "execute"
    return "final"


def route_from_reflect(state: AgentState) -> str:
    return state.get("next_action", "final")


def route_after_replan(state: AgentState) -> str:
    if state.get("next_action") == "continue" and state.get("current_step_index", 0) < len(
        state.get("plan", {}).get("steps", [])
    ):
        return "execute"
    return "final"
