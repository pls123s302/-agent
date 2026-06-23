from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    route: str
    environment_id: str
    environment_type: str
    plan: dict[str, Any]
    current_step_index: int
    observations: list[dict[str, Any]]
    next_action: str
    tool_steps: int
    replan_count: int
