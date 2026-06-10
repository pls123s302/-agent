import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, TypedDict


DEFAULT_WORKSPACE_DIR = Path(__file__).parent.resolve()
DEFAULT_LOG_FILE = DEFAULT_WORKSPACE_DIR / "sample.log"
DEFAULT_MCP_CONFIG = DEFAULT_WORKSPACE_DIR / "mcp_config.json"
DEFAULT_QUESTION = "请分析这个日志里可能的问题，并给出简短建议。"
DEFAULT_MODEL = "qwen3:8b"


class AgentState(TypedDict):
    question: str
    file_path: str
    file_content: str
    answer: str


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph + Qwen3 + Filesystem MCP demo")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_LOG_FILE),
        help="要通过 Filesystem MCP 读取的文件路径",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="发送给模型的问题",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama 模型名称，例如 qwen3:8b",
    )
    parser.add_argument(
        "--mcp-root",
        default=str(DEFAULT_WORKSPACE_DIR),
        help="Filesystem MCP 允许访问的根目录",
    )
    parser.add_argument(
        "--mcp-config",
        default=str(DEFAULT_MCP_CONFIG),
        help="MCP server 配置文件路径",
    )
    return parser.parse_args()


def replace_placeholders(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [replace_placeholders(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: replace_placeholders(item, variables) for key, item in value.items()}
    return value


def load_mcp_config(config_path: Path, mcp_root: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    variables = {
        "mcp_root": str(mcp_root),
    }
    return replace_placeholders(config, variables)


def get_mcp_client(config_path: Path, mcp_root: Path):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(load_mcp_config(config_path, mcp_root))


def find_tool(tools, tool_name: str):
    for tool in tools:
        if tool.name == tool_name or tool.name.endswith(tool_name):
            return tool
    available = ", ".join(tool.name for tool in tools)
    raise RuntimeError(f"Cannot find MCP tool: {tool_name}. Available tools: {available}")


def build_graph(model: str, mcp_config: Path, mcp_root: Path):
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, StateGraph

    llm = ChatOllama(
        model=model,
        temperature=0.2,
    )

    async def read_file_by_mcp(state: AgentState) -> AgentState:
        client = get_mcp_client(mcp_config, mcp_root)
        tools = await client.get_tools()
        read_file_tool = find_tool(tools, "read_file")

        file_content = await read_file_tool.ainvoke({"path": state["file_path"]})
        return {
            **state,
            "file_content": str(file_content),
        }

    async def ask_qwen(state: AgentState) -> AgentState:
        response = await llm.ainvoke(
            [
                ("system", "你是一个回答简洁的中文运维助手。"),
                (
                    "human",
                    f"{state['question']}\n\n"
                    f"下面是通过 Filesystem MCP 读取到的文件内容：\n"
                    f"{state['file_content']}",
                ),
            ]
        )
        return {
            **state,
            "answer": response.content,
        }

    graph = StateGraph(AgentState)
    graph.add_node("read_file", read_file_by_mcp)
    graph.add_node("ask_qwen", ask_qwen)
    graph.set_entry_point("read_file")
    graph.add_edge("read_file", "ask_qwen")
    graph.add_edge("ask_qwen", END)
    return graph.compile()


async def main():
    args = parse_args()
    file_path = Path(args.file).resolve()
    mcp_root = Path(args.mcp_root).resolve()
    mcp_config = Path(args.mcp_config).resolve()

    app = build_graph(model=args.model, mcp_config=mcp_config, mcp_root=mcp_root)
    result = await app.ainvoke(
        {
            "question": args.question,
            "file_path": str(file_path),
            "file_content": "",
            "answer": "",
        }
    )
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
