import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from ops_agent.config import DEFAULT_CONFIG_PATH, load_config
from ops_agent.graph import build_graph


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph ops-agent demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="application config file path")
    parser.add_argument("--question", help="override the default question in config")
    parser.add_argument("--cadvisor-url", help="override cAdvisor base URL, for example http://localhost:8080")
    parser.add_argument("--environment-id", help="target environment id")
    parser.add_argument("--once", action="store_true", help="run a single interactive question and exit")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    if args.question:
        config["task"]["question"] = args.question
    if args.cadvisor_url:
        config["cadvisor"]["base_url"] = args.cadvisor_url
    return config


def resolve_user_question(config: dict[str, Any], args) -> str:
    if args.question:
        return config["task"]["question"]

    prompt = config.get("cli", {}).get("input_prompt", "Please enter your question: ")
    question = input(prompt).strip()
    if not question:
        raise ValueError("No question entered. Exit.")
    return question


async def run():
    args = parse_args()
    config = load_config(Path(args.config))
    config = apply_cli_overrides(config, args)
    environment_id = args.environment_id or default_environment_id(config)
    environment_type = environment_type_for(config, environment_id)

    app = await build_graph(config)
    if args.question or args.once:
        question = resolve_user_question(config, args)
        await run_single_turn(app, question, environment_id, environment_type)
        return

    await run_chat_session(app, config, environment_id, environment_type)


async def run_single_turn(app: Any, question: str, environment_id: str, environment_type: str) -> None:
    result = await app.ainvoke(initial_state([("human", question)], environment_id, environment_type))
    safe_print(result["messages"][-1].content)


async def run_chat_session(app: Any, config: dict[str, Any], environment_id: str, environment_type: str) -> None:
    messages = []
    session_config = config.get("session", {})
    exit_commands = set(session_config.get("exit_commands", ["exit", "quit", "q", "退出"]))
    max_messages = int(session_config.get("max_messages", 30))
    prompt = config.get("cli", {}).get("input_prompt", "Please enter your question: ")

    while True:
        try:
            question = input(prompt).strip()
        except EOFError:
            safe_print("已退出。")
            return

        if not question:
            continue
        if question.lower() in exit_commands:
            safe_print("已退出。")
            return

        messages.append(("human", question))
        result = await app.ainvoke(initial_state(messages, environment_id, environment_type))
        messages = trim_messages(result["messages"], max_messages)
        safe_print(result["messages"][-1].content)


def initial_state(messages: list[Any], environment_id: str, environment_type: str) -> dict[str, Any]:
    return {
        "messages": messages,
        "route": "",
        "environment_id": environment_id,
        "environment_type": environment_type,
        "plan": {},
        "current_step_index": 0,
        "observations": [],
        "next_action": "",
        "tool_steps": 0,
        "replan_count": 0,
    }


def default_environment_id(config: dict[str, Any]) -> str:
    environments = config.get("environments", [])
    for environment in environments:
        if environment.get("default"):
            return environment["id"]
    if environments:
        return environments[0]["id"]
    return "local-docker"


def environment_type_for(config: dict[str, Any], environment_id: str) -> str:
    for environment in config.get("environments", []):
        if environment.get("id") == environment_id:
            return environment.get("type", "docker")
    return "docker"


def trim_messages(messages: list[Any], max_messages: int) -> list[Any]:
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def safe_print(text: str) -> None:
    output = f"{text}\n"
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write(output.encode(encoding, errors="replace"))


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
