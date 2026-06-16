from typing import Any


def create_llm(llm_config: dict[str, Any]):
    from langchain_ollama import ChatOllama

    return ChatOllama(**llm_config)
