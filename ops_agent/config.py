import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)

    variables = {
        "project_root": str(PROJECT_ROOT),
        "config_dir": str(config_path.parent),
        "python_executable": sys.executable,
    }
    return replace_placeholders(config, variables)


def load_mcp_servers(config: dict[str, Any]) -> dict[str, Any]:
    mcp_config_path = Path(config["mcp"]["servers_config_path"]).resolve()
    variables = {
        "project_root": str(PROJECT_ROOT),
        "mcp_root": str(Path(config["mcp"]["root"]).resolve()),
        "python_executable": sys.executable,
    }
    return replace_placeholders(read_json(mcp_config_path), variables)
