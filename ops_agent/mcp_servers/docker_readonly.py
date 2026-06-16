import json
import re
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("docker-readonly")
CONTAINER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
MAX_TAIL_LINES = 500
COMMAND_TIMEOUT_SECONDS = 15


def run_docker_command(args: list[str]) -> str:
    completed = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "docker command failed")
    return completed.stdout.strip()


def validate_container_name(container_name: str) -> str:
    if not CONTAINER_NAME_PATTERN.match(container_name):
        raise ValueError("Invalid container name or id.")
    return container_name


def normalize_tail(tail: int) -> int:
    if tail < 1:
        return 100
    return min(tail, MAX_TAIL_LINES)


def parse_json_lines(output: str) -> list[dict[str, Any]]:
    items = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def list_container_records(all_containers: bool = True) -> list[dict[str, Any]]:
    args = ["ps", "--format", "{{json .}}"]
    if all_containers:
        args.insert(1, "-a")
    output = run_docker_command(args)
    return parse_json_lines(output)


def record_names(record: dict[str, Any]) -> list[str]:
    names = []
    for key in ("Names", "NamesLocal", "Name"):
        value = record.get(key)
        if isinstance(value, str) and value:
            names.extend(item.strip().lstrip("/") for item in value.split(","))
    return [name for name in names if name]


def resolve_exact_container_name(container_name: str) -> str:
    query = validate_container_name(container_name)
    records = list_container_records(all_containers=True)

    for record in records:
        names = record_names(record)
        container_id = str(record.get("ID", ""))
        if query == container_id or container_id.startswith(query) or query in names:
            return names[0] if names else query

    available = []
    for record in records:
        available.extend(record_names(record))
    raise ValueError(
        f"No exact container matched '{container_name}'. "
        "Call search_containers first if the user provided a partial service name. "
        "Available containers: " + ", ".join(available)
    )


def find_container_candidates(query: str) -> list[dict[str, Any]]:
    safe_query = validate_container_name(query)
    records = list_container_records(all_containers=True)
    matches = []
    query_lower = safe_query.lower()

    for record in records:
        names = record_names(record)
        container_id = str(record.get("ID", ""))
        candidates = [*names, container_id]
        if any(query_lower in candidate.lower() for candidate in candidates if candidate):
            matches.append(record)

    return matches


@mcp.tool()
def search_containers(query: str) -> list[dict[str, Any]]:
    """Search Docker containers by partial name/id. Use this before logs/inspect when the user gives an imprecise name."""
    return find_container_candidates(query)


def require_unambiguous_container_name(container_name: str) -> str:
    try:
        return resolve_exact_container_name(container_name)
    except ValueError:
        matches = find_container_candidates(container_name)
        if not matches:
            raise

    matched_names = []
    for record in matches:
        matched_names.extend(record_names(record))

    raise ValueError(
        "Container name is not exact. Ask the user to choose one of these containers: "
        + ", ".join(matched_names)
    )


@mcp.tool()
def list_containers(all_containers: bool = True) -> list[dict[str, Any]]:
    """List Docker containers. Use this before selecting a container by name."""
    return list_container_records(all_containers=all_containers)


@mcp.tool()
def get_container_logs(container_name: str, tail: int = 100) -> str:
    """Get recent logs from an exact Docker container name/id. Use search_containers first for partial names."""
    safe_name = require_unambiguous_container_name(container_name)
    safe_tail = normalize_tail(tail)
    return run_docker_command(["logs", "--tail", str(safe_tail), safe_name])


@mcp.tool()
def inspect_container(container_name: str) -> Any:
    """Inspect one exact Docker container name/id. Use search_containers first for partial names."""
    safe_name = require_unambiguous_container_name(container_name)
    output = run_docker_command(["inspect", safe_name])
    return json.loads(output)


@mcp.tool()
def get_container_stats(container_name: str | None = None) -> list[dict[str, Any]]:
    """Get one-shot Docker container stats. Container name/id must be exact when provided. Read-only."""
    args = ["stats", "--no-stream", "--format", "{{json .}}"]
    if container_name:
        args.append(require_unambiguous_container_name(container_name))
    output = run_docker_command(args)
    return parse_json_lines(output)


if __name__ == "__main__":
    mcp.run()
