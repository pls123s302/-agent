import json
import re
from typing import Any


def summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    action = str(observation.get("step", {}).get("action", ""))
    result = str(observation.get("result", ""))
    base = {
        "step_index": observation.get("step_index"),
        "action": action,
        "target": observation.get("tool_args", {}).get("target")
        or observation.get("tool_args", {}).get("query")
        or observation.get("step", {}).get("target"),
        "ok": bool(observation.get("ok")),
    }
    if not observation.get("ok"):
        return {
            **base,
            "summary": truncate(result, 600),
            "abnormal": True,
            "evidence": [truncate(result, 300)] if result else [],
        }

    summarizers = {
        "resolve_target": summarize_resolve_target,
        "query_metrics": summarize_metrics,
        "query_status": summarize_status,
        "query_logs": summarize_logs,
        "query_runtime_diagnostics": summarize_runtime_diagnostics,
        "query_workload_topology": summarize_workload_topology,
    }
    summary_func = summarizers.get(action, summarize_fallback)
    return {**base, **summary_func(result)}


def summarize_observations_for_model(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for observation in observations:
        summary = observation.get("summary")
        if isinstance(summary, dict):
            summaries.append(summary)
        else:
            summaries.append(summarize_observation(observation))
    return summaries


def summarize_resolve_target(result: str) -> dict[str, Any]:
    data = extract_json_payload(result)
    if not isinstance(data, dict):
        return summarize_fallback(result)
    candidates = data.get("candidates", [])
    resources = [
        str(candidate.get("resource") or f"{candidate.get('kind')}/{candidate.get('name')}")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    mode = data.get("suggested_mode", "")
    return {
        "summary": f"resolve_target found {len(resources)} candidate(s); suggested_mode={mode}.",
        "suggested_mode": mode,
        "ambiguous": bool(data.get("ambiguous")),
        "candidate_count": len(resources),
        "candidates": resources[:20],
        "abnormal": mode in {"not_found", "ask_user"},
    }


def summarize_metrics(result: str) -> dict[str, Any]:
    if re.search(r"metrics are not available|metrics api not available|not installed", result, re.I):
        return {
            "summary": "Metrics are unavailable; resource usage could not be collected.",
            "metrics": [],
            "abnormal": True,
            "evidence": important_lines(result, ("not available", "not installed", "Metrics API"))[:5],
        }

    rows = []
    for line in result.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Environment", "Namespace", "Target", "Requested", "NAME", "POD")):
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) >= 3 and looks_like_metric(parts[-2], parts[-1]):
            rows.append(
                {
                    "name": parts[0],
                    "container": parts[1] if len(parts) >= 4 else "",
                    "cpu": parts[-2],
                    "memory": parts[-1],
                }
            )
    abnormal = any(cpu_millicores(row["cpu"]) >= 800 or memory_mib(row["memory"]) >= 1024 for row in rows)
    if rows:
        summary = f"Collected CPU and memory metrics for {len(rows)} row(s); abnormal={abnormal}."
    else:
        summary = truncate(result, 500)
    return {
        "summary": summary,
        "metrics": rows[:30],
        "abnormal": abnormal,
    }


def summarize_status(result: str) -> dict[str, Any]:
    inventory = summarize_inventory_status(result)
    if inventory:
        return inventory

    sections = split_tool_sections(result, "# Status for ")
    workloads = []
    abnormal = False
    for title, body in sections:
        name = extract_regex(body, r"^Name:\s*(.+)$", re.MULTILINE) or title
        namespace = extract_regex(body, r"^Namespace:\s*(.+)$", re.MULTILINE)
        replicas = extract_regex(body, r"^Replicas:\s*(.+)$", re.MULTILINE)
        unavailable = extract_regex(body, r"(\d+)\s+unavailable")
        available_false = bool(re.search(r"Available\s+False", body))
        bad_state = bool(re.search(r"CrashLoopBackOff|ImagePullBackOff|ErrImagePull|Pending|Failed", body, re.I))
        if unavailable and int(unavailable) > 0:
            abnormal = True
        if available_false or bad_state:
            abnormal = True
        workloads.append(
            {
                "resource": title.strip() if title else name,
                "name": name.strip(),
                "namespace": namespace.strip() if namespace else "",
                "replicas": replicas.strip() if replicas else "",
                "available_false": available_false,
                "bad_state_detected": bad_state,
            }
        )
    if not workloads:
        abnormal = bool(re.search(r"CrashLoopBackOff|ImagePullBackOff|ErrImagePull|Pending|Failed", result, re.I))
    return {
        "summary": f"Checked status for {len(workloads) or 1} resource section(s); abnormal={abnormal}.",
        "workloads": workloads[:30],
        "abnormal": abnormal,
        "evidence": important_lines(result, ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "Failed", "Warning"))[:10],
    }


def summarize_inventory_status(result: str) -> dict[str, Any]:
    if "# Namespace resources:" not in result and not re.search(r"^pod/", result, re.MULTILINE):
        return {}

    lines = result.splitlines()
    in_nodes = False
    in_resources = "# Namespace resources:" not in result
    nodes = []
    pods = []
    deployments = []
    services = []
    abnormal = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# Nodes"):
            in_nodes = True
            in_resources = False
            continue
        if stripped.startswith("# Namespace resources:"):
            in_nodes = False
            in_resources = True
            continue
        if stripped.startswith("NAME"):
            continue
        parts = re.split(r"\s+", stripped)
        if not parts:
            continue
        name = parts[0]
        if in_nodes:
            status = parts[1] if len(parts) > 1 else ""
            nodes.append({"name": name, "status": status})
            if status and status.lower() != "ready":
                abnormal = True
            continue
        if in_resources:
            if name.startswith("pod/"):
                pod_status = parts[2] if len(parts) > 2 else ""
                pods.append({"name": name.removeprefix("pod/"), "ready": parts[1] if len(parts) > 1 else "", "status": pod_status})
                if pod_status and pod_status.lower() not in {"running", "completed", "succeeded"}:
                    abnormal = True
            elif name.startswith("deployment"):
                deployments.append({"name": name, "ready": parts[1] if len(parts) > 1 else ""})
            elif name.startswith("service/") or name == "kubernetes":
                services.append({"name": name})

    summary = (
        f"Inventory collected: node_count={len(nodes)}, pod_count={len(pods)}, "
        f"deployment_count={len(deployments)}, service_count={len(services)}, abnormal={abnormal}."
    )
    return {
        "summary": summary,
        "node_count": len(nodes),
        "pod_count": len(pods),
        "deployment_count": len(deployments),
        "service_count": len(services),
        "nodes": nodes[:30],
        "pods": pods[:50],
        "deployments": deployments[:50],
        "services": services[:50],
        "abnormal": abnormal,
        "evidence": important_lines(result, ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "Failed", "Warning"))[:10],
    }


def summarize_logs(result: str) -> dict[str, Any]:
    lines = result.splitlines()
    error_lines = [
        line
        for line in lines
        if re.search(r"\b(error|exception|traceback|fatal|panic|failed|refused|timeout)\b", line, re.I)
    ]
    warning_lines = [line for line in lines if re.search(r"\b(warn|warning)\b", line, re.I)]
    ready_lines = [line for line in lines if re.search(r"ready|started|accept connections|listening", line, re.I)]
    evidence = [*error_lines[:5], *warning_lines[:5], *ready_lines[:5]]
    abnormal = bool(error_lines)
    return {
        "summary": (
            f"Scanned logs: error_count={len(error_lines)}, "
            f"warning_count={len(warning_lines)}, ready_signal_count={len(ready_lines)}."
        ),
        "error_count": len(error_lines),
        "warning_count": len(warning_lines),
        "ready_signal_count": len(ready_lines),
        "evidence": [truncate(line, 300) for line in evidence[:12]],
        "abnormal": abnormal,
    }


def summarize_runtime_diagnostics(result: str) -> dict[str, Any]:
    sections = split_tool_sections(result, "Environment adapter:")
    diagnostics = []
    abnormal = False
    for title, body in sections:
        text = f"{title}\n{body}"
        resource = extract_regex(text, r"^Resolved resource:\s*(.+)$", re.MULTILINE)
        connected = int_or_none(extract_regex(text, r"^connected_clients:(\d+)$", re.MULTILINE))
        blocked = int_or_none(extract_regex(text, r"^blocked_clients:(\d+)$", re.MULTILINE))
        maxclients = int_or_none(extract_regex(text, r"^maxclients:(\d+)$", re.MULTILINE))
        used_memory = extract_regex(text, r"^used_memory_human:(.+)$", re.MULTILINE)
        total_connections = int_or_none(extract_regex(text, r"^total_connections_received:(\d+)$", re.MULTILINE))
        slowlog_lines = [
            line
            for line in text.splitlines()
            if re.match(r"^\d+\)", line.strip()) or re.match(r"^\d+\s*$", line.strip())
        ]
        blocked_abnormal = blocked is not None and blocked > 0
        client_ratio_abnormal = (
            connected is not None and maxclients is not None and maxclients > 0 and connected / maxclients >= 0.8
        )
        if blocked_abnormal or client_ratio_abnormal:
            abnormal = True
        diagnostics.append(
            {
                "resource": resource or "",
                "connected_clients": connected,
                "blocked_clients": blocked,
                "maxclients": maxclients,
                "used_memory_human": used_memory.strip() if used_memory else "",
                "total_connections_received": total_connections,
                "slowlog_signal_lines": len(slowlog_lines),
            }
        )
    return {
        "summary": f"Collected runtime diagnostics for {len(diagnostics)} resource(s); abnormal={abnormal}.",
        "runtime": diagnostics[:30],
        "abnormal": abnormal,
    }


def summarize_workload_topology(result: str) -> dict[str, Any]:
    data = extract_json_payload(result)
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    topologies = []
    abnormal = False
    for item in items:
        if not isinstance(item, dict):
            continue
        workload = item.get("workload", {})
        summary = item.get("summary", {})
        node_distribution = item.get("node_distribution", {})
        if summary.get("single_node_concentration") or not summary.get("all_pods_ready", True):
            abnormal = True
        topologies.append(
            {
                "resource": workload.get("resource"),
                "replicas": workload.get("replicas"),
                "ready_replicas": workload.get("ready_replicas"),
                "available_replicas": workload.get("available_replicas"),
                "pod_count": summary.get("pod_count"),
                "ready_pod_count": summary.get("ready_pod_count"),
                "all_pods_ready": summary.get("all_pods_ready"),
                "node_distribution": node_distribution,
                "services": [service.get("name") for service in item.get("services", []) if isinstance(service, dict)],
            }
        )
    return {
        "summary": f"Collected workload topology for {len(topologies)} workload(s); abnormal={abnormal}.",
        "topology": topologies[:30],
        "abnormal": abnormal,
    }


def summarize_fallback(result: str) -> dict[str, Any]:
    return {
        "summary": truncate(result, 800),
        "abnormal": False,
    }


def extract_json_payload(text: str) -> Any:
    start = text.find("{")
    list_start = text.find("[")
    if list_start >= 0 and (start < 0 or list_start < start):
        start = list_start
    if start < 0:
        return None
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    if end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def split_tool_sections(text: str, marker: str) -> list[tuple[str, str]]:
    if marker not in text:
        return [("", text)]
    sections = []
    for chunk in text.split(marker):
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        sections.append((title, body))
    return sections


def important_lines(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    return [
        truncate(line.strip(), 300)
        for line in text.splitlines()
        if any(keyword in line.lower() for keyword in lowered_keywords)
    ]


def looks_like_metric(cpu: str, memory: str) -> bool:
    return bool(re.match(r"^\d+m?$", cpu)) and bool(re.match(r"^\d+(\.\d+)?[KMGTE]i?$", memory, re.I))


def cpu_millicores(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("m"):
        return int(float(value[:-1] or 0))
    if re.match(r"^\d+(\.\d+)?$", value):
        return int(float(value) * 1000)
    return 0


def memory_mib(value: str) -> float:
    match = re.match(r"^(\d+(?:\.\d+)?)([kmgt]i?)?$", value.strip(), re.I)
    if not match:
        return 0
    amount = float(match.group(1))
    unit = (match.group(2) or "mi").lower()
    factors = {
        "k": 1 / 1024,
        "ki": 1 / 1024,
        "m": 1,
        "mi": 1,
        "g": 1024,
        "gi": 1024,
        "t": 1024 * 1024,
        "ti": 1024 * 1024,
    }
    return amount * factors.get(unit, 1)


def extract_regex(text: str, pattern: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else ""


def int_or_none(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"
