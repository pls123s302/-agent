import json
import re
from typing import Any

from ops_agent.observation_summarizer import summarize_observation, summarize_observations_for_model
from ops_agent.state import AgentState


DEFAULT_ENVIRONMENT_ID = "local-docker"


class OpsAgentNodes:
    def __init__(
        self,
        llm: Any,
        prompts: dict[str, str],
        tools: list[Any],
        max_tool_steps: int,
        max_replans: int,
        runtime_warnings: list[str] | None = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.tools = {tool.name: tool for tool in tools}
        self.max_tool_steps = max_tool_steps
        self.max_replans = max_replans
        self.runtime_warnings = runtime_warnings or []

    async def router_agent(self, state: AgentState) -> AgentState:
        heuristic_route = route_by_heuristic(state)
        if heuristic_route:
            return {"route": heuristic_route, "next_action": ""}

        response = await self.llm.ainvoke([("system", self.prompts["router"]), *state["messages"]])
        return {"route": parse_route(response.content), "next_action": ""}

    async def plan_agent(self, state: AgentState) -> AgentState:
        fallback = build_fallback_plan(state)
        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["planner"]),
                ("system", f"Current route: {state.get('route', '')}"),
                ("system", f"Current environment: {state.get('environment_id') or DEFAULT_ENVIRONMENT_ID}"),
                *state["messages"],
            ]
        )
        plan = normalize_plan(parse_json_object(response.content), fallback)
        environment_id = state.get("environment_id") or plan.get("environment_id") or DEFAULT_ENVIRONMENT_ID
        plan["environment_id"] = environment_id
        return {
            "plan": plan,
            "route": plan.get("route", state.get("route", "chat")),
            "environment_id": environment_id,
            "environment_type": state.get("environment_type", "docker"),
            "current_step_index": 0,
            "observations": [],
            "next_action": "",
        }

    async def replan_agent(self, state: AgentState) -> AgentState:
        if state.get("replan_count", 0) >= self.max_replans:
            return {"next_action": "final"}

        fallback = append_fallback_replan_steps(state)
        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["replanner"]),
                ("system", format_plan_context(state)),
                ("system", format_observations_context(state)),
                latest_user_message(state),
            ]
        )
        plan = normalize_plan(parse_json_object(response.content), fallback)
        return {
            "plan": plan,
            "current_step_index": min(state.get("current_step_index", 0), len(plan.get("steps", []))),
            "replan_count": state.get("replan_count", 0) + 1,
            "next_action": "continue" if has_remaining_steps({**state, "plan": plan}) else "final",
        }

    async def execute_step(self, state: AgentState) -> AgentState:
        if state.get("tool_steps", 0) >= self.max_tool_steps:
            return {"next_action": "final"}

        step_index = state.get("current_step_index", 0)
        steps = state.get("plan", {}).get("steps", [])
        if step_index >= len(steps):
            return {"next_action": "final"}

        step = normalize_step(steps[step_index])
        action = canonical_action(step.get("action", ""))
        tool = self.tools.get(action)
        if tool is None:
            args = build_tool_args({**step, "action": action}, state)
            print_tool_trace_start(step_index, len(steps), action, args, step.get("reason", ""))
            observation = {
                "step_index": step_index,
                "step": step,
                "tool_args": args,
                "ok": False,
                "result": f"Tool '{action}' is not available. Available tools: {', '.join(sorted(self.tools))}",
            }
            observation["summary"] = summarize_observation(observation)
            print_tool_trace_result(observation)
        else:
            args = build_tool_args({**step, "action": action}, state)
            print_tool_trace_start(step_index, len(steps), action, args, step.get("reason", ""))
            try:
                result = await tool.ainvoke(args)
                raw_result = stringify_result(result)
                observation = {
                    "step_index": step_index,
                    "step": {**step, "action": action},
                    "tool_args": args,
                    "ok": True,
                    "result": raw_result,
                }
            except Exception as error:  # noqa: BLE001
                observation = {
                    "step_index": step_index,
                    "step": {**step, "action": action},
                    "tool_args": args,
                    "ok": False,
                    "result": f"{type(error).__name__}: {error}",
                }
            observation["summary"] = summarize_observation(observation)
            print_tool_trace_result(observation)

        observations = [*state.get("observations", []), observation]
        plan = expand_plan_after_target_resolution(state, observation, step_index + 1)
        return {
            "plan": plan,
            "observations": observations,
            "current_step_index": step_index + 1,
            "tool_steps": state.get("tool_steps", 0) + 1,
            "next_action": "",
        }

    async def reflect_agent(self, state: AgentState) -> AgentState:
        target_decision = latest_target_resolution_decision(state)
        if target_decision == "ask_user":
            return {"next_action": "ask_user"}

        if has_remaining_steps(state):
            return {"next_action": "continue"}
        if state.get("tool_steps", 0) >= self.max_tool_steps:
            return {"next_action": "final"}

        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["reflect"]),
                ("system", format_plan_context(state)),
                ("system", format_observations_context(state)),
                latest_user_message(state),
            ]
        )
        action = parse_next_action(response.content)
        if action == "replan" and state.get("replan_count", 0) >= self.max_replans:
            action = "final"
        return {"next_action": action}

    async def chat_agent(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke([("system", self.prompts["chat"]), *state["messages"]])
        return {"messages": [("assistant", clean_model_output(response.content))]}

    async def log_agent(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke([("system", self.prompts["log"]), *state["messages"]])
        return {"messages": [("assistant", clean_model_output(response.content))]}

    async def final_agent(self, state: AgentState) -> AgentState:
        if state.get("next_action") == "ask_user":
            ask_text = format_target_clarification(state)
            if ask_text:
                return {"messages": [("assistant", ask_text)]}

        response = await self.llm.ainvoke(
            [
                ("system", self.prompts["final"]),
                ("system", format_plan_context(state)),
                ("system", format_observations_context(state)),
                latest_user_message(state),
            ]
        )
        return {"messages": [("assistant", clean_final_output(response.content))]}


def route_by_heuristic(state: AgentState) -> str:
    user_text = latest_user_text(state)
    lowered = user_text.lower()

    if looks_like_pasted_log(user_text):
        return "log"

    if looks_like_ops_request(lowered):
        return "ops"

    return ""


def looks_like_ops_request(lowered: str) -> bool:
    ops_keywords = (
        "docker",
        "container",
        "containers",
        "pod",
        "pods",
        "service",
        "deployment",
        "statefulset",
        "daemonset",
        "namespace",
        "kubectl",
        "k8s",
        "kubernetes",
        "redis",
        "mysql",
        "mongodb",
        "mongo",
        "neo4j",
        "cpu",
        "memory",
        "network",
        "disk",
        "log",
        "logs",
        "slowlog",
        "connections",
        "clients",
        "容器",
        "服务",
        "工作负载",
        "集群",
        "日志",
        "状态",
        "运行",
        "异常",
        "排查",
        "诊断",
        "巡检",
        "资源",
        "指标",
        "内存",
        "网络",
        "磁盘",
        "连接数",
        "慢查询",
        "拓扑",
        "有哪些",
        "都有哪些",
    )
    return any(keyword in lowered for keyword in ops_keywords)


def build_fallback_plan(state: AgentState) -> dict[str, Any]:
    user_text = latest_user_text(state)
    route = state.get("route", "") or route_by_heuristic(state) or "chat"
    extracted_targets = extract_targets(user_text)
    targets = [] if is_inventory_query(user_text) or is_generic_inventory_targets(extracted_targets) else extracted_targets
    requested = infer_requested_evidence(user_text, route)
    environment_id = state.get("environment_id") or DEFAULT_ENVIRONMENT_ID
    return {
        "objective": user_text,
        "route": route,
        "environment_id": environment_id,
        "targets": targets,
        "requested_evidence": requested,
        "steps": build_steps(route, targets, requested, user_text),
        "needs_user_confirmation": False,
        "notes": "fallback plan generated by deterministic rules",
    }


def append_fallback_replan_steps(state: AgentState) -> dict[str, Any]:
    plan = dict(state.get("plan", {}))
    steps = list(plan.get("steps", []))

    if plan.get("route") in {"ops", "metrics"} and "logs" not in plan.get("requested_evidence", []):
        candidate_targets = abnormal_targets_from_observations(state)
        for target in candidate_targets[:2]:
            steps.append(
                {
                    "action": "query_logs",
                    "target": target,
                    "args": {"tail": 200},
                    "reason": "指标结果中出现该目标，补充日志证据判断是否异常。",
                }
            )
        if candidate_targets:
            plan["route"] = "docker"
            plan["requested_evidence"] = unique_preserve_order([*plan.get("requested_evidence", []), "logs"])

    plan["steps"] = steps
    return normalize_plan(plan, build_fallback_plan(state))


def abnormal_targets_from_observations(state: AgentState) -> list[str]:
    targets = []
    skipped = {"", "all", "host", "docker", "restricted", "environment", "adapter"}
    for observation in state.get("observations", []):
        summary = observation.get("summary", {})
        if not isinstance(summary, dict) or not summary.get("abnormal"):
            continue
        target = str(summary.get("target") or observation.get("tool_args", {}).get("target") or "")
        if target.lower() not in skipped:
            targets.append(target)
        targets.extend(resources_from_summary(summary))
    return unique_preserve_order([target for target in targets if target.lower() not in skipped])


def resources_from_summary(summary: dict[str, Any]) -> list[str]:
    resources = []
    for key in ("workloads", "runtime", "topology"):
        values = summary.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            resource = item.get("resource") or item.get("name")
            if isinstance(resource, str) and resource:
                resources.append(resource)
    for candidate in summary.get("candidates", []):
        if isinstance(candidate, str):
            resources.append(candidate)
    return resources


def normalize_plan(model_plan: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    if not model_plan:
        return fallback

    plan = {**fallback, **model_plan}
    plan["route"] = normalize_route(safe_string(plan.get("route")) or fallback["route"])
    plan["objective"] = safe_string(plan.get("objective")) or fallback["objective"]
    plan["environment_id"] = safe_string(plan.get("environment_id")) or fallback.get("environment_id", DEFAULT_ENVIRONMENT_ID)
    plan["targets"] = normalize_string_list(plan.get("targets")) or fallback["targets"]
    plan["requested_evidence"] = normalize_string_list(plan.get("requested_evidence")) or fallback["requested_evidence"]
    plan["steps"] = normalize_steps(plan.get("steps")) or fallback["steps"]
    plan["needs_user_confirmation"] = bool(plan.get("needs_user_confirmation", False))
    plan["notes"] = safe_string(plan.get("notes"))
    return plan


def build_steps(route: str, targets: list[str], requested: list[str], user_text: str) -> list[dict[str, Any]]:
    if route in {"ops", "docker"}:
        if is_inventory_query(user_text) or is_generic_inventory_targets(targets):
            return [
                {
                    "action": "query_status",
                    "target": "",
                    "args": {},
                    "reason": "用户需要查看当前环境中的资源清单。",
                }
            ]

        if needs_topology_from_text(user_text) and not targets:
            return [
                {
                    "action": "query_status",
                    "target": "",
                    "args": {},
                    "reason": "用户询问系统级拓扑，先获取节点和资源清单。",
                },
                {
                    "action": "query_workload_topology",
                    "target": "all",
                    "args": {},
                    "reason": "用户询问系统级拓扑，获取所有工作负载、Service、Pod 和 Node 分布。",
                },
            ]

        if is_global_health_query(user_text, targets):
            return [
                {
                    "action": "query_status",
                    "target": "",
                    "args": {},
                    "reason": "用户询问整体运行状态，先发现资源清单和状态。",
                },
                {
                    "action": "query_metrics",
                    "target": "all",
                    "args": {"metric_types": "cpu,memory,network,disk", "top_n": 8},
                    "reason": "用户询问整体运行状态，补充全局资源指标。",
                },
            ]

        steps = []
        for target in targets or [""]:
            if target:
                steps.append(
                    {
                        "action": "resolve_target",
                        "target": target,
                        "args": {"query": target},
                        "reason": "用户给出的目标可能是简称，先解析候选目标。",
                    }
                )
            if needs_metrics_from_text(user_text):
                steps.append(
                    {
                        "action": "query_metrics",
                        "target": target or "all",
                        "args": {"metric_types": ",".join(requested_metrics_from_text(user_text)), "top_n": 8},
                        "reason": "用户需要查看目标资源指标。",
                    }
                )
            runtime_checks = runtime_checks_from_text(user_text)
            if needs_topology_from_text(user_text):
                steps.append(
                    {
                        "action": "query_workload_topology",
                        "target": target,
                        "args": {},
                        "reason": "用户需要查看工作负载、Pod 和 Node 的拓扑分布。",
                    }
                )
            if runtime_checks:
                steps.append(
                    {
                        "action": "query_runtime_diagnostics",
                        "target": target,
                        "args": {"profile": "auto", "checks": ",".join(runtime_checks)},
                        "reason": "用户需要查看应用内部运行时状态。",
                    }
                )
            if "logs" in requested:
                steps.append(
                    {
                        "action": "query_logs",
                        "target": target,
                        "args": {"tail": infer_log_tail(user_text) or 100},
                        "reason": "用户需要查看目标日志。",
                    }
                )
            if "status" in requested:
                steps.append(
                    {
                        "action": "query_status",
                        "target": target,
                        "args": {},
                        "reason": "用户需要查看目标状态和元信息。",
                    }
                )
        return steps

    if route == "metrics":
        return [
            {
                "action": "query_metrics",
                "target": ",".join(targets) if targets else "all",
                "args": {
                    "metric_types": ",".join(requested or ["cpu", "memory", "network", "disk"]),
                    "top_n": 8,
                },
                "reason": "先收集全局指标，判断资源层面是否异常。",
            }
        ]

    return []


def is_global_health_query(user_text: str, targets: list[str]) -> bool:
    if targets:
        return False
    lowered = user_text.lower()
    keywords = (
        "系统运行状态",
        "系统现在",
        "整体状态",
        "全局",
        "环境",
        "集群",
        "巡检",
        "有没有异常",
        "是否异常",
        "health",
        "overall",
    )
    return any(keyword in lowered for keyword in keywords)


def needs_metrics_from_text(user_text: str) -> bool:
    lowered = user_text.lower()
    keywords = (
        "运行情况",
        "状态如何",
        "怎么样",
        "健康",
        "cpu",
        "内存",
        "memory",
        "网络",
        "流量",
        "磁盘",
        "disk",
        "资源",
        "性能",
        "占用",
    )
    return any(keyword in lowered for keyword in keywords)


def requested_metrics_from_text(user_text: str) -> list[str]:
    lowered = user_text.lower()
    metrics = []
    mapping = {
        "cpu": "cpu",
        "内存": "memory",
        "memory": "memory",
        "网络": "network",
        "流量": "network",
        "磁盘": "disk",
        "disk": "disk",
    }
    for keyword, metric in mapping.items():
        if keyword in lowered:
            metrics.append(metric)
    return unique_preserve_order(metrics) or ["cpu", "memory", "network", "disk"]


def runtime_checks_from_text(user_text: str) -> list[str]:
    lowered = user_text.lower()
    checks = []
    if any(keyword in lowered for keyword in ("连接数", "连接", "client", "clients", "connections")):
        checks.append("connections")
    if any(keyword in lowered for keyword in ("慢查询", "slowlog", "slow query", "slow_queries")):
        checks.append("slowlog")
    if any(keyword in lowered for keyword in ("内部状态", "运行时", "runtime", "info")):
        checks.extend(["connections", "stats"])
    if any(keyword in lowered for keyword in ("命中率", "hit", "miss", "keyspace")):
        checks.extend(["stats", "keyspace"])
    if any(keyword in lowered for keyword in ("应用内存", "redis内存", "runtime memory")):
        checks.append("memory")
    return unique_preserve_order(checks)


def needs_topology_from_text(user_text: str) -> bool:
    lowered = user_text.lower()
    keywords = (
        "拓扑",
        "分布",
        "几个pod",
        "几个 pod",
        "多少pod",
        "多少 pod",
        "哪些节点",
        "哪个节点",
        "node",
        "nodes",
        "副本",
        "replica",
        "replicas",
        "service对应",
        "service 对应",
        "deployment对应",
        "deployment 对应",
    )
    return any(keyword in lowered for keyword in keywords)


def infer_requested_evidence(user_text: str, route: str) -> list[str]:
    lowered = user_text.lower()
    evidence = []
    if route in {"ops", "docker"}:
        if "日志" in lowered or "log" in lowered:
            evidence.append("logs")
        if any(word in lowered for word in ("状态", "status", "info", "inspect", "健康", "重启")):
            evidence.append("status")
        if needs_metrics_from_text(user_text):
            evidence.append("status")
            evidence.extend(requested_metrics_from_text(user_text))
        return evidence or ["status"]

    if route == "metrics":
        checks = {
            "cpu": "cpu",
            "内存": "memory",
            "memory": "memory",
            "网络": "network",
            "流量": "network",
            "磁盘": "disk",
            "disk": "disk",
        }
        for keyword, evidence_name in checks.items():
            if keyword in lowered:
                evidence.append(evidence_name)
        return unique_preserve_order(evidence) or ["cpu", "memory", "network", "disk"]

    return []


def is_inventory_query(user_text: str) -> bool:
    lowered = user_text.lower()
    inventory_keywords = ("有哪些", "都有哪些", "列表", "列出", "查看", "当前", "目前", "get", "list")
    inventory_targets = (
        "pod",
        "pods",
        "service",
        "svc",
        "deployment",
        "deploy",
        "statefulset",
        "daemonset",
        "工作负载",
        "服务",
        "集群",
        "namespace",
        "命名空间",
        "container",
        "containers",
        "容器",
    )
    return any(keyword in lowered for keyword in inventory_keywords) and any(
        keyword in lowered for keyword in inventory_targets
    )


def is_generic_inventory_targets(targets: list[str]) -> bool:
    generic_targets = {
        "pod",
        "pods",
        "service",
        "services",
        "svc",
        "deployment",
        "deployments",
        "deploy",
        "statefulset",
        "statefulsets",
        "daemonset",
        "daemonsets",
        "namespace",
        "namespaces",
        "ns",
    }
    return bool(targets) and all(target.lower() in generic_targets for target in targets)


def build_tool_args(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    action = canonical_action(step.get("action", ""))
    target = step.get("target", "")
    args = step.get("args", {})
    if not isinstance(args, dict):
        args = {}

    environment_id = state.get("environment_id") or args.get("environment_id") or state.get("plan", {}).get("environment_id") or ""
    if action == "resolve_target":
        return {
            "query": args.get("query") or target,
            "intent": latest_user_text(state),
            "environment_id": environment_id,
        }
    if action == "query_logs":
        return {
            "target": resolve_target_from_observations(target, state) or target,
            "tail": int(args.get("tail", infer_log_tail(latest_user_text(state)) or 100)),
            "filter_keywords": str(args.get("filter_keywords", "")),
            "environment_id": environment_id,
        }
    if action == "query_status":
        resolved_target = resolve_target_from_observations(target, state) or target
        if resolved_target.lower() == "all":
            resolved_target = ""
        return {
            "target": resolved_target,
            "fields": str(args.get("fields", "summary")),
            "environment_id": environment_id,
        }
    if action == "query_metrics":
        requested = args.get("metric_types") or ",".join(state.get("plan", {}).get("requested_evidence", []))
        if isinstance(requested, list):
            requested = ",".join(str(item).strip() for item in requested if str(item).strip())
        return {
            "target": target or "all",
            "metric_types": requested or "cpu,memory,network,disk",
            "top_n": int(args.get("top_n", 8)),
            "environment_id": environment_id,
        }
    if action == "query_runtime_diagnostics":
        checks = args.get("checks") or ",".join(runtime_checks_from_text(latest_user_text(state)))
        if isinstance(checks, list):
            checks = ",".join(str(item).strip() for item in checks if str(item).strip())
        return {
            "target": resolve_target_from_observations(target, state) or target,
            "profile": str(args.get("profile", "auto")),
            "checks": checks or "connections",
            "environment_id": environment_id,
        }
    if action == "query_workload_topology":
        return {
            "target": resolve_target_from_observations(target, state) or target,
            "environment_id": environment_id,
        }
    return args


def expand_plan_after_target_resolution(
    state: AgentState,
    observation: dict[str, Any],
    next_step_index: int,
) -> dict[str, Any]:
    step = observation.get("step", {})
    if canonical_action(step.get("action", "")) != "resolve_target" or not observation.get("ok"):
        return state.get("plan", {})

    resolution = extract_structured_resolution(observation.get("result", ""))
    if resolution.get("suggested_mode") != "batch_expand":
        return state.get("plan", {})

    resources = [
        str(candidate.get("resource", "")).strip()
        for candidate in resolution.get("candidates", [])
        if isinstance(candidate, dict) and str(candidate.get("resource", "")).strip()
    ]
    if not resources:
        return state.get("plan", {})

    original_target = str(resolution.get("query") or step.get("target") or "").strip()
    plan = dict(state.get("plan", {}))
    steps = list(plan.get("steps", []))
    prefix = steps[:next_step_index]
    remaining = steps[next_step_index:]
    expanded_remaining = []
    changed = False

    for raw_step in remaining:
        candidate_step = normalize_step(raw_step)
        action = canonical_action(candidate_step.get("action", ""))
        if is_batch_expandable_action(action) and should_expand_step_target(candidate_step, original_target):
            for resource in resources:
                expanded_step = {
                    **candidate_step,
                    "target": resource,
                    "args": dict(candidate_step.get("args", {})),
                    "reason": f"{candidate_step.get('reason', '')} batch_expand resolved target: {resource}".strip(),
                }
                expanded_remaining.append(expanded_step)
            changed = True
        else:
            expanded_remaining.append(raw_step)

    if not changed:
        return plan

    plan["steps"] = [*prefix, *expanded_remaining]
    plan["notes"] = append_note(plan.get("notes", ""), f"batch_expand {original_target} -> {', '.join(resources)}")
    return plan


def is_batch_expandable_action(action: str) -> bool:
    return action in {
        "query_logs",
        "query_status",
        "query_metrics",
        "query_runtime_diagnostics",
        "query_workload_topology",
    }


def should_expand_step_target(step: dict[str, Any], original_target: str) -> bool:
    target = safe_string(step.get("target")).lower()
    args = step.get("args", {})
    arg_target = safe_string(args.get("target") if isinstance(args, dict) else "").lower()
    original = original_target.lower()
    expandable_targets = {"", "all", "*", original}
    return target in expandable_targets or arg_target in expandable_targets


def append_note(notes: str, note: str) -> str:
    if not notes:
        return note
    if note in notes:
        return notes
    return f"{notes}; {note}"


def canonical_action(action: str) -> str:
    aliases = {
        "resolve_container": "resolve_target",
        "query_container_logs": "query_logs",
        "query_container_info": "query_status",
        "query_container_metrics": "query_metrics",
        "query_app_runtime": "query_runtime_diagnostics",
        "query_runtime": "query_runtime_diagnostics",
        "query_topology": "query_workload_topology",
        "query_workload": "query_workload_topology",
        "query_workload_diagnostics": "query_workload_topology",
        "query_topology_diagnostics": "query_workload_topology",
    }
    return aliases.get(action, action)


def resolve_target_from_observations(target: str, state: AgentState) -> str:
    if not target:
        return ""
    for observation in reversed(state.get("observations", [])):
        step = observation.get("step", {})
        if canonical_action(step.get("action", "")) != "resolve_target" or step.get("target") != target:
            continue
        resolution = extract_structured_resolution(observation.get("result", ""))
        if resolution:
            mode = resolution.get("suggested_mode")
            candidates = resolution.get("candidates", [])
            if mode == "direct" and len(candidates) == 1:
                return str(candidates[0].get("resource", ""))
            if mode == "batch_expand":
                return target
        candidates = extract_target_names(observation.get("result", ""))
        if len(candidates) == 1:
            return candidates[0]
    return ""


def latest_target_resolution_decision(state: AgentState) -> str:
    observations = state.get("observations", [])
    if not observations:
        return ""
    observation = observations[-1]
    if not observation.get("ok"):
        return ""
    step = observation.get("step", {})
    if canonical_action(step.get("action", "")) != "resolve_target":
        return ""
    resolution = extract_structured_resolution(observation.get("result", ""))
    if not resolution:
        return ""
    mode = str(resolution.get("suggested_mode", ""))
    if mode == "ask_user":
        return "ask_user"
    return ""


def format_target_clarification(state: AgentState) -> str:
    for observation in reversed(state.get("observations", [])):
        step = observation.get("step", {})
        if canonical_action(step.get("action", "")) != "resolve_target":
            continue
        resolution = extract_structured_resolution(observation.get("result", ""))
        if not resolution or resolution.get("suggested_mode") != "ask_user":
            continue
        query = resolution.get("query") or step.get("target") or "目标"
        candidates = resolution.get("candidates", [])
        lines = [f"`{query}` 匹配到多个目标，请你指定要查哪一个，或者明确说“全部都查”:"]
        for index, candidate in enumerate(candidates[:10], start=1):
            resource = candidate.get("resource") or f"{candidate.get('kind')}/{candidate.get('name')}"
            labels = candidate.get("labels") or {}
            label_text = ", ".join(f"{key}={value}" for key, value in labels.items())
            suffix = f" labels: {label_text}" if label_text else ""
            lines.append(f"{index}. `{resource}`{suffix}")
        return "\n".join(lines)
    return ""


def extract_structured_resolution(text: str) -> dict[str, Any]:
    data = extract_json_payload(text)
    if isinstance(data, dict) and "suggested_mode" in data and "candidates" in data:
        return data
    return {}


def extract_json_payload(text: str) -> Any:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def extract_target_names(text: str) -> list[str]:
    names = []
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        data = None

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("name", "Name", "Names", "container_name", "pod", "workload"):
                raw = value.get(key)
                if isinstance(raw, str):
                    names.append(raw.lstrip("/"))
                elif isinstance(raw, list):
                    names.extend(str(item).lstrip("/") for item in raw)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    if data is not None:
        visit(data)
    if not names:
        names.extend(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*-[a-zA-Z0-9_.-]*\b", text))
    return unique_preserve_order([name for name in names if name])


def infer_log_tail(latest_text: str) -> int | None:
    match = re.search(r"(\d+)\s*条", latest_text)
    if not match:
        return None
    return min(int(match.group(1)), 500)


def has_remaining_steps(state: AgentState) -> bool:
    return state.get("current_step_index", 0) < len(state.get("plan", {}).get("steps", []))


def latest_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, tuple) and len(message) >= 2 and message[0] == "human":
            return str(message[1])
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return ""


def latest_user_message(state: AgentState) -> tuple[str, str]:
    return ("human", latest_user_text(state))


def print_tool_trace_start(
    step_index: int,
    total_steps: int,
    action: str,
    args: dict[str, Any],
    reason: str,
) -> None:
    print(f"\n[tool {step_index + 1}/{total_steps}] {action}", flush=True)
    print(f"args: {json.dumps(args, ensure_ascii=False)}", flush=True)
    if reason:
        print(f"reason: {reason}", flush=True)


def print_tool_trace_result(observation: dict[str, Any]) -> None:
    status = "ok" if observation.get("ok") else "failed"
    result = truncate_text(str(observation.get("result", "")), 800)
    print(f"status: {status}", flush=True)
    print(f"result_preview:\n{result}\n", flush=True)


def looks_like_pasted_log(text: str) -> bool:
    lowered = text.lower()
    log_markers = ("error", "warn", "exception", "traceback", "timeout", "failed", "stack", " at ")
    if "分析日志" in lowered and any(marker in lowered for marker in log_markers):
        return True
    return "\n" in text and any(marker in lowered for marker in log_markers)


def parse_route(content: str) -> str:
    allowed = {"chat", "log", "ops", "metrics", "docker"}
    parsed = parse_json_object(content)
    route = str(parsed.get("route", "")).lower()
    if route in allowed:
        return normalize_route(route)
    lowered = content.lower()
    for candidate in allowed:
        if candidate in lowered:
            return normalize_route(candidate)
    return "chat"


def normalize_route(route: str) -> str:
    return "ops" if route in {"metrics", "docker"} else route


def parse_next_action(content: str) -> str:
    allowed = {"continue", "replan", "ask_user", "final"}
    parsed = parse_json_object(content)
    action = str(parsed.get("next_action", "")).lower()
    if action in allowed:
        return action
    if isinstance(parsed.get("sufficient"), bool):
        return "final" if parsed["sufficient"] else "replan"
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
    return value if isinstance(value, dict) else {}


def extract_targets(user_text: str) -> list[str]:
    cleaned = re.sub(r"[，。？！、,?!]", " ", user_text)
    stop_words = {
        "最近",
        "日志",
        "容器",
        "看看",
        "一起",
        "说明",
        "什么",
        "多少",
        "是否",
        "异常",
        "当前",
        "本机",
        "网络",
        "流量",
        "内存",
        "磁盘",
        "资源",
        "占用",
        "和",
        "与",
        "的",
        "了",
        "都",
        "请",
        "帮我",
        "系统",
        "环境",
    }
    candidates = []
    for token in re.split(r"\s+|和|与|以及|、", cleaned):
        token = token.strip()
        if not token or token in stop_words:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        match = re.search(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", token)
        if match:
            value = match.group(0)
            if value.lower() not in {"log", "logs", "docker", "cpu", "memory", "network", "disk"}:
                candidates.append(value)
    return unique_preserve_order(candidates)


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                parts = [part.strip() for part in item.split("|") if part.strip()]
                result.extend(parts or [item.strip()])
            elif item is not None:
                result.append(str(item).strip())
        return [item for item in result if item]
    return []


def normalize_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [normalize_step(item) for item in value if isinstance(item, dict)]


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(step)
    normalized["action"] = canonical_action(safe_string(normalized.get("action")))
    normalized["target"] = safe_string(normalized.get("target"))
    args = normalized.get("args", {})
    normalized["args"] = args if isinstance(args, dict) else {}
    normalized["reason"] = safe_string(normalized.get("reason"))
    if "metric_types" in normalized and "metric_types" not in normalized["args"]:
        normalized["args"]["metric_types"] = normalized["metric_types"]
    if "tail" in normalized and "tail" not in normalized["args"]:
        normalized["args"]["tail"] = normalized["tail"]
    return normalized


def safe_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError:
        return str(result)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_final_output(text: str) -> str:
    text = clean_model_output(text)
    chinese_conclusion = re.search(r"(?:^|\n)\s*(?:\*\*)?结论(?:\*\*)?[：:]", text)
    english_conclusion = re.search(r"(?:^|\n)\s*(?:\*\*)?Conclusion(?:\*\*)?[：:]", text, flags=re.IGNORECASE)
    if chinese_conclusion and english_conclusion and english_conclusion.start() < chinese_conclusion.start():
        return text[chinese_conclusion.start() :].strip()
    if chinese_conclusion and chinese_conclusion.start() > 0:
        return text[chinese_conclusion.start() :].strip()
    marker_positions = [match.start() for match in re.finditer(r"(?:^|\n)\s*(?:\*\*)?结论(?:\*\*)?[：:]", text)]
    if len(marker_positions) >= 2:
        return text[marker_positions[-1] :].strip()
    return text


def format_plan_context(state: AgentState) -> str:
    return "Current structured plan:\n" + json.dumps(state.get("plan", {}), ensure_ascii=False, indent=2)


def format_observations_context(state: AgentState) -> str:
    summaries = summarize_observations_for_model(state.get("observations", []))
    return "Current observation summaries:\n" + json.dumps(summaries, ensure_ascii=False, indent=2)
