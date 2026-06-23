import json
import asyncio
from typing import Any

from langchain_core.tools import StructuredTool

from ops_agent.cadvisor import collect_cadvisor_summary


class EnvironmentRegistry:
    def __init__(self, config: dict[str, Any], docker_tools: list[Any]):
        environments = config.get("environments") or [
            {
                "id": "local-docker",
                "type": "docker",
                "default": True,
                "capabilities": ["target", "logs", "status", "metrics"],
            }
        ]
        self.environments = {environment["id"]: environment for environment in environments}
        self.default_environment_id = next(
            (environment["id"] for environment in environments if environment.get("default")),
            environments[0]["id"],
        )
        self.adapters = {
            "docker": DockerEnvironmentAdapter(config["cadvisor"], docker_tools),
            "kubernetes": KubernetesEnvironmentAdapter(config.get("kubernetes", {})),
        }

    def adapter_for(self, environment_id: str | None):
        environment = self.get_environment(environment_id)
        adapter = self.adapters.get(environment["type"])
        if adapter is None:
            raise RuntimeError(f"No adapter registered for environment type '{environment['type']}'.")
        return environment, adapter

    def get_environment(self, environment_id: str | None) -> dict[str, Any]:
        selected_id = environment_id or self.default_environment_id
        environment = self.environments.get(selected_id)
        if environment is None:
            available = ", ".join(sorted(self.environments))
            raise RuntimeError(f"Unknown environment '{selected_id}'. Available environments: {available}")
        return environment


class DockerEnvironmentAdapter:
    def __init__(self, cadvisor_config: dict[str, Any], docker_tools: list[Any]):
        self.cadvisor_config = cadvisor_config
        self.docker_tools = {tool.name: tool for tool in docker_tools}

    async def resolve_target(
        self,
        query: str,
        intent: str = "",
        environment: dict[str, Any] | None = None,
    ) -> str:
        search_tool = self._docker_tool("search_containers")
        result = await search_tool.ainvoke({"query": query})
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def query_logs(
        self,
        target: str,
        tail: int = 100,
        filter_keywords: str = "",
        environment: dict[str, Any] | None = None,
    ) -> str:
        logs_tool = self._docker_tool("get_container_logs")
        result = await logs_tool.ainvoke({"container_name": target, "tail": tail})
        text = str(result)
        if filter_keywords:
            keywords = [keyword.strip().lower() for keyword in filter_keywords.split("|") if keyword.strip()]
            if keywords:
                lines = [line for line in text.splitlines() if any(keyword in line.lower() for keyword in keywords)]
                return "\n".join(lines) if lines else "No log lines matched the filter keywords."
        return text

    async def query_status(
        self,
        target: str = "",
        fields: str = "summary",
        environment: dict[str, Any] | None = None,
    ) -> str:
        if target and target.lower() not in {"all", "*"}:
            inspect_tool = self._docker_tool("inspect_container")
            result = await inspect_tool.ainvoke({"container_name": target})
            return summarize_container_inspect(result, fields)

        list_tool = self._docker_tool("list_containers")
        result = await list_tool.ainvoke({"all_containers": True})
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def query_metrics(
        self,
        target: str = "all",
        metric_types: str = "cpu,memory",
        top_n: int = 8,
        environment: dict[str, Any] | None = None,
    ) -> str:
        requested = {item.strip().lower() for item in metric_types.split(",") if item.strip()}
        summary = await collect_cadvisor_summary({**self.cadvisor_config, "top_n": top_n}, requested)
        return (
            f"Environment adapter: docker\n"
            f"Target: {target}\n"
            f"Requested metric types: {', '.join(sorted(requested)) or 'cpu, memory'}\n\n"
            f"{summary}"
        )

    async def query_runtime_diagnostics(
        self,
        target: str,
        profile: str = "auto",
        checks: str = "connections",
        environment: dict[str, Any] | None = None,
    ) -> str:
        return (
            "Runtime diagnostics are not implemented for the Docker adapter yet. "
            "Use the Kubernetes adapter or add a read-only Docker exec/runtime gateway first."
        )

    async def query_workload_topology(
        self,
        target: str,
        environment: dict[str, Any] | None = None,
    ) -> str:
        return "Workload topology is not implemented for the Docker adapter yet."

    def _docker_tool(self, name: str):
        tool = self.docker_tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self.docker_tools))
            raise RuntimeError(f"Docker adapter requires tool '{name}'. Available tools: {available}")
        return tool


class KubernetesEnvironmentAdapter:
    def __init__(self, kubernetes_config: dict[str, Any]):
        self.kubectl_command = kubernetes_config.get("kubectl_command", "kubectl")
        self.timeout_seconds = int(kubernetes_config.get("timeout_seconds", 10))

    async def resolve_target(
        self,
        query: str,
        intent: str = "",
        environment: dict[str, Any] | None = None,
    ) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        raw_matches = await self._raw_target_matches(environment, namespace, query)
        candidates = await self._workload_candidates_from_matches(environment, namespace, raw_matches)
        suggested_mode = suggest_target_mode(len(candidates), intent or query)
        result = {
            "query": query,
            "intent": intent,
            "namespace": namespace,
            "ambiguous": len(candidates) > 1,
            "candidate_count": len(candidates),
            "suggested_mode": suggested_mode,
            "reason": target_mode_reason(suggested_mode, len(candidates)),
            "candidates": candidates,
            "raw_match_count": len(raw_matches),
            "raw_matches": raw_matches[:20],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def query_logs(
        self,
        target: str,
        tail: int = 100,
        filter_keywords: str = "",
        environment: dict[str, Any] | None = None,
    ) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        expanded_resources = await self._expand_workload_resources(environment, namespace, target)
        if len(expanded_resources) > 1:
            sections = []
            for resource in expanded_resources:
                output = await self._kubectl(environment, "logs", resource, "-n", namespace, "--tail", str(tail))
                sections.append(f"# Logs for {resource}\n{output}")
            text = "\n\n====================\n\n".join(sections)
            if filter_keywords:
                keywords = [keyword.strip().lower() for keyword in filter_keywords.split("|") if keyword.strip()]
                lines = [line for line in text.splitlines() if any(keyword in line.lower() for keyword in keywords)]
                return "\n".join(lines) if lines else "No log lines matched the filter keywords."
            return text

        resource = await self._resolve_log_resource(environment, namespace, target)
        output = await self._kubectl(environment, "logs", resource, "-n", namespace, "--tail", str(tail))
        if filter_keywords:
            keywords = [keyword.strip().lower() for keyword in filter_keywords.split("|") if keyword.strip()]
            lines = [line for line in output.splitlines() if any(keyword in line.lower() for keyword in keywords)]
            return "\n".join(lines) if lines else "No log lines matched the filter keywords."
        return output

    async def query_status(
        self,
        target: str = "",
        fields: str = "summary",
        environment: dict[str, Any] | None = None,
    ) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        if not target or target.lower() in {"all", "*"}:
            nodes = await self._kubectl(environment, "get", "nodes", "-o", "wide")
            resources = await self._kubectl(
                environment,
                "get",
                "pods,deployments,statefulsets,daemonsets,services",
                "-n",
                namespace,
                "-o",
                "wide",
            )
            return f"# Nodes\n{nodes}\n\n# Namespace resources: {namespace}\n{resources}"
        expanded_resources = await self._expand_workload_resources(environment, namespace, target)
        if len(expanded_resources) > 1:
            sections = []
            for resource in expanded_resources:
                output = await self._kubectl(environment, "describe", resource, "-n", namespace)
                sections.append(f"# Status for {resource}\n{output}")
            return "\n\n====================\n\n".join(sections)
        resource = await self._resolve_status_resource(environment, namespace, target)
        return await self._kubectl(environment, "describe", resource, "-n", namespace)

    async def query_metrics(
        self,
        target: str = "all",
        metric_types: str = "cpu,memory",
        top_n: int = 8,
        environment: dict[str, Any] | None = None,
    ) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        try:
            if target and target != "all":
                output = await self._kubectl(environment, "top", "pod", "-n", namespace)
                pod_names = await self._pod_names_for_metric_target(environment, namespace, target)
                output = filter_kubectl_table_by_names(output, pod_names)
            else:
                output = await self._kubectl(environment, "top", "pod", "-n", namespace)
        except RuntimeError as error:
            return (
                "Kubernetes metrics are not available from kubectl top.\n"
                "Most likely metrics-server is not installed in this local cluster.\n"
                f"Original error: {error}"
            )
        return (
            "Environment adapter: kubernetes\n"
            f"Namespace: {namespace}\n"
            f"Target: {target}\n"
            f"Requested metric types: {metric_types}\n\n"
            f"{output}"
        )

    async def query_runtime_diagnostics(
        self,
        target: str,
        profile: str = "auto",
        checks: str = "connections",
        environment: dict[str, Any] | None = None,
    ) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        expanded_resources = await self._expand_workload_resources(environment, namespace, target)
        if len(expanded_resources) > 1:
            results = []
            for resource in expanded_resources:
                results.append(
                    await self._query_runtime_diagnostics_for_resource(
                        environment=environment,
                        namespace=namespace,
                        original_target=target,
                        resource=resource,
                        profile=profile,
                        checks=checks,
                    )
                )
            return "\n\n====================\n\n".join(results)

        resource = await self._resolve_exec_resource(environment, namespace, target)
        return await self._query_runtime_diagnostics_for_resource(
            environment=environment,
            namespace=namespace,
            original_target=target,
            resource=resource,
            profile=profile,
            checks=checks,
        )

    async def _query_runtime_diagnostics_for_resource(
        self,
        environment: dict[str, Any],
        namespace: str,
        original_target: str,
        resource: str,
        profile: str,
        checks: str,
    ) -> str:
        detected_profile = profile if profile and profile != "auto" else await self._detect_profile(
            environment,
            namespace,
            resource,
        )
        requested = parse_csv(checks) or ["connections"]
        if detected_profile != "redis":
            return (
                f"Runtime profile '{detected_profile}' is not implemented yet.\n"
                "Implemented profiles: redis.\n"
                f"Target: {original_target}\n"
                f"Checks: {', '.join(requested)}"
            )

        outputs = []
        for check in requested:
            normalized = check.lower().strip()
            if normalized in {"connections", "clients"}:
                outputs.append(
                    await self._kubectl_exec(
                        environment,
                        namespace,
                        resource,
                        "redis-cli",
                        "INFO",
                        "clients",
                    )
                )
            elif normalized in {"stats", "ops", "throughput"}:
                outputs.append(
                    await self._kubectl_exec(
                        environment,
                        namespace,
                        resource,
                        "redis-cli",
                        "INFO",
                        "stats",
                    )
                )
            elif normalized == "memory":
                outputs.append(
                    await self._kubectl_exec(
                        environment,
                        namespace,
                        resource,
                        "redis-cli",
                        "INFO",
                        "memory",
                    )
                )
            elif normalized == "keyspace":
                outputs.append(
                    await self._kubectl_exec(
                        environment,
                        namespace,
                        resource,
                        "redis-cli",
                        "INFO",
                        "keyspace",
                    )
                )
            elif normalized in {"slowlog", "slow_queries"}:
                outputs.append(
                    await self._kubectl_exec(
                        environment,
                        namespace,
                        resource,
                        "redis-cli",
                        "SLOWLOG",
                        "GET",
                        "10",
                    )
                )
            else:
                outputs.append(f"# Unsupported redis runtime check: {check}")

        return (
            "Environment adapter: kubernetes\n"
            f"Namespace: {namespace}\n"
            f"Target: {original_target}\n"
            f"Resolved resource: {resource}\n"
            f"Runtime profile: {detected_profile}\n"
            f"Checks: {', '.join(requested)}\n\n"
            + "\n\n---\n\n".join(outputs)
        )

    async def query_workload_topology(self, target: str, environment: dict[str, Any] | None = None) -> str:
        environment = environment or {}
        namespace = environment.get("namespace", "default")
        if not target or target.lower() in {"all", "*"}:
            topologies = []
            for resource in await self._all_workload_resources(environment, namespace):
                topology_text = await self._query_workload_topology_for_resource(environment, namespace, "all", resource)
                topologies.append(json.loads(topology_text))
            return json.dumps(topologies, ensure_ascii=False, indent=2)

        expanded_resources = await self._expand_workload_resources(environment, namespace, target)
        if len(expanded_resources) > 1:
            topologies = []
            for resource in expanded_resources:
                topology_text = await self._query_workload_topology_for_resource(environment, namespace, target, resource)
                topologies.append(json.loads(topology_text))
            return json.dumps(topologies, ensure_ascii=False, indent=2)

        workload_resource = await self._resolve_workload_resource(environment, namespace, target)
        return await self._query_workload_topology_for_resource(environment, namespace, target, workload_resource)

    async def _query_workload_topology_for_resource(
        self,
        environment: dict[str, Any],
        namespace: str,
        target: str,
        workload_resource: str,
    ) -> str:
        workload_json = await self._kubectl(environment, "get", workload_resource, "-n", namespace, "-o", "json")
        workload = json.loads(workload_json)
        workload_kind = workload.get("kind", "")
        workload_name = workload.get("metadata", {}).get("name", target)
        selector = workload.get("spec", {}).get("selector", {}).get("matchLabels", {})
        pods = await self._pods_for_selector(environment, namespace, selector)
        services = await self._services_for_selector(environment, namespace, selector)

        pod_items = []
        node_counts: dict[str, int] = {}
        ready_count = 0
        for pod in pods:
            pod_summary = summarize_pod(pod)
            pod_items.append(pod_summary)
            node = pod_summary.get("node") or "<pending>"
            node_counts[node] = node_counts.get(node, 0) + 1
            if pod_summary.get("ready"):
                ready_count += 1

        topology = {
            "namespace": namespace,
            "target": target,
            "workload": {
                "kind": workload_kind,
                "name": workload_name,
                "resource": workload_resource,
                "replicas": workload.get("spec", {}).get("replicas"),
                "ready_replicas": workload.get("status", {}).get("readyReplicas", 0),
                "available_replicas": workload.get("status", {}).get("availableReplicas", 0),
                "selector": selector,
                "labels": workload.get("metadata", {}).get("labels", {}),
            },
            "services": [summarize_service(service) for service in services],
            "pods": pod_items,
            "node_distribution": node_counts,
            "summary": {
                "pod_count": len(pod_items),
                "ready_pod_count": ready_count,
                "all_pods_ready": len(pod_items) == ready_count,
                "single_node_concentration": len(node_counts) == 1 and len(pod_items) > 1,
            },
        }
        return json.dumps(topology, ensure_ascii=False, indent=2)

    async def _resolve_log_resource(self, environment: dict[str, Any], namespace: str, target: str) -> str:
        if "/" in target:
            return target
        for resource_type in ("deployment", "statefulset", "daemonset", "pod"):
            if await self._resource_exists(environment, namespace, resource_type, target):
                return f"{resource_type}/{target}"
        candidates = await self._workload_candidates_from_matches(
            environment,
            namespace,
            await self._raw_target_matches(environment, namespace, target),
        )
        if len(candidates) == 1:
            return candidates[0]["resource"]
        if len(candidates) > 1:
            names = ", ".join(item["resource"] for item in candidates)
            raise RuntimeError(f"Target '{target}' is ambiguous. Candidates: {names}")
        raise RuntimeError(f"Target '{target}' was not found in namespace '{namespace}'.")

    async def _resolve_status_resource(self, environment: dict[str, Any], namespace: str, target: str) -> str:
        if "/" in target:
            return target
        for resource_type in ("deployment", "statefulset", "daemonset", "pod", "service"):
            if await self._resource_exists(environment, namespace, resource_type, target):
                return f"{resource_type}/{target}"
        return await self._resolve_log_resource(environment, namespace, target)

    async def _resolve_exec_resource(self, environment: dict[str, Any], namespace: str, target: str) -> str:
        if "/" in target:
            return target
        for resource_type in ("deployment", "statefulset", "pod"):
            if await self._resource_exists(environment, namespace, resource_type, target):
                return f"{resource_type}/{target}"
        return await self._resolve_log_resource(environment, namespace, target)

    async def _expand_workload_resources(
        self,
        environment: dict[str, Any],
        namespace: str,
        target: str,
    ) -> list[str]:
        if not target or "/" in target:
            return []
        for resource_type in ("deployment", "statefulset", "daemonset"):
            if await self._resource_exists(environment, namespace, resource_type, target):
                return [f"{resource_type}/{target}"]

        candidates = await self._workload_candidates_from_matches(
            environment,
            namespace,
            await self._raw_target_matches(environment, namespace, target),
        )
        workload_resources = []
        seen = set()
        for candidate in candidates:
            resource = candidate.get("resource", "")
            if resource and resource not in seen:
                seen.add(resource)
                workload_resources.append(resource)

        return workload_resources

    async def _all_workload_resources(self, environment: dict[str, Any], namespace: str) -> list[str]:
        output = await self._kubectl(
            environment,
            "get",
            "deployments,statefulsets,daemonsets",
            "-n",
            namespace,
            "-o",
            "json",
        )
        data = json.loads(output)
        resources = []
        for item in data.get("items", []):
            kind = str(item.get("kind", "")).lower()
            name = item.get("metadata", {}).get("name")
            if kind and name:
                resources.append(f"{kind}/{name}")
        return resources

    async def _pod_names_for_metric_target(
        self,
        environment: dict[str, Any],
        namespace: str,
        target: str,
    ) -> list[str]:
        if "/" in target:
            kind, name = target.split("/", 1)
            if kind.lower() == "pod":
                return [name]
            if kind.lower() in {"deployment", "statefulset", "daemonset"}:
                output = await self._kubectl(environment, "get", target, "-n", namespace, "-o", "json")
                workload = json.loads(output)
                selector = workload.get("spec", {}).get("selector", {}).get("matchLabels", {})
                pods = await self._pods_for_selector(environment, namespace, selector)
                return [pod.get("metadata", {}).get("name", "") for pod in pods if pod.get("metadata", {}).get("name")]
        expanded_resources = await self._expand_workload_resources(environment, namespace, target)
        names = []
        for resource in expanded_resources:
            names.extend(await self._pod_names_for_metric_target(environment, namespace, resource))
        return names

    async def _raw_target_matches(
        self,
        environment: dict[str, Any],
        namespace: str,
        query: str,
    ) -> list[dict[str, Any]]:
        output = await self._kubectl(
            environment,
            "get",
            "pods,deployments,statefulsets,daemonsets,services",
            "-n",
            namespace,
            "-o",
            "json",
        )
        data = json.loads(output)
        matches = []
        query_lower = query.lower().strip()
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            kind = item.get("kind", "")
            name = metadata.get("name", "")
            labels = metadata.get("labels", {})
            searchable = " ".join([kind, name, json.dumps(labels, ensure_ascii=False)]).lower()
            if not query_lower or query_lower in searchable:
                matches.append(
                    {
                        "kind": kind,
                        "name": name,
                        "namespace": metadata.get("namespace", namespace),
                        "labels": labels,
                    }
                )
        return matches

    async def _workload_candidates_from_matches(
        self,
        environment: dict[str, Any],
        namespace: str,
        matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resources = []
        seen = set()
        for match in matches:
            kind = str(match.get("kind", "")).lower()
            name = str(match.get("name", ""))
            if not name:
                continue
            resource = ""
            if kind in {"deployment", "statefulset", "daemonset"}:
                resource = f"{kind}/{name}"
            elif kind in {"pod", "service"}:
                try:
                    resource = await self._resolve_workload_resource(environment, namespace, f"{kind}/{name}")
                except RuntimeError:
                    resource = f"{kind}/{name}"
            if resource and resource not in seen:
                seen.add(resource)
                resources.append(resource)

        candidates = []
        for resource in resources:
            candidates.append(await self._summarize_workload_candidate(environment, namespace, resource))
        return candidates

    async def _summarize_workload_candidate(
        self,
        environment: dict[str, Any],
        namespace: str,
        resource: str,
    ) -> dict[str, Any]:
        kind, name = resource.split("/", 1)
        try:
            output = await self._kubectl(environment, "get", resource, "-n", namespace, "-o", "json")
            data = json.loads(output)
        except RuntimeError:
            return {
                "kind": kind,
                "name": name,
                "resource": resource,
                "namespace": namespace,
                "labels": {},
                "services": [],
                "pods": [],
            }

        selector = data.get("spec", {}).get("selector", {}).get("matchLabels", {})
        pods = await self._pods_for_selector(environment, namespace, selector)
        services = await self._services_for_selector(environment, namespace, selector)
        return {
            "kind": data.get("kind", kind),
            "name": data.get("metadata", {}).get("name", name),
            "resource": resource,
            "namespace": data.get("metadata", {}).get("namespace", namespace),
            "labels": data.get("metadata", {}).get("labels", {}),
            "selector": selector,
            "services": [summarize_service(service) for service in services],
            "pods": [summarize_pod(pod) for pod in pods],
        }

    async def _resolve_workload_resource(self, environment: dict[str, Any], namespace: str, target: str) -> str:
        if "/" in target:
            kind, name = target.split("/", 1)
            if kind.lower() == "pod":
                pod_json = await self._kubectl(environment, "get", "pod", name, "-n", namespace, "-o", "json")
                owner = owner_reference(json.loads(pod_json))
                if owner and owner["kind"] == "ReplicaSet":
                    replicaset_json = await self._kubectl(
                        environment,
                        "get",
                        "replicaset",
                        owner["name"],
                        "-n",
                        namespace,
                        "-o",
                        "json",
                    )
                    replicaset_owner = owner_reference(json.loads(replicaset_json))
                    if replicaset_owner and replicaset_owner["kind"] == "Deployment":
                        return f"deployment/{replicaset_owner['name']}"
                return target
            if kind.lower() == "service":
                service_json = await self._kubectl(environment, "get", "service", name, "-n", namespace, "-o", "json")
                service = json.loads(service_json)
                selector = service.get("spec", {}).get("selector", {})
                pods = await self._pods_for_selector(environment, namespace, selector)
                if pods:
                    first_owner = owner_reference(pods[0])
                    if first_owner and first_owner["kind"] == "ReplicaSet":
                        replicaset_json = await self._kubectl(
                            environment,
                            "get",
                            "replicaset",
                            first_owner["name"],
                            "-n",
                            namespace,
                            "-o",
                            "json",
                        )
                        replicaset_owner = owner_reference(json.loads(replicaset_json))
                        if replicaset_owner and replicaset_owner["kind"] == "Deployment":
                            return f"deployment/{replicaset_owner['name']}"
                    if first_owner and first_owner["kind"] in {"StatefulSet", "DaemonSet"}:
                        return f"{first_owner['kind'].lower()}/{first_owner['name']}"
                return target
            return target
        for resource_type in ("deployment", "statefulset", "daemonset"):
            if await self._resource_exists(environment, namespace, resource_type, target):
                return f"{resource_type}/{target}"
        if await self._resource_exists(environment, namespace, "service", target):
            service_json = await self._kubectl(environment, "get", "service", target, "-n", namespace, "-o", "json")
            service = json.loads(service_json)
            selector = service.get("spec", {}).get("selector", {})
            pods = await self._pods_for_selector(environment, namespace, selector)
            if pods:
                first_owner = owner_reference(pods[0])
                if first_owner and first_owner["kind"] == "ReplicaSet":
                    replicaset_json = await self._kubectl(
                        environment,
                        "get",
                        "replicaset",
                        first_owner["name"],
                        "-n",
                        namespace,
                        "-o",
                        "json",
                    )
                    replicaset_owner = owner_reference(json.loads(replicaset_json))
                    if replicaset_owner and replicaset_owner["kind"] == "Deployment":
                        return f"deployment/{replicaset_owner['name']}"
                if first_owner and first_owner["kind"] in {"StatefulSet", "DaemonSet"}:
                    return f"{first_owner['kind'].lower()}/{first_owner['name']}"
        return await self._resolve_exec_resource(environment, namespace, target)

    async def _pods_for_selector(
        self,
        environment: dict[str, Any],
        namespace: str,
        selector: dict[str, str],
    ) -> list[dict[str, Any]]:
        selector_text = ",".join(f"{key}={value}" for key, value in selector.items())
        if not selector_text:
            return []
        output = await self._kubectl(environment, "get", "pods", "-n", namespace, "-l", selector_text, "-o", "json")
        return json.loads(output).get("items", [])

    async def _services_for_selector(
        self,
        environment: dict[str, Any],
        namespace: str,
        selector: dict[str, str],
    ) -> list[dict[str, Any]]:
        output = await self._kubectl(environment, "get", "services", "-n", namespace, "-o", "json")
        services = json.loads(output).get("items", [])
        result = []
        for service in services:
            service_selector = service.get("spec", {}).get("selector", {})
            if service_selector and all(selector.get(key) == value for key, value in service_selector.items()):
                result.append(service)
        return result

    async def _detect_profile(self, environment: dict[str, Any], namespace: str, resource: str) -> str:
        output = await self._kubectl(environment, "get", resource, "-n", namespace, "-o", "json")
        data = json.loads(output)
        containers = data.get("spec", {}).get("template", {}).get("spec", {}).get("containers")
        if containers is None:
            containers = data.get("spec", {}).get("containers", [])
        searchable = json.dumps(containers, ensure_ascii=False).lower()
        if "redis" in searchable or ":6379" in searchable or '"containerport": 6379' in searchable:
            return "redis"
        if "mysql" in searchable or ":3306" in searchable or '"containerport": 3306' in searchable:
            return "mysql"
        if "mongo" in searchable or ":27017" in searchable or '"containerport": 27017' in searchable:
            return "mongodb"
        if "neo4j" in searchable or ":7687" in searchable or ":7474" in searchable:
            return "neo4j"
        return "unknown"

    async def _resource_exists(self, environment: dict[str, Any], namespace: str, resource_type: str, name: str) -> bool:
        try:
            await self._kubectl(environment, "get", resource_type, name, "-n", namespace, "-o", "name")
        except RuntimeError:
            return False
        return True

    async def _kubectl(self, environment: dict[str, Any], *args: str) -> str:
        command = [self.kubectl_command]
        context = environment.get("context")
        if context:
            command.extend(["--context", context])
        command.extend(args)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.wait()
            raise RuntimeError(f"kubectl command timed out: {' '.join(command)}") from error

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(stderr_text or stdout_text or f"kubectl exited with {process.returncode}")
        return stdout_text

    async def _kubectl_exec(self, environment: dict[str, Any], namespace: str, resource: str, *args: str) -> str:
        return await self._kubectl(environment, "exec", resource, "-n", namespace, "--", *args)


class OpsCapabilityTools:
    def __init__(self, registry: EnvironmentRegistry):
        self.registry = registry

    async def resolve_target(self, query: str, intent: str = "", environment_id: str = "") -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.resolve_target(query, intent=intent, environment=environment)
        return add_environment_header(environment, result)

    async def query_logs(
        self,
        target: str,
        tail: int = 100,
        filter_keywords: str = "",
        environment_id: str = "",
    ) -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.query_logs(
            target=target,
            tail=tail,
            filter_keywords=filter_keywords,
            environment=environment,
        )
        return add_environment_header(environment, result)

    async def query_status(self, target: str = "", fields: str = "summary", environment_id: str = "") -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.query_status(target=target, fields=fields, environment=environment)
        return add_environment_header(environment, result)

    async def query_metrics(
        self,
        target: str = "all",
        metric_types: str = "cpu,memory",
        top_n: int = 8,
        environment_id: str = "",
    ) -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.query_metrics(
            target=target,
            metric_types=metric_types,
            top_n=top_n,
            environment=environment,
        )
        return add_environment_header(environment, result)

    async def query_runtime_diagnostics(
        self,
        target: str,
        profile: str = "auto",
        checks: str = "connections",
        environment_id: str = "",
    ) -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.query_runtime_diagnostics(
            target=target,
            profile=profile,
            checks=checks,
            environment=environment,
        )
        return add_environment_header(environment, result)

    async def query_workload_topology(self, target: str, environment_id: str = "") -> str:
        environment, adapter = self.registry.adapter_for(environment_id)
        result = await adapter.query_workload_topology(target=target, environment=environment)
        return add_environment_header(environment, result)

    async def resolve_container(self, query: str) -> str:
        return await self.resolve_target(query=query)

    async def query_container_logs(self, target: str, tail: int = 100, filter_keywords: str = "") -> str:
        return await self.query_logs(target=target, tail=tail, filter_keywords=filter_keywords)

    async def query_container_info(self, target: str = "", fields: str = "summary") -> str:
        return await self.query_status(target=target, fields=fields)

    async def query_container_metrics(self, target: str = "all", metric_types: str = "cpu,memory", top_n: int = 8) -> str:
        return await self.query_metrics(target=target, metric_types=metric_types, top_n=top_n)


def add_environment_header(environment: dict[str, Any], result: str) -> str:
    return f"Environment: {environment['id']} ({environment['type']})\n\n{result}"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_kubectl_table_by_names(output: str, names: list[str]) -> str:
    wanted = {name for name in names if name}
    if not wanted:
        return output
    lines = output.splitlines()
    if not lines:
        return output
    header = lines[0]
    matched = [line for line in lines[1:] if line.split() and line.split()[0] in wanted]
    return "\n".join([header, *matched]) if matched else f"{header}\nNo metric rows matched target pods: {', '.join(sorted(wanted))}"


def suggest_target_mode(candidate_count: int, intent: str) -> str:
    if candidate_count == 0:
        return "not_found"
    if candidate_count == 1:
        return "direct"
    if looks_like_single_target_intent(intent):
        return "ask_user"
    if looks_like_batch_intent(intent):
        return "batch_expand"
    return "ask_user"


def target_mode_reason(mode: str, candidate_count: int) -> str:
    reasons = {
        "not_found": "No matching workload candidate was found.",
        "direct": "Exactly one workload candidate matched.",
        "batch_expand": "Multiple candidates matched and the user intent appears to request all of them.",
        "ask_user": "Multiple candidates matched and the user did not clearly request all of them.",
    }
    return f"{reasons.get(mode, 'Target mode was inferred.')} candidate_count={candidate_count}"


def looks_like_batch_intent(text: str) -> bool:
    lowered = text.lower()
    keywords = (
        "all",
        "every",
        "each",
        "overall",
        "check",
        "inspect",
        "diagnose",
        "health",
        "status",
        "全部",
        "所有",
        "整体",
        "各个",
        "每个",
        "都",
        "各种",
        "一起",
        "全面",
        "检查",
        "排查",
        "诊断",
        "拓扑",
        "结构",
        "系统",
        "运行情况",
        "健康",
        "状况",
        "情况",
        "异常",
    )
    return any(keyword in lowered for keyword in keywords)


def looks_like_single_target_intent(text: str) -> bool:
    lowered = text.lower()
    keywords = (
        "which",
        "specific",
        "this",
        "that",
        "哪个",
        "哪一个",
        "某个",
        "这个",
        "那个",
        "指定",
        "单个",
        "其中一个",
    )
    return any(keyword in lowered for keyword in keywords)


def owner_reference(resource: dict[str, Any]) -> dict[str, str] | None:
    owners = resource.get("metadata", {}).get("ownerReferences", [])
    if not owners:
        return None
    controller = next((owner for owner in owners if owner.get("controller")), owners[0])
    return {"kind": controller.get("kind", ""), "name": controller.get("name", "")}


def summarize_pod(pod: dict[str, Any]) -> dict[str, Any]:
    metadata = pod.get("metadata", {})
    status = pod.get("status", {})
    spec = pod.get("spec", {})
    container_statuses = status.get("containerStatuses", [])
    ready = bool(container_statuses) and all(item.get("ready") for item in container_statuses)
    restarts = sum(int(item.get("restartCount", 0)) for item in container_statuses)
    return {
        "name": metadata.get("name"),
        "status": status.get("phase"),
        "ready": ready,
        "restart_count": restarts,
        "node": spec.get("nodeName"),
        "pod_ip": status.get("podIP"),
        "owner": owner_reference(pod),
        "labels": metadata.get("labels", {}),
        "containers": [
            {
                "name": container.get("name"),
                "image": container.get("image"),
                "ports": container.get("ports", []),
            }
            for container in spec.get("containers", [])
        ],
    }


def summarize_service(service: dict[str, Any]) -> dict[str, Any]:
    metadata = service.get("metadata", {})
    spec = service.get("spec", {})
    return {
        "name": metadata.get("name"),
        "type": spec.get("type"),
        "cluster_ip": spec.get("clusterIP"),
        "selector": spec.get("selector", {}),
        "ports": spec.get("ports", []),
    }


def summarize_container_inspect(result: Any, fields: str) -> str:
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return result

    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)

    state = data.get("State", {})
    config = data.get("Config", {})
    network = data.get("NetworkSettings", {})
    host_config = data.get("HostConfig", {})
    summary = {
        "id": str(data.get("Id", ""))[:12],
        "name": str(data.get("Name", "")).lstrip("/"),
        "image": config.get("Image"),
        "status": state.get("Status"),
        "running": state.get("Running"),
        "restart_count": data.get("RestartCount"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "exit_code": state.get("ExitCode"),
        "health": state.get("Health"),
        "ports": network.get("Ports"),
        "restart_policy": host_config.get("RestartPolicy"),
    }

    if fields and fields != "summary":
        requested = {field.strip() for field in fields.split(",") if field.strip()}
        summary = {key: value for key, value in summary.items() if key in requested}

    return json.dumps(summary, ensure_ascii=False, indent=2)


def create_capability_tools(config: dict[str, Any], docker_tools: list[Any]) -> dict[str, list[Any]]:
    capabilities = OpsCapabilityTools(EnvironmentRegistry(config, docker_tools))
    tool_configs = config["capability_tools"]

    generic_tools = [
        StructuredTool.from_function(
            coroutine=capabilities.resolve_target,
            name=tool_configs["resolve_target"]["name"],
            description=tool_configs["resolve_target"]["description"],
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_logs,
            name=tool_configs["query_logs"]["name"],
            description=tool_configs["query_logs"]["description"],
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_status,
            name=tool_configs["query_status"]["name"],
            description=tool_configs["query_status"]["description"],
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_metrics,
            name=tool_configs["query_metrics"]["name"],
            description=tool_configs["query_metrics"]["description"],
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_runtime_diagnostics,
            name=tool_configs["query_runtime_diagnostics"]["name"],
            description=tool_configs["query_runtime_diagnostics"]["description"],
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_workload_topology,
            name=tool_configs["query_workload_topology"]["name"],
            description=tool_configs["query_workload_topology"]["description"],
        ),
    ]
    legacy_tools = [
        StructuredTool.from_function(
            coroutine=capabilities.resolve_container,
            name="resolve_container",
            description="兼容旧名称：根据容器简称或 ID 搜索候选容器。",
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_container_logs,
            name="query_container_logs",
            description="兼容旧名称：查询指定容器最近日志。",
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_container_info,
            name="query_container_info",
            description="兼容旧名称：查询指定容器状态和元信息。",
        ),
        StructuredTool.from_function(
            coroutine=capabilities.query_container_metrics,
            name="query_container_metrics",
            description="兼容旧名称：查询容器指标摘要。",
        ),
    ]

    return {
        "generic": generic_tools,
        "metrics": [tool for tool in generic_tools if tool.name == "query_metrics"],
        "docker": [
            tool
            for tool in generic_tools
            if tool.name
            in {
                "resolve_target",
                "query_logs",
                "query_status",
                "query_runtime_diagnostics",
                "query_workload_topology",
            }
        ],
        "legacy": legacy_tools,
        "all": [*generic_tools, *legacy_tools],
    }
