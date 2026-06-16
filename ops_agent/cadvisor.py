import asyncio
import re
import urllib.request
from collections import defaultdict
from typing import Any


LABEL_PATTERN = re.compile(r'(\w+)="([^"]*)"')


def fetch_metrics(metrics_url: str, timeout_seconds: float) -> str:
    with urllib.request.urlopen(metrics_url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_labels(raw_labels: str) -> dict[str, str]:
    return dict(LABEL_PATTERN.findall(raw_labels))


def parse_prometheus_metrics(metrics_text: str, metric_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    parsed: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        metric_part, _, value_part = line.partition(" ")
        if not value_part:
            continue

        name, labels = parse_metric_name_and_labels(metric_part)
        if name not in metric_names:
            continue

        try:
            value = float(value_part.split()[0])
        except ValueError:
            continue

        parsed[name].append(
            {
                "labels": labels,
                "value": value,
            }
        )

    return dict(parsed)


def parse_metric_name_and_labels(metric_part: str) -> tuple[str, dict[str, str]]:
    if "{" not in metric_part:
        return metric_part, {}

    name, raw_labels = metric_part.split("{", 1)
    return name, parse_labels(raw_labels.rstrip("}"))


def container_name(labels: dict[str, str]) -> str:
    for key in ("name", "container_label_com_docker_compose_service", "container_label_io_kubernetes_container_name"):
        value = labels.get(key)
        if value:
            return value.lstrip("/")

    container_id = labels.get("id", "")
    if container_id == "/":
        return "host"
    if container_id:
        return container_id.split("/")[-1][:12]
    return "unknown"


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def first_value(metrics: dict[str, list[dict[str, Any]]], metric_name: str, default: float = 0) -> float:
    samples = metrics.get(metric_name, [])
    if not samples:
        return default
    return float(samples[0]["value"])


def collect_memory(metrics: dict[str, list[dict[str, Any]]], metric_name: str) -> dict[str, float]:
    memory_by_container: dict[str, float] = {}
    for sample in metrics.get(metric_name, []):
        labels = sample["labels"]
        name = container_name(labels)
        memory_by_container[name] = max(memory_by_container.get(name, 0), float(sample["value"]))
    return memory_by_container


def collect_cpu(metrics: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    cpu_by_container: dict[str, float] = defaultdict(float)
    for sample in metrics.get("container_cpu_usage_seconds_total", []):
        labels = sample["labels"]
        cpu_label = labels.get("cpu")
        if cpu_label and cpu_label != "total":
            continue
        cpu_by_container[container_name(labels)] += float(sample["value"])
    return dict(cpu_by_container)


def calculate_cpu_rates(
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
) -> dict[str, float]:
    first_cpu = collect_cpu(first_metrics)
    second_cpu = collect_cpu(second_metrics)
    rates: dict[str, float] = {}

    for name, second_value in second_cpu.items():
        first_value_for_name = first_cpu.get(name)
        if first_value_for_name is None:
            continue
        delta = max(0, second_value - first_value_for_name)
        rates[name] = delta / interval_seconds

    return rates


def build_metrics_summary(
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
    top_n: int,
) -> str:
    cpu_cores = first_value(second_metrics, "machine_cpu_cores")
    machine_memory = first_value(second_metrics, "machine_memory_bytes")
    cpu_rates = calculate_cpu_rates(first_metrics, second_metrics, interval_seconds)

    memory_metric_name = "container_memory_working_set_bytes"
    memory_by_container = collect_memory(second_metrics, memory_metric_name)
    if not memory_by_container:
        memory_metric_name = "container_memory_usage_bytes"
        memory_by_container = collect_memory(second_metrics, memory_metric_name)

    top_cpu = sorted(cpu_rates.items(), key=lambda item: item[1], reverse=True)[:top_n]
    top_memory = sorted(memory_by_container.items(), key=lambda item: item[1], reverse=True)[:top_n]

    lines = [
        "cAdvisor metrics summary:",
        f"- CPU cores: {cpu_cores:.0f}" if cpu_cores else "- CPU cores: unknown",
        f"- Machine memory: {format_bytes(machine_memory)}" if machine_memory else "- Machine memory: unknown",
        f"- CPU sample interval: {interval_seconds:.1f}s",
        f"- Memory metric: {memory_metric_name}",
        "",
        "Top containers by CPU cores used:",
    ]

    if top_cpu:
        for name, cpu_rate in top_cpu:
            percent = (cpu_rate / cpu_cores * 100) if cpu_cores else 0
            lines.append(f"- {name}: {cpu_rate:.4f} cores, {percent:.2f}% of host CPU")
    else:
        lines.append("- no CPU samples found")

    lines.extend(["", "Top containers by memory:"])
    if top_memory:
        for name, memory in top_memory:
            percent = (memory / machine_memory * 100) if machine_memory else 0
            lines.append(f"- {name}: {format_bytes(memory)}, {percent:.2f}% of host memory")
    else:
        lines.append("- no memory samples found")

    return "\n".join(lines)


async def collect_cadvisor_summary(cadvisor_config: dict[str, Any]) -> str:
    base_url = cadvisor_config["base_url"].rstrip("/")
    metrics_url = f"{base_url}{cadvisor_config['metrics_path']}"
    timeout_seconds = float(cadvisor_config["timeout_seconds"])
    interval_seconds = float(cadvisor_config["cpu_sample_interval_seconds"])
    top_n = int(cadvisor_config["top_n"])

    metric_names = {
        "machine_cpu_cores",
        "machine_memory_bytes",
        "container_cpu_usage_seconds_total",
        "container_memory_usage_bytes",
        "container_memory_working_set_bytes",
    }

    first_text = await asyncio.to_thread(fetch_metrics, metrics_url, timeout_seconds)
    await asyncio.sleep(interval_seconds)
    second_text = await asyncio.to_thread(fetch_metrics, metrics_url, timeout_seconds)

    first_metrics = parse_prometheus_metrics(first_text, metric_names)
    second_metrics = parse_prometheus_metrics(second_text, metric_names)
    return build_metrics_summary(first_metrics, second_metrics, interval_seconds, top_n)
