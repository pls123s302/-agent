import asyncio
import re
import urllib.request
from collections import defaultdict
from typing import Any, Callable


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

        parsed[name].append({"labels": labels, "value": value})

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


def format_bytes_per_second(value: float) -> str:
    return f"{format_bytes(value)}/s"


def first_value(metrics: dict[str, list[dict[str, Any]]], metric_name: str, default: float = 0) -> float:
    samples = metrics.get(metric_name, [])
    if not samples:
        return default
    return float(samples[0]["value"])


def collect_gauge_max(metrics: dict[str, list[dict[str, Any]]], metric_name: str) -> dict[str, float]:
    values_by_container: dict[str, float] = {}
    for sample in metrics.get(metric_name, []):
        name = container_name(sample["labels"])
        values_by_container[name] = max(values_by_container.get(name, 0), float(sample["value"]))
    return values_by_container


def collect_counter_sum(metrics: dict[str, list[dict[str, Any]]], metric_name: str) -> dict[str, float]:
    values_by_container: dict[str, float] = defaultdict(float)
    for sample in metrics.get(metric_name, []):
        values_by_container[container_name(sample["labels"])] += float(sample["value"])
    return dict(values_by_container)


def collect_cpu(metrics: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    cpu_by_container: dict[str, float] = defaultdict(float)
    for sample in metrics.get("container_cpu_usage_seconds_total", []):
        labels = sample["labels"]
        cpu_label = labels.get("cpu")
        if cpu_label and cpu_label != "total":
            continue
        cpu_by_container[container_name(labels)] += float(sample["value"])
    return dict(cpu_by_container)


def calculate_counter_rates(
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    metric_name: str,
    interval_seconds: float,
) -> dict[str, float]:
    first_values = collect_counter_sum(first_metrics, metric_name)
    second_values = collect_counter_sum(second_metrics, metric_name)
    rates: dict[str, float] = {}

    for name, second_value in second_values.items():
        first_value_for_name = first_values.get(name)
        if first_value_for_name is None:
            continue
        rates[name] = max(0, second_value - first_value_for_name) / interval_seconds

    return rates


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
        rates[name] = max(0, second_value - first_value_for_name) / interval_seconds

    return rates


def merge_rates(*rate_maps: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for rate_map in rate_maps:
        for name, value in rate_map.items():
            merged[name] += value
    return dict(merged)


def append_top(
    lines: list[str],
    title: str,
    values: dict[str, float],
    formatter: Callable[[float], str],
    top_n: int,
) -> None:
    lines.extend(["", title])
    top_values = sorted(values.items(), key=lambda item: item[1], reverse=True)[:top_n]
    if not top_values:
        lines.append("- no samples found")
        return

    for name, value in top_values:
        lines.append(f"- {name}: {formatter(value)}")


def build_metrics_summary(
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
    top_n: int,
    metric_types: set[str],
) -> str:
    cpu_cores = first_value(second_metrics, "machine_cpu_cores")
    machine_memory = first_value(second_metrics, "machine_memory_bytes")

    lines = [
        "cAdvisor metrics summary:",
        f"- CPU cores: {cpu_cores:.0f}" if cpu_cores else "- CPU cores: unknown",
        f"- Machine memory: {format_bytes(machine_memory)}" if machine_memory else "- Machine memory: unknown",
        f"- Sample interval for rate metrics: {interval_seconds:.1f}s",
    ]

    if "cpu" in metric_types:
        append_cpu_summary(lines, first_metrics, second_metrics, interval_seconds, top_n, cpu_cores)

    if "memory" in metric_types:
        append_memory_summary(lines, second_metrics, top_n, machine_memory)

    if "network" in metric_types:
        append_network_summary(lines, first_metrics, second_metrics, interval_seconds, top_n)

    if "disk" in metric_types:
        append_disk_summary(lines, first_metrics, second_metrics, interval_seconds, top_n)

    return "\n".join(lines)


def append_cpu_summary(
    lines: list[str],
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
    top_n: int,
    cpu_cores: float,
) -> None:
    cpu_rates = calculate_cpu_rates(first_metrics, second_metrics, interval_seconds)
    top_cpu = sorted(cpu_rates.items(), key=lambda item: item[1], reverse=True)[:top_n]

    lines.extend(["", "Top containers by CPU cores used:"])
    if not top_cpu:
        lines.append("- no CPU samples found")
        return

    for name, cpu_rate in top_cpu:
        percent = (cpu_rate / cpu_cores * 100) if cpu_cores else 0
        lines.append(f"- {name}: {cpu_rate:.4f} cores, {percent:.2f}% of host CPU")


def append_memory_summary(
    lines: list[str],
    second_metrics: dict[str, list[dict[str, Any]]],
    top_n: int,
    machine_memory: float,
) -> None:
    memory_metric_name = "container_memory_working_set_bytes"
    memory_by_container = collect_gauge_max(second_metrics, memory_metric_name)
    if not memory_by_container:
        memory_metric_name = "container_memory_usage_bytes"
        memory_by_container = collect_gauge_max(second_metrics, memory_metric_name)

    top_memory = sorted(memory_by_container.items(), key=lambda item: item[1], reverse=True)[:top_n]
    lines.extend(["", f"Top containers by memory ({memory_metric_name}):"])
    if not top_memory:
        lines.append("- no memory samples found")
        return

    for name, memory in top_memory:
        percent = (memory / machine_memory * 100) if machine_memory else 0
        lines.append(f"- {name}: {format_bytes(memory)}, {percent:.2f}% of host memory")


def append_network_summary(
    lines: list[str],
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
    top_n: int,
) -> None:
    rx_bytes = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_receive_bytes_total", interval_seconds
    )
    tx_bytes = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_transmit_bytes_total", interval_seconds
    )
    rx_packets = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_receive_packets_total", interval_seconds
    )
    tx_packets = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_transmit_packets_total", interval_seconds
    )
    rx_errors = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_receive_errors_total", interval_seconds
    )
    tx_errors = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_transmit_errors_total", interval_seconds
    )
    rx_drops = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_receive_packets_dropped_total", interval_seconds
    )
    tx_drops = calculate_counter_rates(
        first_metrics, second_metrics, "container_network_transmit_packets_dropped_total", interval_seconds
    )

    append_top(lines, "Top containers by network receive:", rx_bytes, format_bytes_per_second, top_n)
    append_top(lines, "Top containers by network transmit:", tx_bytes, format_bytes_per_second, top_n)
    append_top(lines, "Top containers by network receive packets:", rx_packets, lambda value: f"{value:.2f} packets/s", top_n)
    append_top(lines, "Top containers by network transmit packets:", tx_packets, lambda value: f"{value:.2f} packets/s", top_n)
    append_top(lines, "Top containers by network errors:", merge_rates(rx_errors, tx_errors), lambda value: f"{value:.4f}/s", top_n)
    append_top(lines, "Top containers by network drops:", merge_rates(rx_drops, tx_drops), lambda value: f"{value:.4f}/s", top_n)


def append_disk_summary(
    lines: list[str],
    first_metrics: dict[str, list[dict[str, Any]]],
    second_metrics: dict[str, list[dict[str, Any]]],
    interval_seconds: float,
    top_n: int,
) -> None:
    fs_usage = collect_gauge_max(second_metrics, "container_fs_usage_bytes")
    fs_limit = collect_gauge_max(second_metrics, "container_fs_limit_bytes")
    reads_bytes = calculate_counter_rates(first_metrics, second_metrics, "container_fs_reads_bytes_total", interval_seconds)
    writes_bytes = calculate_counter_rates(first_metrics, second_metrics, "container_fs_writes_bytes_total", interval_seconds)
    reads_ops = calculate_counter_rates(first_metrics, second_metrics, "container_fs_reads_total", interval_seconds)
    writes_ops = calculate_counter_rates(first_metrics, second_metrics, "container_fs_writes_total", interval_seconds)

    lines.extend(["", "Top containers by filesystem usage:"])
    top_fs_usage = sorted(fs_usage.items(), key=lambda item: item[1], reverse=True)[:top_n]
    if top_fs_usage:
        for name, usage in top_fs_usage:
            limit = fs_limit.get(name, 0)
            percent = (usage / limit * 100) if limit else 0
            if limit:
                lines.append(f"- {name}: {format_bytes(usage)} / {format_bytes(limit)}, {percent:.2f}%")
            else:
                lines.append(f"- {name}: {format_bytes(usage)}, limit unknown")
    else:
        lines.append("- no filesystem usage samples found")

    append_top(lines, "Top containers by disk read throughput:", reads_bytes, format_bytes_per_second, top_n)
    append_top(lines, "Top containers by disk write throughput:", writes_bytes, format_bytes_per_second, top_n)
    append_top(lines, "Top containers by disk read ops:", reads_ops, lambda value: f"{value:.2f} ops/s", top_n)
    append_top(lines, "Top containers by disk write ops:", writes_ops, lambda value: f"{value:.2f} ops/s", top_n)


def normalize_metric_types(metric_types: set[str] | None) -> set[str]:
    supported = {"cpu", "memory", "network", "disk"}
    if not metric_types:
        return {"cpu", "memory"}
    normalized = {metric_type for metric_type in metric_types if metric_type in supported}
    return normalized or {"cpu", "memory"}


async def collect_cadvisor_summary(cadvisor_config: dict[str, Any], metric_types: set[str] | None = None) -> str:
    base_url = cadvisor_config["base_url"].rstrip("/")
    metrics_url = f"{base_url}{cadvisor_config['metrics_path']}"
    timeout_seconds = float(cadvisor_config["timeout_seconds"])
    interval_seconds = float(cadvisor_config["cpu_sample_interval_seconds"])
    top_n = int(cadvisor_config["top_n"])
    requested_metric_types = normalize_metric_types(metric_types)

    metric_names = {
        "machine_cpu_cores",
        "machine_memory_bytes",
        "container_cpu_usage_seconds_total",
        "container_memory_usage_bytes",
        "container_memory_working_set_bytes",
        "container_network_receive_bytes_total",
        "container_network_transmit_bytes_total",
        "container_network_receive_packets_total",
        "container_network_transmit_packets_total",
        "container_network_receive_errors_total",
        "container_network_transmit_errors_total",
        "container_network_receive_packets_dropped_total",
        "container_network_transmit_packets_dropped_total",
        "container_fs_usage_bytes",
        "container_fs_limit_bytes",
        "container_fs_reads_bytes_total",
        "container_fs_writes_bytes_total",
        "container_fs_reads_total",
        "container_fs_writes_total",
    }

    first_text = await asyncio.to_thread(fetch_metrics, metrics_url, timeout_seconds)
    await asyncio.sleep(interval_seconds)
    second_text = await asyncio.to_thread(fetch_metrics, metrics_url, timeout_seconds)

    first_metrics = parse_prometheus_metrics(first_text, metric_names)
    second_metrics = parse_prometheus_metrics(second_text, metric_names)
    return build_metrics_summary(first_metrics, second_metrics, interval_seconds, top_n, requested_metric_types)
