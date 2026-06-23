# LangGraph 运维 Agent Demo

这是一个本机运维 Agent 骨架，使用 LangGraph 编排多个节点，使用本机 Ollama `qwen3:8b` 做推理。

当前架构重点是：Agent 面向“运维能力”，不是直接面向 Docker、K8S 或某台机器。

```text
用户问题
  -> router_agent    判断任务类型
  -> plan_agent      生成初始调查计划
  -> execute_step    按当前 step 调用高层能力工具
  -> reflect_agent   判断继续、重规划、追问用户或总结
  -> replan_agent    根据 observations 追加调查步骤
  -> final_agent     输出结论、证据、建议
```

核心流程：

```text
plan -> execute -> observe -> reflect -> replan -> execute -> final
```

## 环境抽象

配置里已经引入 `environments`：

```json
{
  "id": "local-docker",
  "type": "docker",
  "default": true,
  "capabilities": ["target", "logs", "status", "metrics"]
}
```

Agent 计划的是通用能力：

```text
resolve_target
query_logs
query_status
query_metrics
```

当前 `local-docker` 环境由 Docker adapter 实现：

```text
resolve_target -> Docker MCP search_containers
query_logs    -> Docker MCP get_container_logs
query_status  -> Docker MCP inspect/list
query_metrics -> cAdvisor /metrics
```

以后接 K8S 时，不需要重写 Agent 主流程，只需要新增一个 Kubernetes adapter，把同样的能力映射到 K8S API、Prometheus、Loki 等数据源。

## Tool Gateway 是什么

Tool Gateway 可以理解为 Agent 和真实运维环境之间的一层“工具网关”。

```text
Agent
  -> Tool Gateway
      -> Docker adapter
      -> Kubernetes adapter
      -> Prometheus adapter
      -> Loki adapter
      -> CMDB adapter
```

它负责：

```text
统一鉴权
环境路由
权限控制
只读/写操作白名单
审计日志
限流
屏蔽底层差异
```

如果 Agent 和 Docker 不在一台机器上，推荐不是让 Agent 直接远程执行 Docker 命令，而是：

```text
Agent -> Tool Gateway / Remote MCP Server -> 目标环境
```

这样目标网络只暴露受控能力，例如查日志、查状态、查指标，而不是暴露完整主机权限。

## 当前高层工具

```text
resolve_target(query, environment_id)
query_logs(target, tail, filter_keywords, environment_id)
query_status(target, fields, environment_id)
query_metrics(target, metric_types, top_n, environment_id)
```

为了兼容旧代码，也暂时保留了旧名称：

```text
resolve_container
query_container_logs
query_container_info
query_container_metrics
```

## 指标来源

当前 Docker 环境的指标来自 cAdvisor，支持：

```text
cpu
memory
network
disk
```

网络吞吐、磁盘读写吞吐、包速率、错误速率是通过两次采样计算出来的瞬时速率。采样间隔配置在 `config/app_config.json` 的 `cadvisor.cpu_sample_interval_seconds`。

## 目录结构

```text
config/
  app_config.json       模型、Agent、环境、提示词、高层工具、cAdvisor 配置
  mcp_servers.json      Docker 只读 MCP server 配置

ops_agent/
  capability_tools.py   环境注册、通用能力工具、Docker adapter
  cadvisor.py           cAdvisor 指标 adapter
  cli.py                命令行入口和短期记忆会话
  config.py             配置加载和占位符替换
  graph.py              LangGraph 循环式调查流程
  llm.py                LLM 创建
  nodes.py              router/plan/execute/reflect/replan/final 节点
  state.py              LangGraph state
  tools.py              工具分组和 MCP adapter 加载
  mcp_servers/
    docker_readonly.py  Docker 只读 MCP server

langgraph_qwen_demo.py  兼容入口
```

## 运行

```powershell
conda activate Langgraph
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python langgraph_qwen_demo.py
```

直接传问题：

```powershell
python langgraph_qwen_demo.py --question "帮我看看这个环境有没有异常"
python langgraph_qwen_demo.py --question "redis 最近 300 条日志都说了什么？"
python langgraph_qwen_demo.py --question "redis 和 neo4j 的日志一起看能说明什么？"
```

指定环境：

```powershell
python langgraph_qwen_demo.py --environment-id local-docker --question "看一下当前容器网络和磁盘有没有异常"
```

## Docker MCP

底层 Docker MCP 仍然只读，只提供查询能力：

```text
list_containers
search_containers
get_container_logs
inspect_container
get_container_stats
```

不会提供 `stop`、`restart`、`rm`、`exec`、`prune` 等修改类能力。
