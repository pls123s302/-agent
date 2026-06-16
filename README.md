# LangGraph 运维 Agent Demo

当前版本是一个多 Agent 运维助手骨架：

```text
用户问题
    ↓
router_agent 判断任务类型
    ├─ chat_agent    普通问答
    ├─ log_agent     分析用户直接提供的日志
    ├─ metrics_agent 查询 cAdvisor CPU / 内存指标
    └─ docker_agent  查询 Docker 容器信息
            ↓
        docker_tools
            ↓
        reflect_agent 判断继续查、追问用户或总结
            ↓
        final_agent
```

## 目录结构

```text
config/
  app_config.json       模型、Agent 配置、提示词、cAdvisor、工具描述
  mcp_servers.json      Docker 只读 MCP server 配置
ops_agent/
  cadvisor.py           cAdvisor 指标采集和摘要生成
  cli.py                命令行入口
  config.py             配置加载和占位符替换
  graph.py              多 Agent LangGraph 流程
  llm.py                LLM 创建
  mcp_servers/
    docker_readonly.py  Docker 只读 MCP server
  nodes.py              router/chat/log/metrics/docker/reflect/final 节点
  state.py              LangGraph state
  tools.py              工具分组和 MCP tools 加载
langgraph_qwen_demo.py  兼容入口
```

## 工具

本地工具：

```text
query_cadvisor_metrics
```

Docker 只读 MCP 工具：

```text
list_containers
search_containers
get_container_logs
inspect_container
get_container_stats
```

Docker 工具只做查询，不提供 stop、restart、rm、exec、prune 等修改能力。

## 运行

交互式输入问题：

```powershell
python langgraph_qwen_demo.py
```

交互模式会保留当前进程内的短期上下文。退出命令：

```text
exit
quit
q
退出
```

直接传问题：

```powershell
python langgraph_qwen_demo.py --question "当前本机 CPU 和内存是否异常？"
python langgraph_qwen_demo.py --question "看看 neo4j 容器最近 100 条日志"
python langgraph_qwen_demo.py --question "分析日志：ERROR database query timeout"
```

只输入一轮后退出：

```powershell
python langgraph_qwen_demo.py --once
```

## 环境

建议在你的 `Langgraph` conda 环境里运行：

```powershell
conda activate Langgraph
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证依赖：

```powershell
python -c "from mcp.server.fastmcp import FastMCP; print('mcp ok')"
python -c "from langchain_mcp_adapters.client import MultiServerMCPClient; print('adapter ok')"
docker ps
```

如果只是普通聊天或直接分析用户粘贴的日志，Docker MCP 不可用时也不会影响这些分支。
