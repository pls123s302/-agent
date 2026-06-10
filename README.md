# LangGraph + Qwen3 + Filesystem MCP Demo

这个 demo 使用 LangGraph 编排两个步骤：

```text
Filesystem MCP 读取日志文件
    ↓
qwen3:8b 分析日志内容
```

## 准备

确认本机已经安装并启动 Ollama：

```powershell
ollama pull qwen3:8b
ollama serve
```

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

Filesystem MCP 通过 `npx` 启动，所以还需要本机安装 Node.js。

## 运行

```powershell
python langgraph_qwen_demo.py
```

默认会读取当前项目里的 `sample.log`，并使用本机 `qwen3:8b` 分析。

## MCP 配置

MCP server 配置不写死在代码里，而是在 `mcp_config.json`：

```json
{
  "filesystem": {
    "command": "cmd",
    "args": [
      "/c",
      "npx",
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "{mcp_root}"
    ],
    "transport": "stdio"
  }
}
```

`{mcp_root}` 是占位符，运行时会替换成 `--mcp-root` 的值。

## 自定义参数

```powershell
python langgraph_qwen_demo.py --file .\sample.log --question "分析这个日志" --model qwen3:8b --mcp-root .
```

如果要读取其他目录里的文件，`--mcp-root` 必须包含那个文件所在目录。例如：

```powershell
python langgraph_qwen_demo.py --file C:\logs\app.log --mcp-root C:\logs --question "分析这个日志"
```

也可以指定另一个 MCP 配置文件：

```powershell
python langgraph_qwen_demo.py --mcp-config .\mcp_config.json
```
