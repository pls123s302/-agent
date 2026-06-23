# Kubernetes 环境

当前项目已经增加 `k8s-current` 环境，使用本机 `kubectl` 当前 kubeconfig/context 做只读查询。这个环境可以指向本机 Docker Desktop Kubernetes，也可以指向远程 Kubernetes 集群。

它复用同一组高层能力工具：

```text
resolve_target -> kubectl get pods,deployments,statefulsets,daemonsets,services
query_logs    -> kubectl logs deployment/pod/statefulset/daemonset
query_status  -> kubectl get / describe
query_metrics -> kubectl top pod
```

运行示例：

```powershell
python langgraph_qwen_demo.py --environment-id k8s-current --question "redis 最近 100 条日志都说了什么？"
python langgraph_qwen_demo.py --environment-id k8s-current --question "default 命名空间现在有哪些工作负载？"
python langgraph_qwen_demo.py --environment-id k8s-current --question "redis 现在健康吗？"
```

如果 `query_metrics` 返回 metrics-server 不可用，说明当前本地 Kubernetes 没装 metrics-server；这不影响日志、状态、Service、Pod 查询。
