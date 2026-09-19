# LangGraph 多 Agent Demo（Supervisor 模式）

一个最小但完整的多 Agent 协作示例：**supervisor 负责调度，三个专家 Agent 依次干活**，全部跑在 LangGraph 的 `StateGraph` 上。默认使用内置的规则化 mock LLM，**无需任何 API key 即可离线运行**；配置环境变量后无缝切换真实大模型。

## 架构

```
            ┌──────────────<──────────────┐
START ────> supervisor ────(JSON next)───> researcher
            │                              analyst
            │                              writer
            └──(next = FINISH)──> END       （专家节点执行完都回到 supervisor）
```

- `graph.py` — 图定义：共享状态 `TeamState`（`messages` 列表，`add` reducer 累积），
  supervisor 节点让 LLM 输出 `{"next": "..."}`，条件边据此路由到专家或结束。
- `llms.py` — LLM 工厂 + `MockChatModel`（继承 `BaseChatModel`，按提示词约定做确定性应答）。
- `app.py` — 命令行入口，`graph.stream(..., stream_mode="updates")` 逐节点打印轨迹。

## 运行

```powershell
# venv 已由 uv 创建好（Python 3.12），激活：
.venv\Scripts\activate

# 离线 mock 模式（默认）
python app.py "分析低空经济赛道2026年的投资机会"
```

输出示例：

```
(supervisor) {"next": "researcher"}
(researcher) [researcher] ...
(supervisor) {"next": "analyst"}
   ...
最终答案: [writer] ...
```

### 接真实 LLM（OpenAI 兼容接口）

```powershell
$env:LLM_API_KEY  = "sk-..."
$env:LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 例：阿里通义
$env:LLM_MODEL    = "qwen-plus"
python app.py "..."
```

`LLM_*` 与 `OPENAI_*` 两组变量名均可识别；DeepSeek、Moonshot、vLLM 等兼容 OpenAI 协议的服务同理。

## 重建环境（本机踩坑记录）

- 系统默认 Python 是 3.8，msys64 是 3.14，都装不上 langgraph（依赖 `uuid-utils` 缺对应 wheel）。
  本项目的 `.venv` 用 **uv 下载的独立 CPython 3.12** 创建。
- 本机系统代理是坏的，pip/uv 需带 `NO_PROXY='*'`；PyPI 走清华镜像。
- Python 3.12 由 uv 托管，GitHub 直连超时，需设
  `UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone`。

```powershell
py -m pip install --user uv
py -m uv python install 3.12          # 配合上面的镜像变量
py -m uv venv .venv --python 3.12
py -m uv pip install -p .venv -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 可以继续演化的方向

- 给 researcher 挂 `tool`（搜索/数据库），升级为 ReAct 节点（`langgraph.prebuilt.create_react_agent`）
- supervisor 改用 structured output / `Command` 做 handoff
- 加 checkpointer（`MemorySaver`）实现多轮对话与断点恢复
- 并行分支：researcher 与 analyst 同时执行再汇总
