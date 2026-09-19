"""LangGraph 多 Agent demo — Streamlit 前端。

离线默认走 mock LLM；在 Streamlit Cloud 的 Secrets 里配置：
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 即可切换真实模型。
"""
import os

import streamlit as st
from langchain_core.messages import HumanMessage

from graph import SPECIALISTS, build_graph

st.set_page_config(page_title="LangGraph 多 Agent 协作", page_icon="🕸", layout="centered")

try:
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:  # 本地无 secrets 文件时 st.secrets 会抛异常
    pass

st.title("🕸 LangGraph 多 Agent 协作")
st.caption("Supervisor 模式：supervisor 逐步决策，调度 researcher → analyst → writer 协作完成任务")

task = st.chat_input("给团队一个任务，例如：分析低空经济赛道2026年的投资机会")
if task:
    st.chat_message("user").markdown(f"**{task}**")

    graph, llm_label = build_graph()
    with st.chat_message("assistant"):
        st.sidebar.success(f"模型: {llm_label}")
        trace = st.container()
        with st.status("团队协作中…", expanded=True) as status:
            answer = None
            for update in graph.stream(
                {"messages": [HumanMessage(content=task)]}, stream_mode="updates"
            ):
                for node, delta in update.items():
                    for msg in delta.get("messages", []):
                        icon = "🧭" if node == "supervisor" else "🛠"
                        trace.markdown(f"{icon} **{node}** — {msg.content}")
                    if node in SPECIALISTS:
                        answer = delta["messages"][-1].content
            status.update(label="协作完成", state="complete")
        st.markdown("### 最终答案")
        st.markdown(answer or "（无输出）")

with st.sidebar:
    st.markdown("#### 关于")
    st.markdown(
        "- 图结构：`StateGraph`，supervisor 输出 JSON 路由\n"
        "- 未配置 `LLM_API_KEY` 时使用内置离线 mock，流程一致\n"
        "- 源码见 [GitHub](https://github.com/abaic2/langgraph-demo)"
    )
