"""LangGraph 多 Agent demo：supervisor 调度 researcher/analyst/writer。

用法:
  python app.py "分析低空经济赛道 2026 年的投资机会"
离线运行使用内置 mock LLM；设置 LLM_API_KEY 等环境变量后自动切换真实模型（见 README）。
"""
import sys

from langchain_core.messages import HumanMessage

from graph import SPECIALISTS, build_graph

if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    task = " ".join(sys.argv[1:]) or "分析一下远程办公对城市房价的影响"
    graph, llm_label = build_graph()

    print(f"任务: {task}\n模型: {llm_label}\n" + "-" * 60)
    answer = None
    for update in graph.stream({"messages": [HumanMessage(content=task)]}, stream_mode="updates"):
        for node, delta in update.items():
            for msg in delta.get("messages", []):
                print(f"({node:>10}) {msg.content}")
            if node in SPECIALISTS:
                answer = delta["messages"][-1].content
    print("-" * 60)
    print("最终答案:", answer or "（图未产生专家输出，检查路由/模型配置）")


if __name__ == "__main__":
    main()
