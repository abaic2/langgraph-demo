"""Supervisor 模式多 Agent 图：supervisor 路由，三个专家 Agent 依次干活。

        ┌──────────────<──────────────┐
START -> supervisor --(JSON next)--> researcher/analyst/writer
        └──(FINISH)--> END
"""
import json
from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from llms import build_llm

SPECIALISTS = ("researcher", "analyst", "writer")

SYSTEM_PROMPTS = {
    "supervisor": (
        "You are the supervisor of a small agent team. Decide which specialist works "
        "next. Specialists already done are shown in the transcript as [name]. "
        "Reply with ONLY a JSON object: "
        '{"next": "researcher" | "analyst" | "writer" | "FINISH"}'
    ),
    "researcher": (
        "You are the researcher on the team. Collect the key facts for the task."
    ),
    "analyst": (
        "You are the analyst. Interpret the research findings and give conclusions."
    ),
    "writer": (
        "You are the writer. Produce the final polished answer from all prior work."
    ),
}


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]


def _transcript(state: TeamState) -> str:
    lines = []
    for m in state["messages"]:
        tag = m.name or type(m).__name__
        lines.append(f"{tag}: {m.content}")
    return "\n".join(lines)


def _make_supervisor(llm):
    def supervisor(state: TeamState) -> dict:
        prompt = [
            ("system", SYSTEM_PROMPTS["supervisor"]),
            *[("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in state["messages"]],
        ]
        reply = llm.invoke(prompt)
        return {"messages": [AIMessage(name="supervisor", content=reply.content)]}

    return supervisor


def _route(state: TeamState) -> Literal["researcher", "analyst", "writer", "__end__"]:
    for m in reversed(state["messages"]):
        if m.name == "supervisor":
            try:
                nxt = json.loads(m.content)["next"]
            except (json.JSONDecodeError, KeyError):
                return "__end__"
            return nxt if nxt in SPECIALISTS else "__end__"
    return "__end__"


def _make_specialist(llm, name: str):
    def specialist(state: TeamState) -> dict:
        task = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
        prompt = [
            ("system", SYSTEM_PROMPTS[name]),
            ("system", f"Transcript so far:\n{_transcript(state)}"),
            ("human", f"as the {name}, handle this task: {task}"),
        ]
        reply = llm.invoke(prompt)
        return {"messages": [AIMessage(name=name, content=f"[{name}] {reply.content}")]}

    return specialist


def build_graph():
    llm, llm_label = build_llm()
    g = StateGraph(TeamState)
    g.add_node("supervisor", _make_supervisor(llm))
    for name in SPECIALISTS:
        g.add_node(name, _make_specialist(llm, name))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route,
        {**{s: s for s in SPECIALISTS}, "__end__": END},
    )
    for name in SPECIALISTS:
        g.add_edge(name, "supervisor")
    return g.compile(), llm_label
