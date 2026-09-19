"""LLM factory: real OpenAI-compatible model when configured, deterministic mock otherwise.

Set env vars to use a real model:
  LLM_API_KEY    (or OPENAI_API_KEY)
  LLM_BASE_URL   (or OPENAI_BASE_URL) e.g. https://dashscope.aliyuncs.com/compatible-mode/v1
  LLM_MODEL      (or OPENAI_MODEL)    e.g. qwen-plus / deepseek-chat / gpt-4o-mini
"""
import os
from typing import ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _env(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


class MockChatModel(BaseChatModel):
    """Offline stand-in for an LLM.

    It obeys two prompt conventions used by the demo so the graph logic
    (routing, state passing, message flow) is identical to real-model mode:
      1. Supervisor prompt asks for JSON {"next": "..."}; we route by which
         specialist outputs (tagged "[researcher]" etc.) already appear in the transcript.
      2. Specialist prompts contain the task text; we echo a canned section.
    """

    ROUTE_ORDER: ClassVar[tuple[str, ...]] = ("researcher", "analyst", "writer")

    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        transcript = "\n".join(str(m.content) for m in messages)
        if "You are the supervisor" in transcript:
            done = [r for r in self.ROUTE_ORDER if f"[{r}]" in transcript]
            nxt = next((r for r in self.ROUTE_ORDER if r not in done), "FINISH")
            text = '{"next": "%s"}' % nxt
        else:
            last = str(messages[-1].content)
            text = f"(mock) 已围绕任务完成工作：{last.splitlines()[-1][:120]}"
        msg = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def build_llm() -> tuple[BaseChatModel, str]:
    api_key = _env("LLM_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        return MockChatModel(), "mock(离线)"
    from langchain_openai import ChatOpenAI

    model = _env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL")
    return (
        ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0),
        f"{model} @ {base_url or 'api.openai.com'}",
    )
