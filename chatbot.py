# chatbot.py

import os
from typing import Any

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import (  # pyright: ignore[reportMissingImports]
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tools import lc_tools


class ChatBot:
    """
    ChatBot：多轮对话 + 原生 tool calling + 工具循环（ReAct 风格上限）
    """

    def __init__(self, model_name: str, api_key: str):
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0,
            request_timeout=_timeout,
            max_retries=_retries,
        )
        self.llm_tools = self.llm.bind_tools(lc_tools)
        self._tool_by_name = {t.name: t for t in lc_tools}
        try:
            self._max_agent_steps = int((os.getenv("AGENT_MAX_ITERATIONS") or "12").strip())
        except ValueError:
            self._max_agent_steps = 12
        self._max_agent_steps = max(1, min(self._max_agent_steps, 50))

        self.messages: list = [
            SystemMessage(content=self.build_system_prompt()),
        ]

    def clear_history(self) -> None:
        """清空多轮对话，仅保留 system prompt（用于网页端「新对话」）。"""
        self.messages = [SystemMessage(content=self.build_system_prompt())]

    def build_system_prompt(self) -> str:
        lines = [
            "你是「原神深渊与养成」方向的助手，语气清晰、友好，回答尽量结构化（可分点、小标题），避免无根据的臆测。",
            "",
            "【任务与工具使用原则】",
            "1. 深渊相关（含深境螺旋、12层、配队、敌人机制、打法要点、环境Buff、版本环境等）：以本地知识库为准。请先使用 rag_search，用简短、可检索的中文关键词构造 query；若结果不足可换关键词再次 rag_search；不得编造知识库中未出现的关键机制或数值。",
            "2. 简单计算（伤害期望、词条对比等纯算术式）：使用 calculator，仅写数值与运算符。",
            "3. 泛化搜索：在 rag_search 仍不足且用户需要更广信息时可使用 search；若与深渊攻略冲突，以 rag_search 为准。",
            "4. 玩家「当前账号」信息（角色持有、库存等）：若下列工具列表中无对应工具，请直接说明暂不支持查库，严禁捏造数据。",
            "",
            "【当前可用工具】",
        ]
        for t in lc_tools:
            desc = (t.description or "").strip().replace("\n", " ")
            lines.append(f"- {t.name}: {desc}")
        return "\n".join(lines)

    def trim_messages(self, max_len: int = 24):
        """保留 system + 最近若干条消息（工具循环会较快占满上下文）。"""
        if len(self.messages) > max_len:
            self.messages = [self.messages[0]] + self.messages[-max_len + 1 :]

    def _invoke_tool(self, name: str, args: dict[str, Any]) -> str:
        print(f"[Tool调用] {name} 参数: {args}")
        tool = self._tool_by_name.get(name)
        if tool is None:
            return f"未知工具: {name}"
        try:
            out = tool.invoke(args)
        except Exception as e:
            out = f"工具执行失败: {e}"
        print(f"[Tool返回] {out}")
        return str(out)

    def _run_tool_loop(self) -> str:
        """在已有 self.messages 末尾为用户轮次的前提下，多轮 tool 直至无 tool_calls 或达上限。"""
        steps = 0
        while steps < self._max_agent_steps:
            steps += 1
            print(f"[LLM] 正在请求模型（第 {steps} 步）...")
            response = self.llm_tools.invoke(self.messages)
            print("[LLM] 响应完成")

            if not isinstance(response, AIMessage):
                self.messages.append(response)
                return getattr(response, "content", "") or ""

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                self.messages.append(response)
                return (response.content or "").strip()

            self.messages.append(response)

            for tc in tool_calls:
                if isinstance(tc, dict):
                    tid = tc.get("id") or ""
                    name = tc.get("name") or ""
                    args = tc.get("args") or {}
                else:
                    tid = getattr(tc, "id", "") or ""
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", None) or {}

                payload = self._invoke_tool(name, args)
                self.messages.append(ToolMessage(content=payload, tool_call_id=tid))

            self.trim_messages()

        # 步数用尽：在不绑定工具的情况下强制收束为自然语言回答
        print("[Agent] 已达 AGENT_MAX_ITERATIONS，请求最终回答（不再调用工具）...")
        self.messages.append(
            HumanMessage(
                content="工具调用次数已达上限。请仅根据已有对话与工具返回，直接给出最终中文回答，不要再调用任何工具。"
            )
        )
        final = self.llm.invoke(self.messages)
        self.messages.append(final)
        return (final.content or "").strip()

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()

        reply = self._run_tool_loop()
        return reply
