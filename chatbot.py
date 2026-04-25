# chatbot.py

import json
import os
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]

from tools import tools


class ChatBot:
    """
    ChatBot 类：实现一个支持多轮对话 + 工具调用的 Agent
    """

    def __init__(self, model_name: str, api_key: str):
        """
        初始化 ChatBot

        参数:
            model_name: 使用的模型名称（如 deepseek-chat）
            api_key: API Key
        """

        # 初始化 LLM（工具+RAG 二次调用时上下文很长，默认超时放宽，可用环境变量覆盖）
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0,  # 低温度让输出更稳定
            request_timeout=_timeout,
            max_retries=_retries,
        )

        # 初始化对话历史（加入 System Prompt）
        self.messages = [
            SystemMessage(content=self.build_system_prompt())
        ]

    # =========================
    # Prompt 相关
    # =========================

    def build_system_prompt(self) -> str:
        """
        构建系统提示词（告诉模型如何使用工具）
        """
        tool_desc = []

        for name, tool in tools.items():
            tool_desc.append(f"{name}: {tool['schema']['description']}")

        return f"""
你是「原神深渊与养成」方向的助手，语气清晰、友好，回答尽量结构化（可分点、小标题），避免无根据的臆测。

【任务与工具使用原则】
1. 深渊相关（含深境螺旋、12层、配队、敌人机制、打法要点、环境Buff、版本环境等）：知识以本地知识库为准。请先使用 rag_search，用简短、可检索的中文关键词构造 query（可包含：层数、半区、Boss/精英名、元素、机制关键词）。若一次检索结果不足，可换关键词再调一次 rag_search，再结合检索内容作答；不得编造知识库中未出现的关键机制或数值。
2. 简单计算（伤害期望、词条对比等纯算式）：使用 calculator，将完整表达式放入 arguments。
3. 泛化搜索或百科式补充：在 rag_search 仍不足且用户明确需要更广信息时，可使用 search；若与深渊攻略冲突，以 rag_search 结果为准。
4. 玩家「当前账号」信息（如已拥有角色、等级、武器、圣遗物、深渊进度等需查数据库的内容）：仅当下方工具列表里已提供对应工具时才可调用；若当前列表中尚无此类工具，应直接说明暂不支持查库、请用户自行描述或稍后再试，严禁捏造账号数据。

【输出格式】
仅输出一个 JSON 对象，不要 Markdown 代码围栏、不要前后解释文字。

需要调用工具时：
{{
  "tool": "工具名称",
  "arguments": {{
    "参数名": "参数值"
  }}
}}

不需要调用工具时（例如寒暄、已能仅凭上文回答、或工具不可用时的说明）：
{{
  "answer": "你的回答"
}}

【当前可用工具】
{chr(10).join(tool_desc)}
"""

    # =========================
    # 工具调用相关
    # =========================

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        调用工具

        参数:
            tool_name: 工具名称
            arguments: 工具参数（dict）

        返回:
            工具执行结果
        """
        print(f"[Tool调用] {tool_name} 参数: {arguments}")

        tool = tools.get(tool_name)
        if not tool:
            return "工具不存在"

        try:
            result = tool["func"](**arguments)
        except Exception as e:
            result = f"工具执行失败: {str(e)}"

        print(f"[Tool返回] {result}")

        return result

    # =========================
    # 响应解析
    # =========================

    def parse_response(self, content: str) -> dict:
        """
        解析 LLM 输出（JSON）

        参数:
            content: LLM返回的文本

        返回:
            dict（结构化结果）
        """
        try:
            return json.loads(content)
        except Exception:
            # 如果解析失败，当作普通回答
            return {"answer": content}

    # =========================
    # 上下文控制
    # =========================

    def trim_messages(self, max_len: int = 10):
        """
        限制对话历史长度（防止token爆炸）
        """
        if len(self.messages) > max_len:
            self.messages = [self.messages[0]] + self.messages[-max_len:]

    # =========================
    # 主对话逻辑
    # =========================

    def chat(self, user_input: str) -> str:
        """
        主对话函数

        参数:
            user_input: 用户输入

        返回:
            AI回复
        """

        # 1️⃣ 添加用户消息
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()

        # 2️⃣ 调用 LLM
        print("[LLM] 正在请求模型（首次）...")
        response = self.llm.invoke(self.messages)
        print("[LLM] 首次响应完成")
        content = response.content

        print("[LLM输出]", content)

        # 3️⃣ 解析 JSON
        data = self.parse_response(content)

        # 4️⃣ 如果需要调用工具
        if "tool" in data:
            tool_name = data["tool"]
            arguments = data.get("arguments", {})

            # 调用工具
            result = self.call_tool(tool_name, arguments)

            # 把工具结果喂回 LLM
            self.messages.append(response)
            self.messages.append(
                HumanMessage(
                    content=f"""
                    工具返回结果: {result}

                    请基于该结果，严格按照以下JSON格式返回：

                    {{
                         "answer": "你的最终回答"
                    }}

                    要求：
                    - answer 必须是一行字符串
                    - 不要使用换行符（\n）
                    - 不要使用列表或Markdown格式
                    - 内容简洁、自然
                    """
                )
            )

            # 再次调用 LLM 得到最终答案
            print("[LLM] 正在生成最终回答...")
            final_response = self.llm.invoke(self.messages)
            print("[LLM] 最终回答生成完成")
            self.messages.append(final_response)

            return final_response.content

        # 5️⃣ 普通回答
        else:
            self.messages.append(response)
            return data.get("answer", content)