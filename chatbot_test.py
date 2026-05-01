from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]


class ChatBot:
    """简化版测试 ChatBot：用于验证基础多轮对话能力。"""

    def __init__(self,model_name, api_key):
        """
        初始化测试模型客户端。

        参数:
            model_name: 模型名称。
            api_key: API Key。
        """
        # LLM 客户端（OpenAI 兼容协议）。
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1"
        )

        # messages：多轮对话历史，首条为系统提示词。
        self.messages = [
            SystemMessage(content="你是一个专业的AI面试官")
        ]

    def trim_messages(self, max_len=100):
        """限制历史消息长度，避免上下文无限增长。"""
        if len(self.messages) > max_len:
            self.messages = [self.messages[0]] + self.messages[-max_len:]

    def chat(self, user_input):
        """
        执行一轮对话并返回模型文本。

        参数:
            user_input: 用户本轮输入。
        """
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()
        # response：模型返回的消息对象（包含 content）。
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response.content