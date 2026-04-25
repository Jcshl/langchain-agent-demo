from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]


print("chatbot module loaded")
class ChatBot:
    def __init__(self,model_name, api_key):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1"
        )
        
        self.messages = [
            SystemMessage(content="你是一个专业的AI面试官")
        ]

    def trim_messages(self, max_len=10):
        if len(self.messages) > max_len:
            self.messages = [self.messages[0]] + self.messages[-max_len:]

    def chat(self, user_input):
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response.content