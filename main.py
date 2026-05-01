# main.py

"""
程序入口：运行对话系统
"""

import os
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from chatbot import ChatBot


# 加载环境变量
load_dotenv()

# 创建 ChatBot 实例
bot = ChatBot(
    # 大模型名称（例如 deepseek-chat），来自 .env。
    model_name=os.getenv("MODEL_NAME"),
    # API Key，来自 .env。
    api_key=os.getenv("SILICONFLOW_API_KEY")
)


# CLI 循环
while True:
    # user_input：用户在命令行输入的本轮问题。
    user_input = input("用户输入：")

    if user_input.lower() in ["exit", "quit"]:
        break

    # reply：模型本轮最终回复文本。
    reply = bot.chat(user_input)
    print("AI：", reply)