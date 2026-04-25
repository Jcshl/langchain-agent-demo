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
    model_name=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)


# CLI 循环
while True:
    user_input = input("用户输入：")

    if user_input.lower() in ["exit", "quit"]:
        break

    reply = bot.chat(user_input)
    print("AI：", reply)