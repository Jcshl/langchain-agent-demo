# streamlit_app.py
# 运行：streamlit run streamlit_app.py
# 需先在 .env 配置 MODEL_NAME、SILICONFLOW_API_KEY（与 main.py 相同）

import os

import streamlit as st
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

from chatbot import ChatBot

load_dotenv()


def _ensure_bot() -> ChatBot:
    if st.session_state.get("bot") is None:
        model = (os.getenv("MODEL_NAME") or "").strip()
        key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
        if not model or not key:
            st.error("请在项目根目录 `.env` 中配置 `MODEL_NAME` 与 `SILICONFLOW_API_KEY`。")
            st.stop()
        st.session_state.bot = ChatBot(model_name=model, api_key=key)
    return st.session_state.bot


st.set_page_config(page_title="对话", layout="centered")
st.title("原神深渊与养成助手")

with st.sidebar:
    st.caption("与 `main.py` 共用同一套 ChatBot（工具 + RAG）。")
    if st.button("新对话"):
        st.session_state.messages = []
        bot = st.session_state.get("bot")
        if bot is not None:
            bot.clear_history()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

bot = _ensure_bot()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入你的问题…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                reply = bot.chat(prompt)
            except Exception as e:
                reply = f"请求出错：{e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
