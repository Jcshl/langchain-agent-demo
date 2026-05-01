# api_app.py
# 运行：uvicorn api_app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/ （与 streamlit_app 共用 .env）

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]

from chatbot import ChatBot

load_dotenv()

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"

_sessions: dict[str, ChatBot] = {}
_sessions_lock = threading.Lock()


def _env_credentials() -> tuple[str, str] | None:
    model = (os.getenv("MODEL_NAME") or "").strip()
    key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
    if not model or not key:
        return None
    return model, key


def _get_or_create_bot(session_id: str | None) -> tuple[str, ChatBot]:
    cred = _env_credentials()
    if cred is None:
        raise HTTPException(
            status_code=500,
            detail="请在项目根目录 `.env` 中配置 MODEL_NAME 与 SILICONFLOW_API_KEY。",
        )
    model, api_key = cred
    sid = (session_id or "").strip() or str(uuid.uuid4())
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = ChatBot(model_name=model, api_key=api_key)
        return sid, _sessions[sid]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


app = FastAPI(title="LangChain Agent Demo", version="0.1.0")


@app.get("/")
def index_page():
    return FileResponse(_STATIC / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(body: ChatRequest):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message 不能为空")
    sid, bot = _get_or_create_bot(body.session_id)
    reply = bot.chat(msg)
    return ChatResponse(reply=reply, session_id=sid)


@app.post("/api/chat/reset")
def api_reset(body: ResetRequest):
    sid = body.session_id.strip()
    with _sessions_lock:
        _sessions.pop(sid, None)
    return {"ok": True}


app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC)),
    name="static",
)
