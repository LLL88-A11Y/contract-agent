"""HTTP 接口。

第 1 小步只有两个：/health 检查服务在不在；/ask 流式回答一个问题，
用来确认"FastAPI 收到请求 → 调模型 → 边生成边推给前端"这条链路是通的。
"""

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .llm import stream_chat

app = FastAPI(title="租房合同助手", version="0.1.0")


class AskRequest(BaseModel):
    # 请求体的格式由 Pydantic 校验：不是字符串、空字符串、太长，FastAPI 都会直接返回 422
    question: str = Field(min_length=1, max_length=2000)


def sse(event: dict) -> str:
    """拼一条 SSE 消息：'data: ' 后面跟一行 JSON，再空一行表示这条结束。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": "你是租房合同助手，用简短的中文回答。"},
            {"role": "user", "content": req.question},
        ]
        try:
            async for text in stream_chat(messages):
                yield sse({"type": "delta", "text": text})
            yield sse({"type": "done"})
        except Exception as e:  # 模型出错也要给前端一个明确的结尾，而不是连接突然断掉
            yield sse({"type": "error", "message": f"模型调用失败：{e}"})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
