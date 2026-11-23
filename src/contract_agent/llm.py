"""调用大模型。中转接口是 OpenAI 兼容的 /chat/completions。
直接用 httpx 发请求、自己解析流式数据，不用官方 SDK，方便看清楚每一步在做什么。"""

import json
from collections.abc import AsyncIterator

import httpx

from .config import settings

# 连上接口最多等 10 秒；之后每一段数据之间最多等 60 秒
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


async def stream_chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> AsyncIterator[str]:
    """流式调用：模型每生成一小段文字，就 yield 一段。

    接口返回的是一行一行的 "data: {...}"，最后一行是 "data: [DONE]"。
    每一行里，新生成的文字在 choices[0].delta.content。
    """
    s = settings()
    payload = {"model": s.llm_model, "messages": messages, "temperature": temperature, "stream": True}
    headers = {"Authorization": f"Bearer {s.llm_api_key}"}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", f"{s.llm_base_url}/chat/completions", json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                raise RuntimeError(f"模型接口返回 {resp.status_code}：{body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                text = (choices[0].get("delta") or {}).get("content") if choices else None
                if text:
                    yield text
