"""HTTP 接口。

/health        服务在不在
/ask           流式回答一个问题（确认"收到请求 → 调模型 → 边生成边推给前端"这条链路是通的）
/contracts/parse    上传合同 PDF，返回按条款切好的内容
/contracts/extract  再抽出关键信息（租金、押金、租期……），每项附原文，核对不上的扔掉并说明原因
/contracts/review   关键信息 + 风险条款（和人工审核过的清单比对，每条附原文和页码）+ 合同里没写、值得问的事
/contracts/ask      就这份合同追问，流式回答，说到合同的地方附出处并由代码核对
"""

import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .clauses import Clause, split_clauses
from .extract import ContractInfo, Dropped, extract
from .risks import RiskReport, review
from .llm import stream_chat
from .qa import answer
from .pdf import NoTextError, Page, read_pages

app = FastAPI(title="租房合同助手", version="0.1.0")

MAX_PDF_BYTES = 10 * 1024 * 1024


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


class ParseResult(BaseModel):
    filename: str
    pages: int
    clauses: list[Clause]


class ExtractResult(ParseResult):
    info: ContractInfo
    dropped: list[Dropped]  # 模型给了、但没通过核对被扔掉的项，以及原因
    fixed: list[str]  # 出处写错了、按原文改正的项


async def read_upload(file: UploadFile) -> list[Page]:
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "文件超过 10 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "只支持 PDF 文件")
    try:
        # 解析 PDF 是纯计算、要花一点时间：放到线程池里跑，不卡住同时进来的其他请求
        return await run_in_threadpool(read_pages, data)
    except NoTextError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, "这个 PDF 打不开，可能已经损坏") from e


@app.post("/contracts/parse")
async def parse_contract(file: UploadFile) -> ParseResult:
    pages = await read_upload(file)
    return ParseResult(filename=file.filename or "", pages=len(pages), clauses=split_clauses(pages))


@app.post("/contracts/extract")
async def extract_contract(file: UploadFile) -> ExtractResult:
    pages = await read_upload(file)
    clauses = split_clauses(pages)
    try:
        result = await extract(clauses)
    except Exception as e:
        raise HTTPException(502, f"模型调用失败：{e}") from e
    return ExtractResult(filename=file.filename or "", pages=len(pages), clauses=clauses, info=result.info, dropped=result.dropped, fixed=result.fixed)


class ReviewResult(ExtractResult):
    risks: RiskReport


@app.post("/contracts/review")
async def review_contract(file: UploadFile) -> ReviewResult:
    pages = await read_upload(file)
    clauses = split_clauses(pages)
    try:
        extraction, risks = await review(clauses)
    except Exception as e:
        raise HTTPException(502, f"模型调用失败：{e}") from e
    return ReviewResult(
        filename=file.filename or "", pages=len(pages), clauses=clauses,
        info=extraction.info, dropped=extraction.dropped, fixed=extraction.fixed, risks=risks,
    )


@app.post("/contracts/ask")
async def ask_contract(
    file: UploadFile,
    question: str = Form(min_length=1, max_length=500),
    mode: Literal["auto", "full", "retrieve"] = Form("auto"),
) -> StreamingResponse:
    clauses = split_clauses(await read_upload(file))

    async def events() -> AsyncIterator[str]:
        try:
            async for event in answer(question, clauses, mode):
                yield sse(event)
        except Exception as e:  # 和 /ask 一样：出错也给前端一个明确的结尾
            yield sse({"type": "error", "message": f"回答失败：{e}"})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
