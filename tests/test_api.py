"""接口：上传 PDF → 按条款切好；不是 PDF、读不出文字时给出明确的错误。不调用模型。"""

from pathlib import Path

from fastapi.testclient import TestClient

from contract_agent.api import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_parse_pdf():
    r = client.post("/contracts/parse", files={"file": ("zufang-01.pdf", (SAMPLES / "zufang-01.pdf").read_bytes(), "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["pages"] == 3
    assert [c["id"] for c in body["clauses"]] == ["preamble", *[f"c{i}" for i in range(1, 11)], "sign"]


def test_not_a_pdf():
    r = client.post("/contracts/parse", files={"file": ("合同.txt", "第一条 租金".encode(), "text/plain")})
    assert r.status_code == 415


def test_broken_pdf():
    r = client.post("/contracts/parse", files={"file": ("坏的.pdf", b"%PDF-1.7 not really a pdf", "application/pdf")})
    assert r.status_code == 422


def test_extract_returns_info_and_clauses(monkeypatch):
    # 不调用模型：换成一个直接返回结果的假函数，只测接口怎么把各部分拼起来
    from contract_agent import api
    from contract_agent.extract import verify

    async def fake_extract(clauses):
        return verify({"rent": {"yuan": 3200, "quotes": [{"clause": "c3", "text": "月租金为人民币 3200 元"}]}}, clauses)

    monkeypatch.setattr(api, "extract", fake_extract)
    r = client.post("/contracts/extract", files={"file": ("zufang-01.pdf", (SAMPLES / "zufang-01.pdf").read_bytes(), "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["info"]["rent"]["yuan"] == 3200
    assert body["info"]["deposit"] is None
    assert len(body["clauses"]) == 12


def test_extract_reports_model_failure(monkeypatch):
    from contract_agent import api

    async def broken(clauses):
        raise RuntimeError("模型接口返回 503")

    monkeypatch.setattr(api, "extract", broken)
    r = client.post("/contracts/extract", files={"file": ("zufang-01.pdf", (SAMPLES / "zufang-01.pdf").read_bytes(), "application/pdf")})
    assert r.status_code == 502


def test_ask_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422
