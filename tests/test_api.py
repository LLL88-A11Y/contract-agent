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


def test_review_returns_info_risks_and_missing(monkeypatch):
    from contract_agent import risks
    from contract_agent.extract import verify
    from contract_agent.risks import verify_hits

    async def fake_extract(clauses):
        return verify({}, clauses)

    async def fake_match(clauses):
        text = "乙方逾期支付租金的，每逾期一日，应按当期租金的 1% 向甲方支付滞纳金。"
        return verify_hits({"hits": [{"risk": "R06", "how": "每天 1%", "quotes": [{"clause": "c3", "text": text}]}]}, clauses, risks.load())

    monkeypatch.setattr(risks, "extract", fake_extract)
    monkeypatch.setattr(risks, "match", fake_match)
    r = client.post("/contracts/review", files={"file": ("zufang-01.pdf", (SAMPLES / "zufang-01.pdf").read_bytes(), "application/pdf")})
    assert r.status_code == 200
    report = r.json()["risks"]
    assert [f["risk"]["id"] for f in report["findings"]] == ["R06"]
    assert "note" not in report["findings"][0]["risk"]
    # 假的抽取什么都没抽到：4 项"没写也提示"都要列出来
    assert [m["risk"]["id"] for m in report["missing"]] == ["R02", "R09", "R11", "R13"]
