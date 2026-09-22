"""P1 DoD 验证（DoD 见 DESIGN.md 3.1 / 章 README）——全部走 httpx.MockTransport，零网络、零 API key。

1) .env/环境变量配置加载与缺失时的清晰报错
2) 重试逻辑：429/5xx/网络错误指数退避后成功；耗尽后抛错；4xx 不重试
3) 结构化输出：JSON 解析成功；非法 JSON 触发修复重试；两次失败明确报错
4) prompt 实验框架：mock 下生成对照结果表
"""

from __future__ import annotations

import json

import httpx
import pytest
from client import LLMClient, LLMClientError, LLMConfig, load_config
from prompt_experiments import run_experiments


def ok_body(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def _config(**kw) -> LLMConfig:
    return LLMConfig(api_key="sk-test", max_retries=kw.get("max_retries", 3), backoff_base=0.0)


# ---------- 配置加载 ----------


def test_load_config_from_env(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-file\nOPENAI_BASE_URL=http://localhost:8080/v1\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = load_config(str(env))
    assert cfg.api_key == "sk-from-file"
    assert cfg.base_url == "http://localhost:8080/v1"


def test_load_config_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
        load_config(str(tmp_path / ".env"))


# ---------- 重试 ----------


def test_retry_on_429_then_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, content=ok_body("hello"))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert len(calls) == 3


def test_retry_exhausted_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = LLMClient(_config(max_retries=2), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMClientError, match="重试 2 次后仍失败"):
        client.chat([{"role": "user", "content": "hi"}])


def test_client_error_no_retry():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMClientError, match="401"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1  # 4xx 立即失败，不浪费配额


def test_network_error_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=ok_body("recovered"))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "hi"}]) == "recovered"


# ---------- 结构化输出 ----------


def test_chat_json_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, content=ok_body('{"sentiment": "pos", "score": 0.9}'))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    assert client.chat_json([{"role": "user", "content": "分析情感"}]) == {"sentiment": "pos", "score": 0.9}


def test_chat_json_repair_retry():
    replies = ["不是 JSON 的回答", '{"ok": true}']

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ok_body(replies.pop(0)))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    assert client.chat_json([{"role": "user", "content": "x"}]) == {"ok": True}


def test_chat_json_fails_after_two_bad_outputs():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ok_body("还是不是 JSON"))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMClientError, match="合法 JSON"):
        client.chat_json([{"role": "user", "content": "x"}])


def test_malformed_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"unexpected": []}).encode())

    client = LLMClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMClientError, match="响应结构异常"):
        client.chat([{"role": "user", "content": "hi"}])


# ---------- prompt 实验框架（mock 驱动） ----------


def test_run_experiments_produces_table():
    # 每种 prompt 变体给一个可区分的固定回复
    def handler_resp(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        tag = "简短" if "一句话" in prompt else ("分点" if "分点" in prompt else "默认")
        return httpx.Response(200, content=ok_body(f"{tag}回答"))

    client = LLMClient(_config(), transport=httpx.MockTransport(handler_resp))
    rows = run_experiments(client, "什么是注意力机制")
    assert len(rows) == 3  # >=3 组对照（DoD）
    assert {r["variant"] for r in rows} == {"baseline", "concise", "structured"}
    assert all(r["reply"] and r["latency_ms"] >= 0 for r in rows)
