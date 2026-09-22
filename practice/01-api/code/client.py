"""P1 OpenAI 兼容 API 客户端封装：.env 配置 / 指数退避重试 / 结构化输出。

设计取向：不用 openai SDK，直接基于 httpx 封装——把 SDK 替你做的事
（鉴权头、重试、JSON 解析）摊开写一遍，之后用任何兼容服务
（OpenAI / 中转 / 本地 llama.cpp server / vLLM）只是换 base_url。

密钥只从环境变量/.env 读取（AGENTS.md 安全红线），绝不进代码与日志。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

RETRYABLE_STATUS = {429, 500, 502, 503, 504}  # 限流与临时性服务端错误


class LLMClientError(RuntimeError):
    """客户端层错误（配置缺失/重试耗尽/响应不合法），message 面向用户。"""


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 0.5  # 首次退避秒数，指数翻倍


def load_config(env_file: str | None = None) -> LLMConfig:
    """从环境变量/.env 读配置。key 缺失时抛带指引的错误。"""
    load_dotenv(env_file)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise LLMClientError(
            "未配置 OPENAI_API_KEY：复制 .env.example 为 .env 并填入，"
            "或 export OPENAI_API_KEY=sk-...（密钥不要写进代码）"
        )
    return LLMConfig(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", LLMConfig.base_url),
        model=os.environ.get("OPENAI_MODEL", LLMConfig.model),
    )


class LLMClient:
    """最小但健壮的 OpenAI 兼容 chat 客户端。transport 可注入供测试 mock。"""

    def __init__(self, config: LLMConfig | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config or load_config()
        self._transport = transport  # 测试注入 httpx.MockTransport；生产为 None 走真实网络

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """单轮/多轮对话，返回首个 choice 的文本。可重试错误自动指数退避。"""
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = self._request_with_retry(payload)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMClientError(f"响应结构异常: {body!r:.200}") from e
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError(f"响应内容为空: {body!r:.200}")
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Any:
        """要求模型输出 JSON 并解析返回。解析失败给一次「修复重试」。"""
        messages = [*messages]
        for attempt in range(2):
            raw = self.chat(messages, model=model, temperature=temperature, json_mode=True)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise LLMClientError(f"模型两次都未返回合法 JSON: {raw[:200]!r}") from None
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "你上次的输出不是合法 JSON，请只输出合法 JSON，不要任何解释。"},
                ]
        raise LLMClientError("unreachable")

    # ---------- 内部：带重试的 POST ----------

    def _request_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                time.sleep(self.config.backoff_base * 2 ** (attempt - 1))
            try:
                with httpx.Client(timeout=self.config.timeout, transport=self._transport) as client:
                    resp = client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as e:  # 网络层：超时/连接失败，值得重试
                last_err = e
                continue
            if resp.status_code in RETRYABLE_STATUS:
                last_err = LLMClientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            if resp.status_code != 200:
                raise LLMClientError(f"HTTP {resp.status_code}（不重试）: {resp.text[:200]}")
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                raise LLMClientError(f"响应不是 JSON: {resp.text[:200]!r}") from e
        raise LLMClientError(f"重试 {self.config.max_retries} 次后仍失败: {last_err}")
