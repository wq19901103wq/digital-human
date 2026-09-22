"""OpenAI / Anthropic 调用：端点与密钥走环境变量，行为参数随版本冻结。"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from . import cache, tracing
from .iteration.control import StopRequested, request

_logger = logging.getLogger("digital_human.llm")


class LLMError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMRefusal(LLMError):
    """接口明确拒答；保留为单题失败，不按网络或输出格式错误重试。"""


class ChatClient:
    def __init__(self, settings: dict[str, Any], model_cfg: dict[str, Any]):
        llm = settings["llm"]
        base_env = model_cfg.get("base_url_env", llm["base_url_env"])
        key_env = model_cfg.get("api_key_env", llm["api_key_env"])
        self._base_url = os.environ.get(base_env, "").rstrip("/")
        self._api_key = os.environ.get(key_env, "")
        if not self._base_url or not self._api_key:
            raise LLMError(f"请在项目 .env 或环境变量中设置 {base_env} 和 {key_env}")
        self._model = model_cfg["model"]
        self._temperature = model_cfg.get("temperature", 0.7)
        self._max_tokens = int(model_cfg.get("max_tokens", 2500))
        self._timeout = float(model_cfg.get("timeout_seconds", 60))
        self._max_retries = int(llm.get("max_retries", 2))
        self._model_revision = model_cfg.get("model_revision")
        self._protocol = model_cfg.get("protocol") or (
            "anthropic" if "/api/coding" in urlsplit(self._base_url).path else "openai")
        if self._protocol not in {"openai", "anthropic"}:
            raise LLMError(f"不支持的 LLM 协议：{self._protocol}")
        self._client = None
        if self._protocol == "openai":
            from openai import OpenAI
            # 重试统一由本类控制，避免 SDK 和外层叠加多次重试。
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=self._timeout, max_retries=0)

    @property
    def model_name(self) -> str:
        return self._model

    def cache_identity(self):
        return {"provider": self._protocol, "endpoint_sha256": cache.digest(self._base_url),
                "model": self._model, "temperature": self._temperature, "max_tokens": self._max_tokens,
                "model_revision": self._model_revision, "adapter": cache.code_digest(__file__)}

    def _error(self, exc: Exception) -> LLMError:
        if isinstance(exc, StopRequested):
            raise exc
        if isinstance(exc, LLMError):
            return exc
        status = getattr(exc, "status_code", None)
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        else:
            detail = str(exc)
        detail = detail.replace(self._api_key, "[redacted]")[:500]
        hints = {401: "密钥无效", 402: "当前接口账户余额不足", 403: "当前接口无模型权限",
                 404: "当前接口未提供该模型或模型未开通", 429: "接口限流"}
        if status:
            return LLMError(f"{self._model} 调用失败 HTTP {status}（{hints.get(status, '接口错误')}）：{detail}",
                            retryable=status in {408, 409, 429} or status >= 500)
        if isinstance(exc, (TimeoutError, urllib.error.URLError)) or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}:
            return LLMError(f"{self._model} 连接失败或超过 {self._timeout:g} 秒：{detail}", retryable=True)
        return LLMError(f"{self._model} 调用失败：{detail}")

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        body = {"model": self._model, "messages": messages,
                "temperature": self._temperature, "max_tokens": self._max_tokens}
        if self._protocol == "anthropic":
            system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
            if json_mode:
                system = (system + "\n只输出合法 JSON，不要输出任何其他文字。").strip()
            body["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
            if system:
                body["system"] = system
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        return cache.memo("llm_request", {"client": self.cache_identity(), "body": body},
                          lambda: self._send(body), valid=cache.response_valid)

    def _send(self, body):
        for attempt in range(self._max_retries + 1):
            try:
                with request(), tracing.step("llm", {"protocol": self._protocol, "body": body,
                        "attempt": attempt + 1, "max_attempts": self._max_retries + 1,
                        "timeout_seconds": self._timeout}) as event:
                    try:
                        if self._protocol == "anthropic":
                            return self._chat_anthropic(body, event)
                        response = self._client.chat.completions.create(**body)
                        choice = response.choices[0]
                        usage = getattr(response, "usage", None)
                        event["response"] = {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                                             "usage": usage.model_dump() if hasattr(usage, "model_dump") else None}
                        if choice.finish_reason == "length":
                            raise LLMError(f"{self._model} 输出被 max_tokens={self._max_tokens} 截断")
                        text = choice.message.content or ""
                        if not text.strip():
                            raise LLMError(f"{self._model} 返回空文本", retryable=True)
                        return text
                    except Exception as exc:
                        # 先去除密钥再写入实录，包括 HTTP 错误响应。
                        raise self._error(exc) from exc
            except Exception as exc:
                error = self._error(exc)
                if not error.retryable or attempt == self._max_retries:
                    raise error from exc
                _logger.warning("LLM 暂时失败，第 %d 次重试：%s", attempt + 1, error)
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("LLM 重试循环未返回")

    def _chat_anthropic(self, body: dict, event: dict) -> str:
        url = self._base_url
        if not url.endswith("/messages"):
            url += "/messages" if url.endswith("/v1") else "/v1/messages"
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers={"x-api-key": self._api_key,
                                                  "Authorization": f"Bearer {self._api_key}",
                                                  "Content-Type": "application/json",
                                                  "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data = json.loads(response.read())
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        event["response"] = {"text": text, "stop_reason": data.get("stop_reason"), "usage": data.get("usage"),
                             "content_types": [block.get("type") for block in data.get("content", [])]}
        if data.get("type") == "error" or data.get("error"):
            raise LLMError(f"{self._model} 返回错误：{data.get('error', {}).get('type', 'unknown')}")
        if data.get("stop_reason") == "refusal":
            raise LLMRefusal(f"{self._model} 接口拒答（stop_reason=refusal）")
        # 即使有部分 JSON 也不能把截断结果当成功。
        if data.get("stop_reason") == "max_tokens":
            raise LLMError(f"{self._model} 输出被 max_tokens={self._max_tokens} 截断")
        if not text.strip():
            types = [block.get("type") for block in data.get("content", [])]
            tokens = (data.get("usage") or {}).get("output_tokens", 0)
            if tokens >= self._max_tokens:
                raise LLMError(f"{self._model} 思考已用满 max_tokens={self._max_tokens}，未生成回复")
            raise LLMError(f"{self._model} 返回空文本（结束原因 {data.get('stop_reason')}，"
                           f"内容类型 {types}，输出 token {tokens}）", retryable=True)
        return text


def build_clients(settings: dict[str, Any], llm_cfg: dict[str, Any]) -> dict[str, ChatClient]:
    if "private" in llm_cfg or "group" in llm_cfg:
        clients = {key: ChatClient(settings, llm_cfg[key]) for key in ("private", "group") if key in llm_cfg}
        fallback = next(iter(clients.values()))
        return {key: clients.get(key, fallback) for key in ("private", "group")}
    client = ChatClient(settings, llm_cfg)
    return {"private": client, "group": client}
