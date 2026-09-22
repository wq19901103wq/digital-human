import io
import json
import sys
import urllib.error
from types import SimpleNamespace

import pytest

from src import config, llm


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("TEST_LLM_URL", "https://example.test/api/coding")
    monkeypatch.setenv("TEST_LLM_KEY", "secret-test-key")
    return {"llm": {"base_url_env": "TEST_LLM_URL", "api_key_env": "TEST_LLM_KEY", "max_retries": 2}}


def reply(payload=None):
    return io.BytesIO(json.dumps(payload or {"content": [{"type": "text", "text": '{"ok":true}'}], "stop_reason": "end_turn"}).encode())


def test_anthropic_uses_config_and_plain_url(settings, monkeypatch):
    calls = []
    def urlopen(request, timeout):
        calls.append((request, timeout))
        return reply()
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    client = llm.ChatClient(settings, {"model": "test", "timeout_seconds": 17, "temperature": 0.2, "max_tokens": 1234})
    assert json.loads(client.chat([{"role": "user", "content": "test"}], json_mode=True)) == {"ok": True}
    request, timeout = calls[0]
    assert request.full_url == "https://example.test/api/coding/v1/messages"
    assert timeout == 17
    body = json.loads(request.data)
    assert body["temperature"] == 0.2 and body["max_tokens"] == 1234
    assert "JSON" in body["system"]  # 只有 user 消息也须遵守 JSON 输出约束


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_permanent_failures_do_not_retry_or_leak_key(settings, monkeypatch, status):
    attempts = []
    def urlopen(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.HTTPError("https://example.test", status, "error", {}, io.BytesIO(b"secret-test-key"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: pytest.fail("permanent error must not sleep"))
    with pytest.raises(llm.LLMError, match=f"HTTP {status}") as error:
        llm.ChatClient(settings, {"model": "test"}).chat([])
    assert len(attempts) == 1 and "secret-test-key" not in str(error.value)


def test_temporary_error_retries_and_recovers(settings, monkeypatch):
    attempts, sleeps = [], []
    def urlopen(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("https://example.test", 429, "limited", {}, io.BytesIO(b"limited"))
        return reply()
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    assert llm.ChatClient(settings, {"model": "test"}).chat([])
    assert len(attempts) == 2 and sleeps == [1]


def test_timeout_retries_twice_with_configured_limit(settings, monkeypatch):
    calls, sleeps = [], []
    def urlopen(request, timeout):
        calls.append(timeout)
        raise TimeoutError("timed out")
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    with pytest.raises(llm.LLMError, match="23 秒"):
        llm.ChatClient(settings, {"model": "test", "timeout_seconds": 23}).chat([])
    assert calls == [23, 23, 23] and sleeps == [1, 2]


def test_partial_text_with_truncation_is_failure(settings, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: reply(
        {"content": [{"type": "text", "text": '{"replies":['}], "stop_reason": "max_tokens"}))
    with pytest.raises(llm.LLMError, match="截断"):
        llm.ChatClient(settings, {"model": "test"}).chat([])


@pytest.mark.parametrize('text', ['您的问题我无法回答。', '{"replies":["拒答"]}'])
def test_explicit_refusal_is_failure_without_retry(settings, monkeypatch, text):
    calls = []
    def urlopen(*args, **kwargs):
        calls.append(1)
        return reply({'content': [{'type': 'text', 'text': text}], 'stop_reason': 'refusal'})
    monkeypatch.setattr(llm.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(llm.time, 'sleep', lambda *_: pytest.fail('refusal must not retry'))
    with pytest.raises(llm.LLMRefusal, match='stop_reason=refusal') as error:
        llm.ChatClient(settings, {'model': 'test'}).chat([], json_mode=True)
    assert len(calls) == 1
    assert error.value.retryable is False


def test_reasoning_exhaustion_is_not_retried_as_network_error(settings, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: reply(
        {"content": [{"type": "thinking", "thinking": "private reasoning"}],
         "stop_reason": "end_turn", "usage": {"output_tokens": 2500}}))
    monkeypatch.setattr(llm.time, "sleep", lambda *_: pytest.fail("token limit cannot recover by retrying"))
    with pytest.raises(llm.LLMError, match="思考已用满"):
        llm.ChatClient(settings, {"model": "test", "max_tokens": 2500}).chat([])


def test_openai_has_one_retry_layer_and_honors_parameters(settings, monkeypatch):
    monkeypatch.setenv("TEST_LLM_URL", "https://example.test/v1")
    seen = {}
    def create(**kwargs):
        seen["request"] = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok":true}'))])
    def constructor(**kwargs):
        seen["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    llm.ChatClient(settings, {"model": "test", "max_tokens": 765, "timeout_seconds": 31}).chat([], True)
    assert seen["client"]["max_retries"] == 0 and seen["client"]["timeout"] == 31
    assert seen["request"]["max_tokens"] == 765


def test_project_env_loads_without_overriding_explicit_env(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/settings.yaml").write_text("llm: {}\n")
    (tmp_path / ".env").write_text("TEST_PROJECT_KEY=from-file\nTEST_PROJECT_OVERRIDE=from-file\n")
    monkeypatch.delenv("TEST_PROJECT_KEY", raising=False)
    monkeypatch.setenv("TEST_PROJECT_OVERRIDE", "explicit")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    config.load_settings()
    assert llm.os.environ["TEST_PROJECT_KEY"] == "from-file"
    assert llm.os.environ["TEST_PROJECT_OVERRIDE"] == "explicit"
    monkeypatch.delenv("TEST_PROJECT_KEY")
