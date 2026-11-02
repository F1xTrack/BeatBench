"""Tests for extract_json and ModelClient from beatbench.runner.client."""
import json

import pytest

from beatbench.runner import client as client_mod
from beatbench.runner.client import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = "```json\n{\"a\": 1, \"b\": [2, 3]}\n```"
    assert extract_json(text) == {"a": 1, "b": [2, 3]}


def test_fenced_without_tag():
    text = "```\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_prose_around_json():
    text = 'Here is the result:\n\n{"a": {"nested": 1}}\n\nHope this helps!'
    assert extract_json(text) == {"a": {"nested": 1}}


def test_braces_inside_strings():
    text = '{"a": "not } a brace", "b": "{ not open }"}'
    assert extract_json(text) == {"a": "not } a brace", "b": "{ not open }"}


def test_garbage_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_unbalanced_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


class _Message:
    content = "hello"


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]
    usage = None


class _StubCompletions:
    def create(self, **kwargs):
        return _Response()


class _StubChat:
    completions = _StubCompletions()


class _StubOpenAI:
    def __init__(self, base_url=None, api_key=None):
        self.chat = _StubChat()


def test_chat_tolerates_missing_usage(monkeypatch):
    from beatbench.runner.client import ModelClient

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI)
    client = ModelClient("http://ep", "key", "model")
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "hello"
    assert reply.usage == {}


class _CapturingCompletions:
    def __init__(self):
        self.captured = {}

    def create(self, **kwargs):
        self.captured = kwargs
        return _Response()


def test_chat_forwards_response_format(monkeypatch):
    from beatbench.runner.client import ModelClient

    stub = _CapturingCompletions()

    class _StubChat2:
        completions = stub

    class _StubOpenAI2:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _StubChat2()

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI2)
    client = ModelClient("http://ep", "key", "model")
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    client.chat([{"role": "user", "content": "hi"}], response_format=fmt)
    assert stub.captured["response_format"] == fmt


def test_chat_omits_response_format_when_none(monkeypatch):
    from beatbench.runner.client import ModelClient

    stub = _CapturingCompletions()

    class _StubChat3:
        completions = stub

    class _StubOpenAI3:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _StubChat3()

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI3)
    client = ModelClient("http://ep", "key", "model")
    client.chat([{"role": "user", "content": "hi"}])
    assert "response_format" not in stub.captured


def test_chat_omits_max_tokens(monkeypatch):
    from beatbench.runner.client import ModelClient

    stub = _CapturingCompletions()

    class _StubChat4:
        completions = stub

    class _StubOpenAI4:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _StubChat4()

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI4)
    client = ModelClient("http://ep", "key", "model")
    client.chat([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in stub.captured


def test_chat_passes_reasoning_effort_as_extra_body(monkeypatch):
    from beatbench.runner.client import ModelClient

    stub = _CapturingCompletions()

    class _StubChat5:
        completions = stub

    class _StubOpenAI5:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _StubChat5()

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI5)
    client = ModelClient("http://ep", "key", "model")
    client.chat([{"role": "user", "content": "hi"}], reasoning_effort="high")
    assert stub.captured["extra_body"] == {"reasoning": {"effort": "high"}}


def test_chat_omits_extra_body_when_no_reasoning_effort(monkeypatch):
    from beatbench.runner.client import ModelClient

    stub = _CapturingCompletions()

    class _StubChat6:
        completions = stub

    class _StubOpenAI6:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _StubChat6()

    monkeypatch.setattr(client_mod, "OpenAI", _StubOpenAI6)
    client = ModelClient("http://ep", "key", "model")
    client.chat([{"role": "user", "content": "hi"}])
    assert "extra_body" not in stub.captured


def _response_with_message(message):
    msg_obj = message

    class _ChoiceR:
        message = msg_obj

    class _ResponseR:
        choices = [_ChoiceR()]
        usage = None

    return _ResponseR()


def _client_returning(monkeypatch, response):
    from beatbench.runner.client import ModelClient

    class _C:
        def create(self, **kwargs):
            return response

    class _Chat:
        completions = _C()

    class _OA:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(client_mod, "OpenAI", _OA)
    return ModelClient("http://ep", "key", "model")


def test_chat_strips_think_blocks_into_reasoning(monkeypatch):
    class Msg:
        content = "<think> step by step </think>\n\n  {\"a\": 1}  "

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "{\"a\": 1}"
    assert reply.reasoning == "step by step"


def test_chat_prefers_think_block_over_attributes(monkeypatch):
    class Msg:
        content = "<think>think-path</think>{\"a\": 1}"
        reasoning = "attr-path"

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.reasoning == "think-path"
    assert reply.content == "{\"a\": 1}"


def test_chat_reads_reasoning_attribute(monkeypatch):
    class Msg:
        content = '{"a": 1}'
        reasoning = "openrouter trace"

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.reasoning == "openrouter trace"
    assert reply.content == '{"a": 1}'


def test_chat_reads_reasoning_content_attribute(monkeypatch):
    class Msg:
        content = '{"a": 1}'
        reasoning_content = "deepseek trace"

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.reasoning == "deepseek trace"


def test_chat_serializes_model_extra_reasoning_details(monkeypatch):
    details = [{"type": "text", "text": "why not"}]

    class Msg:
        content = '{"a": 1}'
        model_extra = {"reasoning_details": details}

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.reasoning == json.dumps(details)


def test_chat_no_reasoning_source_yields_empty(monkeypatch):
    class Msg:
        content = '{"a": 1}'

    client = _client_returning(monkeypatch, _response_with_message(Msg()))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.reasoning == ""
    assert reply.content == '{"a": 1}'
