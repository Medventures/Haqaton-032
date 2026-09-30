"""Provider contract tests; all network transports are faked, no API keys used."""

import asyncio
import json

import httpx
import pytest

from app import providers


SCHEMA = {
    "type": "object",
    "title": "Консультация",
    "properties": {
        "complaints": {"type": ["string", "null"], "title": "Жалобы", "x-ui": "textarea", "maxLength": 400},
        "allergy_status": {
            "type": ["string", "null"], "title": "Аллергии", "enum": ["denied", "present", None],
            "x-enumLabels": {"denied": "Отрицает", "present": "Есть"},
        },
    },
    "required": ["complaints", "allergy_status"],
    "additionalProperties": False,
}


def test_demo_uses_labels_and_preserves_unknown_not_negative():
    assert providers.demo_extract(SCHEMA, [{"text": "Жалобы: Болит голова"}], {}) == {
        "complaints": "Болит голова", "allergy_status": None,
    }
    values = providers.demo_extract(SCHEMA, [
        {"text": "Жалобы: Болит голова\nАллергии: Отрицает."},
        {"text": "Жалобы: [нет сведений]\nАллергии: Неизвестный код"},
    ], {})
    assert values == {"complaints": None, "allergy_status": "denied"}


def test_schema_compilation_keeps_enum_meaning_without_ui_extensions():
    compiled = providers.llm_schema(SCHEMA)
    assert compiled["required"] == list(SCHEMA["properties"])
    assert compiled["additionalProperties"] is False
    assert "x-ui" not in compiled["properties"]["complaints"]
    assert compiled["properties"]["complaints"]["maxLength"] == 400
    allergy = compiled["properties"]["allergy_status"]
    assert "x-enumLabels" not in allergy
    assert "Отрицает" in allergy["description"]
    assert None in allergy["enum"]
    assert SCHEMA["properties"]["complaints"]["x-ui"] == "textarea"


class FakeSocket:
    def __init__(self, reject=False):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.reject = reject
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is None:
            raise StopAsyncIteration
        return json.dumps(message)

    async def send(self, message):
        self.sent.append(message)
        if not isinstance(message, str):
            return
        command = json.loads(message)
        if command["message"] == "StartRecognition":
            await self.incoming.put(
                {"message": "Error", "reason": "private payload"}
                if self.reject else {"message": "RecognitionStarted"}
            )
        elif command["message"] == "EndOfStream":
            for kind in ("AddPartialTranscript", "AddTranscript"):
                await self.incoming.put({
                    "message": kind,
                    "metadata": {"transcript": " Болит голова. ", "start_time": 0.2, "end_time": 1.4},
                })
            await self.incoming.put({"message": "EndOfTranscript"})

    async def close(self):
        self.closed = True
        await self.incoming.put(None)


@pytest.mark.asyncio
async def test_speechmatics_protocol_waits_final_and_counts_audio(monkeypatch):
    socket = FakeSocket()
    connections = []

    async def connect(url, **kwargs):
        connections.append((url, kwargs))
        return socket

    events = []

    async def handle(event):
        await asyncio.sleep(0)
        events.append(event)

    monkeypatch.setattr(providers, "connect", connect)
    asr = providers.create_asr(providers.Settings(mode="live", speechmatics_api_key="test-only"), handle)
    await asr.start(16000)
    await asr.send_audio(b"\x00\x00" * 3200)
    await asr.send_audio(b"\x01\x00" * 3200)
    await asr.finish()
    assert json.loads(socket.sent[0])["audio_format"] == {
        "type": "raw", "encoding": "pcm_s16le", "sample_rate": 16000,
    }
    assert json.loads(socket.sent[-1]) == {"message": "EndOfStream", "last_seq_no": 2}
    assert connections[0][1]["additional_headers"]["Authorization"] == "Bearer test-only"
    assert [event["type"] for event in events] == ["partial", "final"]
    assert events[0]["id"] == events[1]["id"]
    assert events[1]["startMs"] == 200
    assert events[1]["endMs"] == 1400
    with pytest.raises(providers.ProviderError, match="не принимает"):
        await asr.send_audio(b"\x00\x00")
    await asr.close()
    assert socket.closed


@pytest.mark.asyncio
async def test_speechmatics_error_before_ready_is_visible_and_sanitized(monkeypatch):
    socket = FakeSocket(reject=True)

    async def connect(*args, **kwargs):
        return socket

    async def handle(event):
        assert "private payload" not in json.dumps(event)

    monkeypatch.setattr(providers, "connect", connect)
    asr = providers.create_asr(providers.Settings(mode="live", speechmatics_api_key="test-only"), handle)
    with pytest.raises(providers.ProviderError) as failure:
        await asr.start(16000)
    assert failure.value.code == "asr_provider_error"
    assert socket.closed


@pytest.mark.asyncio
async def test_responses_uses_strict_schema_and_reads_only_output_text(monkeypatch):
    captured = []

    def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{
                    "type": "output_text", "text": '{"complaints":"Боль","allergy_status":null}',
                }]},
            ],
        })

    client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(transport), **kwargs
    ))
    values = await providers.extract(
        providers.Settings(mode="live", openai_api_key="test-only"), SCHEMA,
        [{"id": "seg_1", "text": "Боль", "revision": 1}], {},
    )
    assert values == {"complaints": "Боль", "allergy_status": None}
    assert captured[0]["store"] is False
    assert captured[0]["text"]["format"]["strict"] is True
    assert "x-ui" not in json.dumps(captured[0]["text"]["format"]["schema"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("response", "code"), [
    ({"status": "incomplete", "output": []}, "llm_incomplete"),
    ({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "private"}]}]}, "llm_refusal"),
    ({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "{broken"}]}]}, "llm_invalid_json"),
    ({"status": "completed", "output": ["unexpected"]}, "llm_invalid_response"),
])
async def test_bad_llm_responses_are_never_returned_as_values(monkeypatch, response, code):
    client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)), **kwargs
    ))
    with pytest.raises(providers.ProviderError) as failure:
        await providers.extract(providers.Settings(mode="live", openai_api_key="test-only"), SCHEMA, [], {})
    assert failure.value.code == code
    assert "private" not in str(failure.value)


@pytest.mark.asyncio
async def test_demo_audio_never_fabricates_transcripts():
    events = []

    async def handle(event):
        events.append(event)

    asr = providers.create_asr(providers.Settings(), handle)
    await asr.start(16000)
    await asr.send_audio(b"\x00\x00" * 3200)
    await asr.finish()
    assert events == []


@pytest.mark.asyncio
async def test_context_limit_fails_without_sending_request(monkeypatch):
    def fail(**kwargs):
        pytest.fail("Oversized context must not be sent")

    monkeypatch.setattr(providers.httpx, "AsyncClient", fail)
    with pytest.raises(providers.ProviderError) as failure:
        await providers.extract(
            providers.Settings(mode="live", openai_api_key="test-only", max_context_characters=1000),
            SCHEMA, [{"text": "x" * 1001}], {},
        )
    assert failure.value.code == "llm_context_limit"


def mock_chat_transport(monkeypatch, body=None, status=200):
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(status, json=body if body is not None else {
            "choices": [{"finish_reason": "stop", "message": {
                "content": '{"complaints":"Боль","allergy_status":null}',
            }}],
        })

    client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(transport), **kwargs
    ))
    return requests


@pytest.mark.asyncio
async def test_deepseek_with_demo_asr_uses_real_extraction_adapter(monkeypatch):
    requests = mock_chat_transport(monkeypatch)
    settings = providers.Settings(llm_provider="deepseek", asr_provider="demo", deepseek_api_key="test-deepseek")

    async def handle(event):
        pytest.fail("The demo audio sink must not emit recognized text")

    assert isinstance(providers.create_asr(settings, handle), providers.DemoASR)
    values = await providers.extract(settings, SCHEMA, [{"text": "Боль"}], {})
    assert values == {"complaints": "Боль", "allergy_status": None}
    request = requests[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer test-deepseek"
    payload = json.loads(request.content)
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert '"allergy_status"' in payload["messages"][0]["content"]
    assert "JSON Schema" in payload["messages"][0]["content"]
    assert payload["stream"] is False
    assert "store" not in payload


@pytest.mark.asyncio
async def test_own_llm_supports_json_schema_and_never_inherits_openai_key(monkeypatch):
    requests = mock_chat_transport(monkeypatch)
    settings = providers.Settings(
        llm_provider="openai-compatible", llm_base_url="http://llm:8000/v1/",
        llm_model="clinic-model", llm_response_format="json_schema", openai_api_key="do-not-forward",
    )
    await providers.extract(settings, SCHEMA, [], {})
    assert str(requests[0].url) == "http://llm:8000/v1/chat/completions"
    assert "authorization" not in requests[0].headers
    payload = json.loads(requests[0].content)
    assert payload["model"] == "clinic-model"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["complaints"]["maxLength"] == 400
    assert "thinking" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(("body", "code"), [
    ({"choices": []}, "llm_invalid_response"),
    ({"choices": [{"finish_reason": "length", "message": {"content": '{"complaints":'}}]}, "llm_incomplete"),
    ({"choices": [{"finish_reason": "content_filter", "message": {"content": "private"}}]}, "llm_refusal"),
    ({"choices": [{"finish_reason": "stop", "message": {"refusal": "private", "content": ""}}]}, "llm_refusal"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": None, "reasoning_content": "private"}}]}, "llm_empty_response"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]}, "llm_empty_response"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}, "llm_invalid_json"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]}, "llm_invalid_json"),
])
async def test_chat_errors_do_not_return_partial_values_or_private_body(monkeypatch, body, code):
    mock_chat_transport(monkeypatch, body)
    with pytest.raises(providers.ProviderError) as failure:
        await providers.extract(providers.Settings(llm_provider="deepseek", llm_api_key="test-only"), SCHEMA, [], {})
    assert failure.value.code == code
    assert "private" not in str(failure.value)


@pytest.mark.asyncio
async def test_provider_http_error_does_not_leak_response(monkeypatch):
    mock_chat_transport(monkeypatch, {"error": {"message": "private api-key patient text"}}, status=429)
    with pytest.raises(providers.ProviderError) as failure:
        await providers.extract(providers.Settings(llm_provider="deepseek", llm_api_key="test-only"), SCHEMA, [], {})
    assert failure.value.code == "llm_request_failed"
    assert failure.value.retryable is True
    assert "private" not in str(failure.value)


def test_independent_settings_preserve_legacy_mode_and_hide_credentials():
    legacy = providers.Settings(mode="live", openai_api_key="legacy-openai")
    assert legacy.effective_llm_provider == "openai"
    assert legacy.effective_asr_provider == "speechmatics"
    assert legacy.effective_llm_api_key == "legacy-openai"
    mixed = providers.Settings(
        mode="live", llm_provider="demo", asr_provider="openai-compatible",
        asr_api_key="private-asr", llm_api_key="private-llm", deepseek_api_key="private-deepseek",
    )
    assert mixed.effective_llm_provider == "demo"
    assert mixed.effective_asr_provider == "openai-compatible"
    assert "private-" not in repr(mixed)


def test_env_provider_specific_credentials_do_not_cross_providers(monkeypatch):
    for variable in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_RESPONSE_FORMAT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("ASR_PROVIDER", "demo")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai")
    deepseek = providers.Settings.from_env()
    assert deepseek.effective_llm_api_key == "test-deepseek"
    assert deepseek.effective_llm_model == "deepseek-flash"
    assert deepseek.effective_asr_provider == "demo"
    monkeypatch.setenv("LLM_API_KEY", "")
    assert providers.Settings.from_env().effective_llm_api_key == "test-deepseek"
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    assert providers.Settings.from_env().effective_llm_api_key == ""
    monkeypatch.setenv("LLM_API_KEY", "explicit-generic")
    assert providers.Settings.from_env().effective_llm_api_key == "explicit-generic"


@pytest.mark.parametrize("url", [
    "http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1",
    "http://transcriber:8000/v1", "http://host.docker.internal:8000/v1",
    "http://192.168.1.20:8000/v1", "https://models.example.com/v1",
])
def test_local_and_secure_custom_endpoints_are_supported(url):
    settings = providers.Settings(llm_provider="openai-compatible", llm_base_url=url)
    assert settings.effective_llm_base_url == url


@pytest.mark.parametrize("url", [
    "http://models.example.com/v1", "ftp://localhost/model", "https://secret:private@example.com/v1",
    "https://example.com/v1?key=private", "https://example.com/v1#private", "https:///v1",
])
def test_bad_endpoint_configs_fail_without_echoing_sensitive_url(url):
    with pytest.raises(ValueError) as failure:
        providers.Settings(llm_provider="openai-compatible", llm_base_url=url)
    assert "private" not in str(failure.value)


def test_public_http_requires_explicit_override():
    settings = providers.Settings(
        llm_provider="openai-compatible", llm_base_url="http://models.example.com/v1", allow_insecure_http=True,
    )
    assert settings.effective_llm_base_url == "http://models.example.com/v1"


def test_deepseek_rejects_unsupported_chat_schema_mode():
    with pytest.raises(ValueError, match="DeepSeek"):
        providers.Settings(llm_provider="deepseek", llm_response_format="json_schema")
