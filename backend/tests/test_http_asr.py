import asyncio
from email import policy
from email.parser import BytesParser
from io import BytesIO
import struct
import wave

import httpx
import pytest

from app.http_asr import HttpASR
from app.providers import ProviderError, Settings, create_asr


def multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
    )
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True) for part in message.iter_parts()}


def pcm(seconds, amplitude=1500):
    return struct.pack("<h", amplitude) * round(seconds * 16000)


async def feed(adapter, audio):
    for offset in range(0, len(audio), 6400):
        await adapter.send_audio(audio[offset:offset + 6400])


@pytest.mark.asyncio
async def test_wav_requests_preserve_audio_order_times_language_and_final_tail():
    requests, events, files = [], [], []

    async def handle(request):
        requests.append(request)
        fields = multipart(request)
        assert fields["model"] == b"custom-model"
        assert fields["language"] == b"kk"
        assert fields["response_format"] == b"json"
        assert request.headers["authorization"] == "Bearer synthetic-token"
        with wave.open(BytesIO(fields["file"]), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            files.append(wav.readframes(wav.getnframes()))
        return httpx.Response(200, json={"text": f" Реплика {len(requests)} "})

    async def emit(event):
        events.append(event)

    settings = Settings(asr_provider="openai-compatible", asr_base_url="http://localhost:8002/v1", asr_model="custom-model", asr_language="kk", asr_api_key="synthetic-token", asr_chunk_seconds=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = HttpASR(settings, emit, client=client)
        await adapter.start(16000)
        audio = pcm(1) + pcm(0.4, 0) + pcm(0.13)
        await feed(adapter, audio)
        await adapter.finish()
        await adapter.close()
    assert b"".join(files) == audio
    assert len(files) == 2
    assert [(event["startMs"], event["endMs"]) for event in events] == [(0, 1400), (1400, 1530)]
    assert [event["text"] for event in events] == ["Реплика 1", "Реплика 2"]
    assert all(str(request.url) == "http://localhost:8002/v1/audio/transcriptions" for request in requests)


@pytest.mark.asyncio
async def test_unbroken_speech_forces_15_second_cut_and_auto_omits_language():
    durations, events = [], []

    async def handle(request):
        fields = multipart(request)
        assert "language" not in fields
        assert "authorization" not in request.headers
        with wave.open(BytesIO(fields["file"]), "rb") as wav:
            durations.append(wav.getnframes() / wav.getframerate())
        return httpx.Response(200, json={"text": "Фрагмент"})

    async def emit(event):
        events.append(event)

    settings = Settings(asr_provider="openai-compatible", asr_language="auto")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = HttpASR(settings, emit, client=client)
        await adapter.start(16000)
        await feed(adapter, pcm(17.123))
        await adapter.finish()
        await adapter.close()
    assert durations == [15, 2.123]
    assert len(events) == 2


@pytest.mark.asyncio
async def test_stop_drains_full_waiting_queue_before_final_short_tail():
    pieces = []

    async def handle(request):
        pieces.append(multipart(request)["file"])
        return httpx.Response(200, json={"text": ""})

    async def emit(event):
        assert event["type"] != "error"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = HttpASR(Settings(asr_chunk_seconds=1), emit, client=client)
        await adapter.start(16000)
        await feed(adapter, pcm(4, 0) + pcm(0.1))
        await asyncio.wait_for(adapter.finish(), 2)
        await adapter.close()
    assert len(pieces) == 5


@pytest.mark.asyncio
async def test_overload_is_visible_and_bounded():
    events = []

    async def emit(event):
        events.append(event)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"text": ""}))) as client:
        adapter = HttpASR(Settings(asr_chunk_seconds=1), emit, client=client)
        await adapter.start(16000)
        with pytest.raises(ProviderError, match="очередь"):
            await feed(adapter, pcm(5, 0))
        assert events[0]["code"] == "asr_overloaded"
        assert adapter._queue.qsize() == 4
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(503, text="private server details"), httpx.Response(307, json={"text": "must not accept redirects"}), httpx.Response(200, json={"result": "wrong"}), httpx.Response(200, text="not-json")])
async def test_provider_failure_wakes_finish_and_never_exposes_payload(response):
    events = []

    async def emit(event):
        events.append(event)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        adapter = HttpASR(Settings(), emit, client=client)
        await adapter.start(16000)
        await feed(adapter, pcm(0.1))
        with pytest.raises(ProviderError) as error:
            await asyncio.wait_for(adapter.finish(), 2)
        assert "private server details" not in str(error.value)
        assert events[0]["type"] == "error"
        await adapter.close()


@pytest.mark.asyncio
async def test_timeout_and_lifecycle_errors_are_explicit():
    async def handle(request):
        raise httpx.ReadTimeout("private timeout detail", request=request)

    async def emit(event):
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = HttpASR(Settings(), emit, client=client)
        with pytest.raises(ProviderError):
            await adapter.send_audio(b"\0\0")
        await adapter.start(16000)
        with pytest.raises(ProviderError):
            await adapter.send_audio(b"\0")
        await adapter.send_audio(b"\0\0")
        with pytest.raises(ProviderError) as error:
            await adapter.finish()
        assert error.value.code == "asr_timeout"
        await adapter.close()


def test_provider_factory_selects_http_adapter_without_network():
    async def emit(event):
        pass

    assert isinstance(create_asr(Settings(asr_provider="openai-compatible"), emit), HttpASR)
