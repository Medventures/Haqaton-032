from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
from types import SimpleNamespace
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_REQUEST_BYTES, RequestBodyLimitMiddleware, Settings, create_app


def wav(samples=1600, channels=1, rate=16000):
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x00" * samples * channels)
    return output.getvalue()


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append((source.read(), kwargs))
        return iter([SimpleNamespace(text=" Жалобы: "), SimpleNamespace(text="болит голова ")]), SimpleNamespace(language="ru")


def test_health_does_not_load_and_warmup_only_loads_once():
    loaded = []
    model = FakeModel()
    app = create_app(Settings(), lambda settings: loaded.append(settings.model) or model)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "modelLoaded": False, "engine": "whisper", "device": "cpu", "model": "small", "revision": None,
        }
        assert loaded == []
        assert client.post("/v1/models/load").json()["modelLoaded"] is True
        assert client.post("/v1/models/load").status_code == 200
        assert loaded == ["small"]
        assert client.get("/health").json()["modelLoaded"] is True


def test_transcription_contract_ru_kk_auto_and_lazy_generator():
    model = FakeModel()
    with TestClient(create_app(Settings(), lambda settings: model)) as client:
        for language in ["ru", "kk", "auto"]:
            response = client.post("/v1/audio/transcriptions", data={"model": "whisper-1", "language": language, "response_format": "json"}, files={"file": ("test.wav", wav(), "audio/wav")})
            assert response.status_code == 200, response.text
            assert response.json() == {"text": "Жалобы: болит голова"}
            assert model.calls[-1][1]["language"] == (None if language == "auto" else language)
            assert model.calls[-1][1]["vad_filter"] is True
            assert model.calls[-1][1]["condition_on_previous_text"] is False
            assert model.calls[-1][1]["beam_size"] == 1
        response = client.post("/v1/audio/transcriptions", files={"file": ("test.wav", wav(), "audio/wav")})
        assert response.status_code == 200
        assert model.calls[-1][1]["language"] is None


def test_beam_size_can_be_configured_for_quality_and_cpu_tradeoff(monkeypatch):
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "5")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    settings = Settings.from_env()
    assert settings.beam_size == 5
    assert settings.model == "large-v3"
    model = FakeModel()
    with TestClient(create_app(settings, lambda settings: model)) as client:
        response = client.post("/v1/audio/transcriptions", data={"language": "ru"}, files={"file": ("test.wav", wav(), "audio/wav")})
        assert response.status_code == 200
        assert model.calls[-1][1]["beam_size"] == 5
        assert model.calls[-1][1]["language"] == "ru"


def test_reject_invalid_file_format_model_and_duration_before_loading():
    loaded = []
    with TestClient(create_app(Settings(), lambda settings: loaded.append(True))) as client:
        for audio in [b"not wav", wav(channels=2), wav(rate=48000), wav(samples=0), wav(samples=16000 * 31), wav()[:-10]]:
            response = client.post("/v1/audio/transcriptions", files={"file": ("test.wav", audio)})
            assert response.status_code in {413, 422}, response.text
        assert client.post("/v1/audio/transcriptions", data={"model": "unknown"}, files={"file": ("test.wav", wav())}).status_code == 400
        assert client.post("/v1/audio/transcriptions", data={"language": "Kazakh language"}, files={"file": ("test.wav", wav())}).status_code == 422
        assert loaded == []


def test_model_failure_does_not_expose_exception_payload():
    def fail(settings):
        raise RuntimeError("private-path-or-token")

    with TestClient(create_app(Settings(), fail)) as client:
        response = client.post("/v1/models/load")
        assert response.status_code == 503
        assert "private-path-or-token" not in response.text


def test_inference_serialization_rejects_overlap_but_health_remains_responsive():
    entered, release = threading.Event(), threading.Event()

    class SlowModel(FakeModel):
        def transcribe(self, source, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().transcribe(source, **kwargs)

    with TestClient(create_app(Settings(), lambda settings: SlowModel())) as client, ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.post, "/v1/audio/transcriptions", files={"file": ("test.wav", wav(), "audio/wav")})
        try:
            assert entered.wait(2)
            assert client.get("/health").status_code == 200
            assert client.post("/v1/models/load").status_code == 429
            assert client.post("/v1/audio/transcriptions", files={"file": ("test.wav", wav(), "audio/wav")}).status_code == 429
        finally:
            release.set()
        assert running.result(timeout=2).status_code == 200


def test_oversized_upload_rejected_before_multipart_parsing():
    with TestClient(create_app(Settings(), lambda settings: None)) as client:
        response = client.post(
            "/v1/audio/transcriptions", content=b"x" * (MAX_REQUEST_BYTES + 1),
            headers={"Content-Type": "multipart/form-data; boundary=not-a-valid-body"},
        )
        assert response.status_code == 413


def test_chunked_body_limit_runs_before_downstream_parser_without_content_length():
    async def run():
        sent, downstream = [], []
        chunks = iter([
            {"type": "http.request", "body": b"x" * 6, "more_body": True},
            {"type": "http.request", "body": b"y" * 6, "more_body": True},
        ])

        async def receive():
            return next(chunks)

        async def send(event):
            sent.append(event)

        async def parser(scope, receive, send):
            downstream.append(True)

        await RequestBodyLimitMiddleware(parser, max_bytes=10)(
            {"type": "http", "method": "POST", "headers": [(b"transfer-encoding", b"chunked")]}, receive, send,
        )
        assert downstream == []
        assert sent[0]["status"] == 413

    asyncio.run(run())


def test_gigaam_metadata_aliases_and_25_second_limit_before_model_load():
    settings = Settings(engine="gigaam", gigaam_device="cpu")
    model = FakeModel()
    loaded = []
    with TestClient(create_app(settings, lambda selected: loaded.append(selected.engine) or model)) as client:
        assert client.get("/health").json() == {
            "status": "ok", "modelLoaded": False, "engine": "gigaam", "device": "cpu",
            "model": "ai-sage/GigaAM-Multilingual", "revision": "3905cd51c3ed4e88c8edf33f3302969ba480a327",
        }
        rejected = client.post("/v1/audio/transcriptions", files={"file": ("long.wav", wav(samples=25 * 16000 + 1))})
        assert rejected.status_code == 422
        assert "25" in rejected.text
        assert loaded == []
        for alias in ("asr-default", "whisper-1", settings.gigaam_model):
            response = client.post("/v1/audio/transcriptions", data={"model": alias, "language": "kk"}, files={"file": ("valid.wav", wav(samples=25 * 16000))})
            assert response.status_code == 200, response.text
            assert response.json() == {"text": "Жалобы: болит голова"}
        assert loaded == ["gigaam"]
        assert client.post("/v1/audio/transcriptions", data={"model": "small"}, files={"file": ("valid.wav", wav())}).status_code == 400
        assert client.post("/v1/models/load").json() == {"engine": "gigaam", "model": settings.gigaam_model, "modelLoaded": True}


def test_engine_settings_keep_whisper_defaults_and_require_pinned_gigaam(monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "gigaam")
    monkeypatch.setenv("GIGAAM_DEVICE", "cuda:0")
    monkeypatch.setenv("GIGAAM_MODEL", "ai-sage/GigaAM-Multilingual")
    monkeypatch.setenv("GIGAAM_REVISION", "3905cd51c3ed4e88c8edf33f3302969ba480a327")
    settings = Settings.from_env()
    assert settings.engine == "gigaam"
    assert settings.effective_device == "cuda:0"
    assert settings.max_audio_seconds == 25
    with pytest.raises(ValueError, match="immutable"):
        Settings(engine="gigaam", gigaam_revision="main")
    with pytest.raises(ValueError, match="ASR_ENGINE"):
        Settings(engine="unsupported")
