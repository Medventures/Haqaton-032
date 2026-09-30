"""Small OpenAI-style WAV transcription endpoint with selectable local engines.

Run one Uvicorn worker. The model is loaded on the first transcription or an
explicit warmup request. Health checks never import faster_whisper or download.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import os
import re
import threading
from typing import Annotated, Any, Callable
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .rukk import DEFAULT_MODEL_SHA256, DEFAULT_TOKENS_SHA256, MODEL_NAME as RUKK_MODEL_NAME, RukkError


MAX_UPLOAD_BYTES = 1_048_576
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 65_536  # Bounded multipart headers and form fields.


class RequestBodyLimitMiddleware:
    """Bound even chunked bodies before the multipart parser can spool uploads."""

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        lengths = [value for name, value in scope.get("headers", []) if name.lower() == b"content-length"]
        if lengths and (len(lengths) != 1 or not lengths[0].isdigit() or int(lengths[0]) > self.max_bytes):
            return await JSONResponse({"detail": "Размер запроса превышает допустимый лимит"}, status_code=413)(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                return await JSONResponse({"detail": "Размер запроса превышает допустимый лимит"}, status_code=413)(scope, receive, send)
            body.extend(chunk)
            if not event.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


@dataclass(frozen=True)
class Settings:
    engine: str = "whisper"
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 1
    download_root: str = "/models"
    gigaam_model: str = "ai-sage/GigaAM-Multilingual"
    gigaam_revision: str = "3905cd51c3ed4e88c8edf33f3302969ba480a327"
    gigaam_device: str = "cuda"
    rukk_model_path: str = "/models/rukk/model.pt"
    rukk_tokens_path: str = "/models/rukk/tokens.lst"
    rukk_device: str = "cpu"
    rukk_threads: int = 6
    rukk_model_sha256: str = DEFAULT_MODEL_SHA256
    rukk_tokens_sha256: str = DEFAULT_TOKENS_SHA256

    def __post_init__(self) -> None:
        if self.engine not in {"whisper", "gigaam", "rukk"}:
            raise ValueError("ASR_ENGINE must be whisper, gigaam, or rukk")
        if not 1 <= self.beam_size <= 10:
            raise ValueError("WHISPER_BEAM_SIZE must be between 1 and 10")
        if not re.fullmatch(r"[a-f0-9]{40}", self.gigaam_revision):
            raise ValueError("GIGAAM_REVISION must be an immutable 40-character commit SHA")
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.gigaam_device):
            raise ValueError("GIGAAM_DEVICE must be cpu, cuda, or cuda:N")
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.rukk_device):
            raise ValueError("RUKK_DEVICE must be cpu, cuda, or cuda:N")
        if not 1 <= self.rukk_threads <= 64:
            raise ValueError("RUKK_CPU_THREADS must be between 1 and 64")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.rukk_model_sha256):
            raise ValueError("RUKK_MODEL_SHA256 must be a 64-character SHA256 digest")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.rukk_tokens_sha256):
            raise ValueError("RUKK_TOKENS_SHA256 must be a 64-character SHA256 digest")
        if not self.rukk_model_path.strip() or not self.rukk_tokens_path.strip():
            raise ValueError("RUKK_MODEL_PATH and RUKK_TOKENS_PATH must not be empty")

    @property
    def effective_model(self) -> str:
        if self.engine == "rukk":
            return RUKK_MODEL_NAME
        return self.gigaam_model if self.engine == "gigaam" else self.model

    @property
    def effective_device(self) -> str:
        if self.engine == "rukk":
            return self.rukk_device
        return self.gigaam_device if self.engine == "gigaam" else self.device

    @property
    def effective_revision(self) -> str | None:
        if self.engine == "rukk":
            return self.rukk_model_sha256.lower()
        return self.gigaam_revision if self.engine == "gigaam" else None

    @property
    def max_audio_seconds(self) -> int:
        return 25 if self.engine in {"gigaam", "rukk"} else 30

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            engine=os.getenv("ASR_ENGINE", "whisper").strip().lower(),
            model=os.getenv("WHISPER_MODEL", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
            download_root=os.getenv("WHISPER_DOWNLOAD_ROOT", "/models"),
            gigaam_model=os.getenv("GIGAAM_MODEL", "ai-sage/GigaAM-Multilingual"),
            gigaam_revision=os.getenv("GIGAAM_REVISION", "3905cd51c3ed4e88c8edf33f3302969ba480a327"),
            gigaam_device=os.getenv("GIGAAM_DEVICE", "cuda"),
            rukk_model_path=os.getenv("RUKK_MODEL_PATH", "/models/rukk/model.pt"),
            rukk_tokens_path=os.getenv("RUKK_TOKENS_PATH", "/models/rukk/tokens.lst"),
            rukk_device=os.getenv("RUKK_DEVICE", "cpu"),
            rukk_threads=int(os.getenv("RUKK_CPU_THREADS", "6")),
            rukk_model_sha256=os.getenv("RUKK_MODEL_SHA256", "").strip().lower() or DEFAULT_MODEL_SHA256,
            rukk_tokens_sha256=os.getenv("RUKK_TOKENS_SHA256", "").strip().lower() or DEFAULT_TOKENS_SHA256,
        )


def load_model(settings: Settings):
    if settings.engine == "rukk":
        from .rukk import load_rukk

        return load_rukk(settings)
    if settings.engine == "gigaam":
        from .gigaam import load_gigaam

        return load_gigaam(settings)
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.model, device=settings.device, compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads, num_workers=1, download_root=settings.download_root,
    )


def validate_wav(audio: bytes, max_seconds: int = 30) -> None:
    try:
        with wave.open(BytesIO(audio), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000 or wav.getcomptype() != "NONE":
                raise HTTPException(422, "Ожидается WAV PCM16 mono 16000 Гц")
            frames = wav.getnframes()
            if not 1 <= frames <= 16000 * max_seconds:
                raise HTTPException(422, f"Продолжительность WAV должна быть от одного sample до {max_seconds} секунд")
            if len(wav.readframes(frames)) != frames * 2:
                raise HTTPException(422, "WAV содержит неполные аудиоданные")
    except (wave.Error, EOFError):
        raise HTTPException(422, "Не удалось прочитать WAV-файл") from None


class Recognizer:
    def __init__(self, settings: Settings, factory: Callable[[Settings], Any]):
        self.settings = settings
        self.factory = factory
        self.model: Any = None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is None:
            self.model = self.factory(self.settings)
        return self.model

    def warmup(self) -> None:
        with self.lock:
            self._load()

    def transcribe(self, audio: bytes, language: str | None) -> str:
        with self.lock:
            model = self._load()
            segments, _info = model.transcribe(
                BytesIO(audio), language=language, beam_size=self.settings.beam_size, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400}, condition_on_previous_text=False,
            )
            # faster-whisper returns a lazy generator; inference must stay inside
            # this worker thread and lock until the entire generator is consumed.
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


def create_app(settings: Settings | None = None, model_factory: Callable[[Settings], Any] = load_model) -> FastAPI:
    config = settings or Settings.from_env()
    recognizer = Recognizer(config, model_factory)
    gate = asyncio.Lock()
    application = FastAPI(title="Local ASR Transcriber", version="0.1.0")
    application.add_middleware(RequestBodyLimitMiddleware)
    application.state.recognizer = recognizer

    @application.get("/health")
    async def health():
        return {
            "status": "ok", "modelLoaded": recognizer.model is not None, "engine": config.engine,
            "device": config.effective_device, "model": config.effective_model, "revision": config.effective_revision,
        }

    @application.post("/v1/models/load")
    async def warmup():
        if gate.locked():
            raise HTTPException(429, "Модель занята; повторите позже")
        async with gate:
            try:
                await run_in_threadpool(recognizer.warmup)
            except RukkError as error:
                raise HTTPException(503, error.detail) from None
            except Exception:
                raise HTTPException(503, "Не удалось загрузить модель. Проверьте ASR_ENGINE, настройки модели, кэш, память и доступ к файлам модели.") from None
        return {"engine": config.engine, "model": config.effective_model, "modelLoaded": True}

    @application.post("/v1/audio/transcriptions")
    async def transcribe(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "asr-default",
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        if model not in {"asr-default", "whisper-1", config.effective_model}:
            raise HTTPException(400, "Модель не совпадает с настроенным ASR; допустимы aliases asr-default и whisper-1")
        if response_format != "json":
            raise HTTPException(400, "Поддерживается только response_format=json")
        selected_language = (language or "auto").strip().lower()
        if selected_language in {"", "auto"}:
            selected_language = None
        elif not re.fullmatch(r"[a-z]{2,3}", selected_language):
            raise HTTPException(422, "Укажите код языка, например ru, kk, или auto")
        if gate.locked():
            raise HTTPException(429, "Модель занята; повторите позже")
        try:
            audio = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        if len(audio) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Размер WAV превышает 1 МиБ")
        validate_wav(audio, max_seconds=config.max_audio_seconds)
        if gate.locked():
            raise HTTPException(429, "Модель занята; повторите позже")
        async with gate:
            try:
                text = await run_in_threadpool(recognizer.transcribe, audio, selected_language)
            except RukkError as error:
                raise HTTPException(503, error.detail) from None
            except Exception:
                raise HTTPException(503, "Распознавание не выполнено. Проверьте модель, ресурсы и поддерживаемый язык.") from None
        return {"text": text}

    return application


app = create_app()
