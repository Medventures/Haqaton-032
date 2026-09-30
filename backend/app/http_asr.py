"""Turn browser PCM into ordered WAV requests for an HTTP transcription API.

This is chunked transcription, not a streaming Whisper decoder. Prefer cuts
after silence once the target duration is reached; force a cut at 15 seconds.
The maximum cut may split a word. Four waiting chunks bound memory and latency;
an overloaded recognizer fails visibly rather than dropping accepted audio.
"""

from __future__ import annotations

import asyncio
from array import array
from contextlib import suppress
from io import BytesIO
import sys
from typing import Any
from uuid import uuid4
import wave

import httpx

from .providers import EventHandler, ProviderError, Settings


class HttpASR:
    def __init__(self, settings: Settings, on_event: EventHandler, *, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.on_event = on_event
        self._client = client
        self._owns_client = client is None
        self._worker: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[tuple[bytes, int, int] | None] = asyncio.Queue(maxsize=4)
        self._buffer = bytearray()
        self._remainder = bytearray()
        self._sample_rate = 16000
        self._offset = 0
        self._silence_samples = 0
        self._counter = 0
        self._epoch = uuid4().hex[:12]
        self._ending = False
        self._closed = False
        self._error: ProviderError | None = None

    def _raise_if_failed(self) -> None:
        if self._error:
            raise self._error

    async def start(self, sample_rate: int) -> None:
        if self._worker is not None or self._closed:
            raise ProviderError("asr_already_started", "Распознавание уже запущено или закрыто.")
        if not 8000 <= sample_rate <= 48000:
            raise ProviderError("audio_format", "Недопустимая частота PCM-аудио.")
        self._sample_rate = sample_rate
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.provider_timeout_seconds)
        self._worker = asyncio.create_task(self._consume())

    async def send_audio(self, pcm: bytes) -> None:
        self._raise_if_failed()
        if self._worker is None or self._ending or self._closed:
            raise ProviderError("asr_not_started", "Аудиопоток распознавания не активен.")
        if len(pcm) % 2:
            raise ProviderError("audio_format", "PCM16 должен содержать целое число 16-битных samples.")
        if len(pcm) > self._sample_rate * 2:
            raise ProviderError("audio_format", "Один пакет PCM не должен превышать одну секунду.")
        self._remainder.extend(pcm)
        frame_bytes = self._sample_rate // 50 * 2  # 20 ms frames, independent of browser packet sizes.
        target_samples = int(self.settings.asr_chunk_seconds * self._sample_rate)
        while len(self._remainder) >= frame_bytes:
            frame = bytes(self._remainder[:frame_bytes])
            del self._remainder[:frame_bytes]
            self._buffer.extend(frame)
            samples = array("h")
            samples.frombytes(frame)
            if sys.byteorder != "little":
                samples.byteswap()
            mean_square = sum(value * value for value in samples) / len(samples)
            self._silence_samples = self._silence_samples + len(samples) if mean_square < 150 ** 2 else 0
            buffered_samples = len(self._buffer) // 2
            silence_cut = buffered_samples >= target_samples and self._silence_samples >= self._sample_rate * 0.4
            if silence_cut or buffered_samples >= self._sample_rate * 15:
                await self._enqueue_buffer()

    async def _record_error(self, error: ProviderError) -> None:
        if self._error is None:
            self._error = error
            await self.on_event({"type": "error", "code": error.code, "message": error.message})

    async def _enqueue_buffer(self) -> None:
        self._raise_if_failed()
        if not self._buffer:
            return
        pcm = bytes(self._buffer)
        start = self._offset
        end = start + len(pcm) // 2
        try:
            self._queue.put_nowait((pcm, start, end))
        except asyncio.QueueFull:
            error = ProviderError("asr_overloaded", "Сервер распознавания не успевает за записью: очередь из четырёх фрагментов заполнена. Остановите запись или используйте более быстрый сервер.")
            await self._record_error(error)
            raise error from None
        self._buffer.clear()
        self._silence_samples = 0
        self._offset = end

    def _wav(self, pcm: bytes) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(pcm)
        return output.getvalue()

    async def _transcribe(self, pcm: bytes) -> str:
        assert self._client is not None
        fields = {"model": self.settings.asr_model, "response_format": "json"}
        language = self.settings.asr_language.strip()
        if language and language.lower() != "auto":
            fields["language"] = language
        headers = {"Authorization": f"Bearer {self.settings.asr_api_key}"} if self.settings.asr_api_key else {}
        try:
            response = await self._client.post(
                self.settings.asr_base_url.rstrip("/") + "/audio/transcriptions",
                data=fields, files={"file": ("chunk.wav", self._wav(pcm), "audio/wav")}, headers=headers,
                timeout=self.settings.provider_timeout_seconds,
            )
        except httpx.TimeoutException:
            raise ProviderError("asr_timeout", "Сервер распознавания не ответил вовремя. Для локальной модели сначала выполните её прогрев и проверьте скорость обработки.", retryable=True) from None
        except httpx.HTTPError:
            raise ProviderError("asr_unreachable", "Не удалось обратиться к HTTP-серверу распознавания. Проверьте ASR_BASE_URL и сеть.", retryable=True) from None
        if not 200 <= response.status_code < 300:
            raise ProviderError("asr_http_error", f"HTTP-сервер распознавания вернул ошибку {response.status_code}. Проверьте модель, ключ и доступность сервиса.", retryable=response.status_code in {429, 502, 503, 504})
        try:
            result = response.json()
        except ValueError:
            raise ProviderError("asr_invalid_response", "Сервер распознавания вернул некорректный JSON.") from None
        if not isinstance(result, dict) or not isinstance(result.get("text"), str) or len(result["text"]) > 20000:
            raise ProviderError("asr_invalid_response", "Ответ распознавания должен содержать строку text длиной до 20000 символов.")
        return result["text"].strip()

    async def _consume(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    if item is None:
                        return
                    pcm, start, end = item
                    text = await self._transcribe(pcm)
                    self._counter += 1
                    if text:
                        await self.on_event({
                            "type": "final", "id": f"http_{self._epoch}_{self._counter}", "revision": 1,
                            "text": text, "startMs": round(start * 1000 / self._sample_rate),
                            "endMs": round(end * 1000 / self._sample_rate), "speaker": None,
                        })
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise
        except ProviderError as error:
            await self._record_error(error)
        except Exception:
            await self._record_error(ProviderError("asr_internal_error", "Ошибка обработки ответа сервера распознавания."))

    async def finish(self) -> None:
        self._raise_if_failed()
        if self._worker is None or self._closed:
            raise ProviderError("asr_not_started", "Распознавание не запущено.")
        if self._ending:
            await self._worker
            self._raise_if_failed()
            return
        self._ending = True
        self._buffer.extend(self._remainder)
        self._remainder.clear()
        # Once recording stops, wait for pending chunks rather than rejecting the
        # final short tail just because the waiting queue is currently full.
        drained = asyncio.create_task(self._queue.join())
        try:
            done, _ = await asyncio.wait({drained, self._worker}, return_when=asyncio.FIRST_COMPLETED)
            if self._worker in done:
                self._raise_if_failed()
            await drained
        finally:
            if not drained.done():
                drained.cancel()
                with suppress(asyncio.CancelledError):
                    await drained
        await self._enqueue_buffer()
        # Do not require a spare queue slot for the sentinel. A failed worker must
        # also wake finalization instead of leaving it blocked on queue.put.
        sentinel = asyncio.create_task(self._queue.put(None))
        try:
            done, _ = await asyncio.wait({sentinel, self._worker}, return_when=asyncio.FIRST_COMPLETED)
            if self._worker in done:
                self._raise_if_failed()
            await sentinel
            await self._worker
            self._raise_if_failed()
        finally:
            if not sentinel.done():
                sentinel.cancel()
                with suppress(asyncio.CancelledError):
                    await sentinel

    async def close(self) -> None:
        self._closed = True
        if self._worker and not self._worker.done():
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        if self._client and self._owns_client:
            await self._client.aclose()
        self._buffer.clear()
        self._remainder.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
