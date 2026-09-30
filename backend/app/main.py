"""REST commands + a numbered PCM WebSocket stream for a trusted prototype."""

import asyncio
import copy
import json
import os
import struct
from datetime import datetime, timezone
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import clinical, providers
from .schemas import CreateSession, EditField, InsertTranscript, InvalidForm, UnlockField, validate_form, validate_values
from .storage import Store


SAMPLE_RATE = 16_000
MAX_AUDIO_PACKET_BYTES = SAMPLE_RATE * 2  # At most one second per packet.
MAX_SESSION_AUDIO_BYTES = SAMPLE_RATE * 2 * 60 * 60  # One hour, bounded on disk.
MAX_TRANSCRIPT_CHARACTERS = 120_000
MAX_TRANSCRIPT_SEGMENTS = 2000
MAX_CACHED_SESSIONS = 64
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"

def default_form() -> dict[str, Any]:
    configured = os.getenv("DEFAULT_FORM_PATH")
    candidates = [Path(configured)] if configured else [
        Path(__file__).resolve().parents[2] / "contracts" / "default-form.schema.json",
        Path("/app/contracts/default-form.schema.json"),
    ]
    for path in candidates:
        if path.is_file():
            schema = json.loads(path.read_text(encoding="utf-8-sig"))
            validate_form(schema)
            return schema
    raise RuntimeError("contracts/default-form.schema.json not found")


class Session:
    def __init__(self, service: "Service", snapshot: dict[str, Any]):
        self.service = service
        self.snapshot = snapshot
        self.snapshot.setdefault("clinicalAssessment", None)
        self.snapshot.setdefault("clinicalStatus", "idle")
        self.snapshot.setdefault("clinicalError", None)
        self.snapshot.setdefault("clinicalRevision", 0)
        self.lock = asyncio.Lock()
        self.operation_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.websocket: WebSocket | None = None
        self.asr: Any = None
        self.stream_id: str | None = None
        self.through_seq = 0
        self.next_sample_offset = 0
        self.audio_bytes = 0
        self.dirty = False
        self.extraction_task: asyncio.Task | None = None
        self.clinical_task: asyncio.Task | None = None
        self.disconnect_task: asyncio.Task | None = None
        self.disconnect_finalizing = False
        self.last_extracted_revision = 0
        self.last_extraction_error: str | None = None

    def save(self) -> None:
        self.service.store.save(self.snapshot)

    async def send(self, event_type: str, payload: Any, websocket: WebSocket | None = None) -> None:
        async with self.send_lock:
            target = websocket or self.websocket
            if target is None:
                return
            with suppress(WebSocketDisconnect, RuntimeError, OSError):
                await target.send_json({"type": event_type, "payload": payload})

    async def publish(self) -> None:
        async with self.lock:
            snapshot = copy.deepcopy(self.snapshot)
        await self.send("session.snapshot", snapshot)

    async def fail(self, message: str, fatal: bool = True) -> None:
        async with self.lock:
            self.snapshot["error"] = message
            if fatal:
                self.snapshot["status"] = "error"
            self.save()
        await self.publish()

    def schedule_extraction(self) -> None:
        self.dirty = True
        if self.extraction_task is None or self.extraction_task.done():
            self.extraction_task = asyncio.create_task(self.extract_loop())

    async def on_asr_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "partial":
            await self.send("transcript.partial", {"text": str(event.get("text", ""))[:8000]})
            return
        if event["type"] == "error":
            await self.fail(str(event.get("message", "Ошибка сервиса распознавания")))
            return
        if event["type"] != "final":
            return
        text = str(event.get("text", "")).strip()
        if not text:
            return
        segment = {
            "id": str(event["id"]), "revision": int(event.get("revision", 1)), "text": text,
            "startMs": int(event.get("startMs", 0)), "endMs": int(event.get("endMs", 0)),
            "speaker": event.get("speaker"),
        }
        exceeded = False
        async with self.lock:
            transcript = self.snapshot["transcript"]
            existing = next((item for item in transcript if item["id"] == segment["id"]), None)
            if existing and existing["revision"] >= segment["revision"]:
                return
            total_chars = sum(len(item["text"]) for item in transcript) + len(text) - (len(existing["text"]) if existing else 0)
            if total_chars > MAX_TRANSCRIPT_CHARACTERS or (existing is None and len(transcript) >= MAX_TRANSCRIPT_SEGMENTS):
                exceeded = True
            else:
                if existing:
                    transcript[transcript.index(existing)] = segment
                else:
                    transcript.append(segment)
                self.snapshot["transcriptRevision"] += 1
                self.save()
                self.schedule_extraction()
        if exceeded:
            await self.fail("Достигнут лимит расшифровки. Остановите запись и создайте новую сессию.")
            return
        await self.send("transcript.partial", {"text": ""})
        await self.publish()

    async def extract_loop(self) -> None:
        while self.dirty:
            await asyncio.sleep(self.service.extraction_delay)
            async with self.lock:
                self.dirty = False
                revision = self.snapshot["transcriptRevision"]
                segments = copy.deepcopy(self.snapshot["transcript"])
                values = copy.deepcopy(self.snapshot["values"])
                schema = self.snapshot["formSchema"]
            if not segments:
                continue
            try:
                candidate = await providers.extract(self.service.settings, schema, segments, values)
                validate_values(schema, candidate)
            except asyncio.CancelledError:
                raise
            except (providers.ProviderError, InvalidForm) as exc:
                self.last_extraction_error = str(exc)
                await self.fail(f"Не удалось заполнить форму: {exc}", fatal=False)
                continue
            except Exception:
                self.last_extraction_error = "Внутренняя ошибка заполнения"
                await self.fail(self.last_extraction_error, fatal=False)
                continue
            async with self.lock:
                # A correction invalidates the evidence used by this candidate. Newly
                # appended segments alone are safe: one sequential worker catches up.
                current_revisions = {item["id"]: item["revision"] for item in self.snapshot["transcript"]}
                if any(current_revisions.get(item["id"]) != item["revision"] for item in segments):
                    self.dirty = True
                    continue
                changed = False
                for key, value in candidate.items():
                    meta = self.snapshot["fieldMeta"][key]
                    current = self.snapshot["values"][key]
                    if meta["locked"]:
                        if value != current:
                            if "suggestion" not in meta or meta["suggestion"] != value:
                                meta["suggestion"] = value
                                meta["revision"] += 1
                                changed = True
                        elif "suggestion" in meta:
                            del meta["suggestion"]
                            meta["revision"] += 1
                            changed = True
                    elif current != value:
                        self.snapshot["values"][key] = value
                        meta["source"] = "llm" if value is not None else "empty"
                        meta["revision"] += 1
                        meta.pop("suggestion", None)
                        changed = True
                if changed:
                    self.snapshot["documentRevision"] += 1
                self.last_extracted_revision = revision
                if self.last_extraction_error and self.snapshot["error"] in {
                    self.last_extraction_error, f"Не удалось заполнить форму: {self.last_extraction_error}"
                }:
                    self.snapshot["error"] = None
                self.last_extraction_error = None
                self.save()
            await self.publish()

    async def start_stream(self, message: dict[str, Any]) -> None:
        audio = message.get("audio")
        if audio != {"encoding": "pcm_s16le", "sampleRate": SAMPLE_RATE, "channels": 1}:
            raise HTTPException(422, "Поддерживается PCM signed 16-bit little-endian, 16000 Гц, mono")
        async with self.operation_lock:
            if message["type"] == "stream.resume":
                if not self.asr or message.get("streamId") != self.stream_id or self.snapshot["status"] != "recording":
                    raise HTTPException(409, "Поток нельзя восстановить. Получите состояние сессии и создайте новую запись.")
            elif not (
                self.asr is not None
                and self.stream_id is not None
                and self.snapshot["status"] == "recording"
                and self.through_seq == 0
            ):
                # Repeating start before any audio is accepted returns the existing
                # stream.ready. This recovers a lost first ready response safely.
                if self.snapshot["status"] != "ready" or self.asr is not None:
                    raise HTTPException(409, "Эта сессия уже запущена; используйте stream.resume или новую сессию")
                self.stream_id = str(uuid4())
                self.asr = providers.create_asr(self.service.settings, self.on_asr_event)
                async with self.lock:
                    self.snapshot["status"] = "recording"
                    self.save()
                try:
                    await self.asr.start(SAMPLE_RATE)
                except Exception as exc:
                    message = str(exc) if isinstance(exc, providers.ProviderError) else "Не удалось подключить сервис распознавания. Проверьте настройки провайдера."
                    await self.fail(message)
                    with suppress(Exception):
                        await self.asr.close()
                    self.asr = None
                    raise HTTPException(502, message) from exc
            await self.send("stream.ready", {
                "streamId": self.stream_id, "throughSeq": self.through_seq,
                "nextSampleOffset": self.next_sample_offset,
            })
        await self.publish()

    async def accept_audio(self, packet: bytes) -> None:
        if len(packet) < 10 or len(packet) > MAX_AUDIO_PACKET_BYTES + 8 or (len(packet) - 8) % 2:
            raise HTTPException(422, "Аудиопакет должен содержать 8-байтовый заголовок и 1–16000 PCM16 samples")
        seq, offset = struct.unpack("<II", packet[:8])
        pcm = packet[8:]
        async with self.operation_lock:
            if self.snapshot["status"] != "recording" or self.asr is None or self.stream_id is None:
                raise HTTPException(409, "Сначала начните аудиопоток")
            if seq <= self.through_seq:
                if not self.service.store.matches_audio(self.snapshot["id"], self.stream_id, seq, offset, pcm):
                    raise HTTPException(409, "Повторный пакет отличается от сохранённого")
            else:
                if seq != self.through_seq + 1 or offset != self.next_sample_offset:
                    raise HTTPException(409, f"Ожидается пакет {self.through_seq + 1}, sampleOffset {self.next_sample_offset}")
                if self.audio_bytes + len(pcm) > MAX_SESSION_AUDIO_BYTES:
                    raise HTTPException(413, "Достигнут лимит записи в один час")
                self.service.store.append_audio(self.snapshot["id"], self.stream_id, seq, offset, pcm)
                self.through_seq = seq
                self.next_sample_offset += len(pcm) // 2
                self.audio_bytes += len(pcm)
                # Ack means persisted audio, not completed recognition or extraction.
                await self.send("audio.ack", {"streamId": self.stream_id, "throughSeq": self.through_seq})
                try:
                    await self.asr.send_audio(pcm)
                except Exception as exc:
                    await self.fail(str(exc) if isinstance(exc, providers.ProviderError) else "Поток распознавания прерван. Принятое аудио сохранено, завершение расшифровки не гарантировано.")
                    raise HTTPException(502, "Ошибка отправки аудио в сервис распознавания")
                return
            await self.send("audio.ack", {"streamId": self.stream_id, "throughSeq": self.through_seq})

    async def edit(self, field_id: str, value: str | None, expected_revision: int) -> dict[str, Any]:
        async with self.lock:
            meta = self.snapshot["fieldMeta"].get(field_id)
            if meta is None:
                raise HTTPException(404, "Поле не найдено")
            if meta["revision"] != expected_revision:
                raise HTTPException(409, {"message": "Поле изменилось, обновите состояние", "revision": meta["revision"]})
            candidate = {**self.snapshot["values"], field_id: value}
            try:
                validate_values(self.snapshot["formSchema"], candidate)
            except InvalidForm as exc:
                raise HTTPException(422, str(exc)) from exc
            self.snapshot["values"][field_id] = value
            meta.update({"revision": meta["revision"] + 1, "source": "doctor", "locked": True})
            meta.pop("suggestion", None)
            self.snapshot["documentRevision"] += 1
            self.save()
            result = copy.deepcopy(self.snapshot)
        await self.publish()
        return result

    async def unlock(self, field_id: str, expected_revision: int) -> dict[str, Any]:
        async with self.lock:
            meta = self.snapshot["fieldMeta"].get(field_id)
            if meta is None:
                raise HTTPException(404, "Поле не найдено")
            if meta["revision"] != expected_revision:
                raise HTTPException(409, {"message": "Поле изменилось, обновите состояние", "revision": meta["revision"]})
            meta["locked"] = False
            meta["revision"] += 1
            meta.pop("suggestion", None)
            self.snapshot["documentRevision"] += 1
            self.save()
            self.schedule_extraction()
            result = copy.deepcopy(self.snapshot)
        await self.publish()
        return result

    async def insert_transcript(self, text: str, speaker: str | None) -> dict[str, Any]:
        if self.service.settings.effective_asr_provider != "demo":
            raise HTTPException(409, "Текстовые реплики доступны только при ASR_PROVIDER=demo")
        if not text.strip():
            raise HTTPException(422, "Введите текст разговора")
        async with self.operation_lock:
            if self.snapshot["status"] not in {"ready", "stopped"}:
                raise HTTPException(409, "Дождитесь завершения текущей обработки или создайте новую сессию")
            async with self.lock:
                self.snapshot["status"] = "processing"
                self.save()
                timestamp = self.snapshot["transcript"][-1]["endMs"] if self.snapshot["transcript"] else 0
            await self.publish()
            await self.on_asr_event({
                "type": "final", "id": f"text_{uuid4().hex}", "revision": 1, "text": text,
                "startMs": timestamp, "endMs": timestamp, "speaker": speaker,
            })
        return await self.stop()

    def start_assessment(self) -> asyncio.Task:
        if self.service.settings.effective_llm_provider == "demo":
            raise HTTPException(409, "Для клинических гипотез подключите LLM, например локальную модель через Ollama.")
        if self.snapshot["status"] not in {"ready", "stopped"} or self.dirty or (
            self.extraction_task and not self.extraction_task.done()
        ):
            raise HTTPException(409, "Дождитесь завершения записи и заполнения формы")
        if self.clinical_task and not self.clinical_task.done():
            raise HTTPException(409, "Клинический анализ уже выполняется")
        if not self.snapshot["transcript"] and not any(
            value and value.strip() for value in self.snapshot["values"].values()
        ):
            raise HTTPException(422, "Добавьте сведения о пациенте или текст консультации")
        self.clinical_task = asyncio.create_task(self.assess_clinical())
        return self.clinical_task

    async def assess_clinical(self) -> dict[str, Any]:
        async with self.lock:
            transcript_revision = self.snapshot["transcriptRevision"]
            document_revision = self.snapshot["documentRevision"]
            segments = copy.deepcopy(self.snapshot["transcript"])
            values = copy.deepcopy(self.snapshot["values"])
            schema = copy.deepcopy(self.snapshot["formSchema"])
            self.snapshot["clinicalStatus"] = "processing"
            self.snapshot["clinicalError"] = None
            self.snapshot["clinicalRevision"] += 1
            self.save()
        await self.publish()
        candidate = None
        error = None
        try:
            candidate = await clinical.assess(self.service.settings, segments, values, schema)
        except asyncio.CancelledError:
            raise
        except (providers.ProviderError, InvalidForm) as exc:
            error = str(exc)
        except Exception:
            error = "Не удалось выполнить клинический анализ. Повторите попытку."
        async with self.lock:
            if transcript_revision != self.snapshot["transcriptRevision"] or document_revision != self.snapshot["documentRevision"]:
                error = "Данные консультации изменились во время анализа. Запустите анализ повторно."
            if error is None:
                self.snapshot["clinicalAssessment"] = {
                    "values": candidate,
                    "provider": self.service.settings.effective_llm_provider,
                    "model": self.service.settings.effective_llm_model,
                    "transcriptRevision": transcript_revision,
                    "documentRevision": document_revision,
                    "generatedAt": datetime.now(timezone.utc).isoformat(),
                }
            self.snapshot["clinicalStatus"] = "error" if error else "ready"
            self.snapshot["clinicalError"] = error
            self.snapshot["clinicalRevision"] += 1
            self.save()
            result = copy.deepcopy(self.snapshot)
        await self.publish()
        return result

    async def stop(self) -> dict[str, Any]:
        async with self.operation_lock:
            if self.snapshot["status"] == "stopped":
                return copy.deepcopy(self.snapshot)
            if self.disconnect_task and self.disconnect_task is not asyncio.current_task() and not self.disconnect_finalizing:
                self.disconnect_task.cancel()
            was_error = self.snapshot["status"] == "error"
            async with self.lock:
                self.snapshot["status"] = "processing"
                self.save()
            await self.publish()
            if self.asr:
                try:
                    await self.asr.finish()
                except Exception:
                    was_error = True
                    await self.fail("Сервис распознавания не подтвердил завершение потока. Расшифровка может быть неполной.")
                finally:
                    with suppress(Exception):
                        await self.asr.close()
                    self.asr = None
            if self.extraction_task:
                await self.extraction_task
            async with self.lock:
                incomplete = self.last_extracted_revision < self.snapshot["transcriptRevision"]
                self.snapshot["status"] = "error" if was_error or incomplete or self.snapshot["status"] == "error" else "stopped"
                if incomplete and not self.snapshot["error"]:
                    self.snapshot["error"] = "Последние реплики не обработаны моделью; проверьте расшифровку и форму."
                self.save()
                result = copy.deepcopy(self.snapshot)
            await self.send("transcript.partial", {"text": ""})
            await self.publish()
            if result["status"] == "stopped" and self.service.settings.effective_llm_provider != "demo":
                # Background analysis keeps finalizing the audio independent of LLM latency.
                with suppress(HTTPException):
                    self.start_assessment()
            return result

    async def disconnect_timeout(self) -> None:
        await asyncio.sleep(self.service.reconnect_timeout)
        if self.websocket is None and self.asr is not None:
            self.disconnect_finalizing = True
            try:
                await self.stop()
                await self.fail("Соединение с браузером не восстановилось за 60 секунд. Запись завершена; проверьте сохранённый черновик.", fatal=False)
            finally:
                self.disconnect_finalizing = False

    async def close(self) -> None:
        tasks = [self.extraction_task, self.disconnect_task, self.clinical_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with suppress(asyncio.CancelledError, Exception):
                    await task
        if self.asr:
            with suppress(Exception):
                await self.asr.close()
        if self.snapshot["status"] in {"recording", "processing"}:
            self.snapshot["status"] = "error"
            self.snapshot["error"] = "Сервер остановлен во время записи. Сохранённый черновик доступен; создайте новую сессию."
            self.save()


class Service:
    def __init__(self, settings: providers.Settings, data_dir: Path, extraction_delay: float):
        self.settings = settings
        self.store = Store(data_dir)
        self.sessions: dict[str, Session] = {}
        self.extraction_delay = extraction_delay
        self.reconnect_timeout = 60.0

    def ensure_cache_space(self) -> None:
        if len(self.sessions) < MAX_CACHED_SESSIONS:
            return
        for key, session in self.sessions.items():
            background = (session.extraction_task, session.disconnect_task, session.clinical_task)
            if session.websocket is None and session.asr is None and all(task is None or task.done() for task in background):
                del self.sessions[key]
                return
        raise HTTPException(429, "Достигнут лимит активных сессий. Завершите одну из записей.")

    def get(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            snapshot = self.store.get(session_id)
            if snapshot is None:
                raise HTTPException(404, "Сессия не найдена")
            self.ensure_cache_space()
            self.sessions[session_id] = Session(self, snapshot)
        return self.sessions[session_id]

    def create(self, schema: dict[str, Any]) -> Session:
        try:
            validate_form(schema)
        except InvalidForm as exc:
            raise HTTPException(422, str(exc)) from exc
        self.ensure_cache_space()
        snapshot = {
            "id": str(uuid4()), "formSchema": schema,
            "values": {key: None for key in schema["properties"]},
            "fieldMeta": {key: {"revision": 0, "source": "empty", "locked": False} for key in schema["properties"]},
            "transcript": [], "status": "ready", "documentRevision": 0, "transcriptRevision": 0, "error": None,
        }
        session = Session(self, snapshot)
        self.sessions[snapshot["id"]] = session
        session.save()
        return session

    async def close(self) -> None:
        for session in self.sessions.values():
            await session.close()
        self.store.close()


def create_app(
    settings: providers.Settings | None = None,
    data_dir: Path | None = None,
    extraction_delay: float = 0.4,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = Service(
            settings or providers.Settings.from_env(),
            data_dir or Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))),
            extraction_delay,
        )
        yield
        await application.state.service.close()

    application = FastAPI(title="Consultation Assistant API", version="0.1.0", lifespan=lifespan)
    origins = [value.strip() for value in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if value.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "PATCH"], allow_headers=["Content-Type"])

    @application.middleware("http")
    async def body_limit(request: Request, call_next):
        from starlette.responses import JSONResponse
        content_length = request.headers.get("content-length", "0")
        if not content_length.isdigit() or int(content_length) > 256_000:
            return JSONResponse({"detail": "Запрос превышает 256 КБ"}, status_code=413)
        return await call_next(request)

    @application.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @application.get("/api/v1/config")
    async def config(request: Request):
        settings = request.app.state.service.settings
        asr = settings.effective_asr_provider
        llm = settings.effective_llm_provider
        return {
            "asrProvider": asr, "llmProvider": llm, "demoMode": asr == "demo", "llmDemoMode": llm == "demo",
            "asrLanguage": settings.speechmatics_language if asr == "speechmatics" else settings.asr_language,
        }

    @application.get("/api/v1/forms/default")
    async def get_default_form():
        return default_form()

    @application.post("/api/v1/sessions", status_code=201)
    async def create_session(body: CreateSession, request: Request):
        return request.app.state.service.create(body.formSchema).snapshot

    @application.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        return request.app.state.service.get(session_id).snapshot

    @application.patch("/api/v1/sessions/{session_id}/fields/{field_id}")
    async def edit_field(session_id: str, field_id: str, body: EditField, request: Request):
        return await request.app.state.service.get(session_id).edit(field_id, body.value, body.expectedRevision)

    @application.post("/api/v1/sessions/{session_id}/fields/{field_id}/unlock")
    async def unlock_field(session_id: str, field_id: str, body: UnlockField, request: Request):
        return await request.app.state.service.get(session_id).unlock(field_id, body.expectedRevision)

    @application.post("/api/v1/sessions/{session_id}/transcript")
    async def insert_transcript(session_id: str, body: InsertTranscript, request: Request):
        return await request.app.state.service.get(session_id).insert_transcript(body.text, body.speaker)

    @application.post("/api/v1/sessions/{session_id}/stop")
    async def stop_session(session_id: str, request: Request):
        return await request.app.state.service.get(session_id).stop()

    @application.post("/api/v1/sessions/{session_id}/clinical-assessment")
    async def clinical_assessment(session_id: str, request: Request):
        session = request.app.state.service.get(session_id)
        result = await asyncio.shield(session.start_assessment())
        if result["clinicalStatus"] == "error":
            raise HTTPException(502, result["clinicalError"])
        return result

    @application.get("/api/v1/sessions/{session_id}/clinical-assessment/export")
    async def export_clinical_assessment(session_id: str, request: Request):
        snapshot = request.app.state.service.get(session_id).snapshot
        assessment = snapshot["clinicalAssessment"]
        if assessment is None:
            raise HTTPException(404, "Клинический анализ ещё не выполнен")
        return {
            **assessment, "reviewStatus": "requires_doctor_review",
            "stale": assessment["transcriptRevision"] != snapshot["transcriptRevision"]
            or assessment["documentRevision"] != snapshot["documentRevision"],
        }

    @application.get("/api/v1/sessions/{session_id}/export")
    async def export_session(session_id: str, request: Request):
        return request.app.state.service.get(session_id).snapshot["values"]

    @application.websocket("/api/v1/sessions/{session_id}/stream")
    async def websocket_stream(websocket: WebSocket, session_id: str):
        origin = websocket.headers.get("origin")
        if origin and origin not in origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            return
        try:
            session = websocket.app.state.service.get(session_id)
        except HTTPException:
            await websocket.close(code=1008, reason="Session not found")
            return
        await websocket.accept()
        async with session.lock:
            if session.websocket is not None:
                await websocket.close(code=1008, reason="Session already has an active socket")
                return
            session.websocket = websocket
            if session.disconnect_task and not session.disconnect_finalizing:
                session.disconnect_task.cancel()
        await session.publish()
        try:
            while True:
                frame = await websocket.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                try:
                    if frame.get("bytes") is not None:
                        await session.accept_audio(frame["bytes"])
                    elif frame.get("text") is not None:
                        if len(frame["text"]) > 8192:
                            raise HTTPException(413, "Слишком длинное управляющее сообщение")
                        try:
                            message = json.loads(frame["text"])
                        except json.JSONDecodeError as exc:
                            raise HTTPException(422, "Ожидается JSON") from exc
                        if not isinstance(message, dict):
                            raise HTTPException(422, "Ожидается JSON object")
                        if message.get("type") in {"stream.start", "stream.resume"}:
                            await session.start_stream(message)
                        elif message.get("type") == "ping":
                            await session.send("pong", {})
                        else:
                            raise HTTPException(422, "Неизвестный тип сообщения")
                except HTTPException as exc:
                    await session.send("error", {"code": str(exc.status_code), "message": exc.detail})
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            async with session.lock:
                if session.websocket is websocket:
                    session.websocket = None
                    if session.asr is not None and not session.disconnect_finalizing:
                        session.disconnect_task = asyncio.create_task(session.disconnect_timeout())

    return application


app = create_app()
