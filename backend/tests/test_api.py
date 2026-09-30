import asyncio
import copy
import json
import struct

import httpx
import pytest
from fastapi.testclient import TestClient

from app import clinical, providers
from app.main import Service, create_app, default_form
from app.schemas import InvalidForm, validate_form, validate_values


@pytest.fixture
def client(tmp_path):
    app = create_app(providers.Settings(), tmp_path, extraction_delay=0)
    with TestClient(app) as test_client:
        yield test_client


def create(client):
    response = client.post("/api/v1/sessions", json={"formSchema": default_form()})
    assert response.status_code == 201, response.text
    return response.json()


def receive_type(socket, expected):
    for _ in range(100):
        event = socket.receive_json()
        if event["type"] == expected:
            return event["payload"]
    raise AssertionError(f"Missing event {expected}")


def test_health_config_default_and_schema_errors(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/config").json() == {
        "asrProvider": "demo", "llmProvider": "demo", "demoMode": True, "llmDemoMode": True, "asrLanguage": "auto",
    }
    schema = client.get("/api/v1/forms/default").json()
    assert "requiredForApproval" not in str(schema)
    assert client.post("/api/v1/sessions", json={"formSchema": schema, "unexpected": True}).status_code == 422
    for bad in [
        {**schema, "additionalProperties": True},
        {**schema, "required": []},
        {**schema, "$ref": "https://example.com/schema"},
        {**schema, "properties": {"nested": {"type": "object"}}},
    ]:
        assert client.post("/api/v1/sessions", json={"formSchema": bad}).status_code == 422
    assert client.get("/api/v1/sessions/missing").status_code == 404


def test_manual_edits_null_enum_and_version_conflicts(client):
    session = create(client)
    base = f"/api/v1/sessions/{session['id']}"
    assert all(value is None for value in session["values"].values())
    endpoint = base + "/fields/allergy_status"
    assert client.patch(endpoint, json={"value": "made_up", "expectedRevision": 0}).status_code == 422
    assert client.patch(endpoint, json={"value": 5, "expectedRevision": 0}).status_code == 422
    changed = client.patch(endpoint, json={"value": "denied", "expectedRevision": 0}).json()
    assert changed["fieldMeta"]["allergy_status"] == {"revision": 1, "source": "doctor", "locked": True}
    assert client.patch(endpoint, json={"value": "present", "expectedRevision": 0}).status_code == 409
    cleared = client.patch(endpoint, json={"value": None, "expectedRevision": 1}).json()
    assert cleared["values"]["allergy_status"] is None
    assert cleared["fieldMeta"]["allergy_status"]["locked"] is True
    assert client.post(endpoint + "/unlock", json={"expectedRevision": 1}).status_code == 409
    unlocked = client.post(endpoint + "/unlock", json={"expectedRevision": 2}).json()
    assert unlocked["fieldMeta"]["allergy_status"]["locked"] is False
    assert client.get(base + "/export").json()["allergy_status"] is None


def test_audio_pipeline_populates_then_stops_and_preserves_manual_text(client, monkeypatch):
    phrases = [
        "Жалобы: Головная боль", "Анамнез: Боль появилась вчера", "Аллергические реакции: Отрицает",
        "Принимаемые препараты: Парацетамол 500 мг", "Рекомендации врача: Повторный приём",
    ]

    class FakeASR(providers.DemoASR):
        def __init__(self, on_event):
            super().__init__()
            self.on_event = on_event
            self.index = 0

        async def send_audio(self, pcm):
            await super().send_audio(pcm)
            await self.on_event({
                "type": "final", "id": f"fake_{self.index}", "revision": 1, "text": phrases[self.index],
                "startMs": self.index * 200, "endMs": (self.index + 1) * 200, "speaker": None,
            })
            self.index += 1

    monkeypatch.setattr(providers, "create_asr", lambda settings, on_event: FakeASR(on_event))
    session = create(client)
    base = f"/api/v1/sessions/{session['id']}"
    client.patch(base + "/fields/complaints", json={"value": "Правка врача", "expectedRevision": 0})
    with client.websocket_connect(base + "/stream") as socket:
        assert socket.receive_json()["type"] == "session.snapshot"
        socket.send_json({"type": "stream.start", "audio": {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1}})
        receive_type(socket, "stream.ready")
        for index in range(len(phrases)):
            socket.send_bytes(struct.pack("<II", index + 1, index * 3200) + b"\x00\x00" * 3200)
            assert receive_type(socket, "audio.ack")["throughSeq"] == index + 1
        final = client.post(base + "/stop").json()
    assert len(final["transcript"]) == 5
    assert final["transcriptRevision"] == 5
    assert final["values"]["complaints"] == "Правка врача"
    assert final["values"]["allergy_status"] == "denied"
    assert final["values"]["allergy_details"] is None
    assert "500" in final["values"]["medications"]
    assert "suggestion" in final["fieldMeta"]["complaints"]
    assert final["error"] is None
    assert client.post(base + "/demo").status_code == 404
    assert not any(path.endswith("/demo") for path in client.get("/openapi.json").json()["paths"])
    assert client.post(base + "/stop").json()["status"] == "stopped"


def test_text_injection_appends_and_drains_extraction(client):
    session = create(client)
    base = f"/api/v1/sessions/{session['id']}"
    assert client.post(base + "/transcript", json={"text": "   "}).status_code == 422
    response = client.post(base + "/transcript", json={"text": "Жалобы: Болит голова", "speaker": None})
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["status"] == "stopped"
    assert snapshot["values"]["complaints"] == "Болит голова"
    response = client.post(base + "/transcript", json={"text": "Жалобы: Боль прошла", "speaker": "Пациент"})
    assert response.status_code == 200
    assert response.json()["values"]["complaints"] == "Боль прошла"
    assert len(response.json()["transcript"]) == 2


@pytest.mark.parametrize("invalid_enum", [False, True])
def test_llm_normalized_fields_preserve_raw_asr_and_pass_session_validation(monkeypatch, tmp_path, invalid_enum):
    # This verifies transport/persistence/validation, not the model's linguistic
    # quality: the LLM answer is an explicit fake and no external API is called.
    raw_text = "Басым үш күннен бері ауырады. Вчера прини мал парацетомол 250 мг. Аллергии нет."
    normalized = {key: None for key in default_form()["properties"]}
    normalized.update({
        "complaints": "Головная боль в течение трёх дней.",
        "medications": "Вчера принимал парацетамол 250 мг.",
        "allergy_status": "unknown-code" if invalid_enum else "denied",
    })
    payloads = []

    async def unused_assessment(*args):
        return {key: None for key in clinical.FIELDS}

    # Clinical analysis is a separate request after stopping; keep this contract
    # test focused on extraction rather than giving it an unrelated mock answer.
    monkeypatch.setattr(clinical, "assess", unused_assessment)

    def transport(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(normalized, ensure_ascii=False)},
        }]})

    async_client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: async_client(
        transport=httpx.MockTransport(transport), **kwargs,
    ))
    settings = providers.Settings(asr_provider="demo", llm_provider="deepseek", llm_api_key="test-only")
    with TestClient(create_app(settings, tmp_path, extraction_delay=0)) as client:
        session = create(client)
        base = f"/api/v1/sessions/{session['id']}"
        response = client.post(base + "/transcript", json={"text": raw_text, "speaker": "Пациент"})
        assert response.status_code == 200
        snapshot = response.json()
        assert len(payloads) == 1
        llm_input = json.loads(payloads[0]["messages"][1]["content"])
        segment = snapshot["transcript"][0]
        assert segment["text"] == raw_text
        assert llm_input["transcript"] == [{
            key: segment[key] for key in ("id", "revision", "text", "speaker")
        }]
        assert llm_input["currentValues"] == session["values"]
        stored = client.get(base).json()
        assert stored["transcript"] == snapshot["transcript"]
        if invalid_enum:
            assert snapshot["status"] == "error"
            assert snapshot["error"]
            assert snapshot["values"] == session["values"]
        else:
            assert snapshot["status"] == "stopped"
            assert snapshot["error"] is None
            assert snapshot["values"] == normalized
            assert stored["values"] == normalized
            assert snapshot["fieldMeta"]["medications"]["source"] == "llm"


def test_demo_asr_can_use_real_llm_and_live_asr_rejects_injection(monkeypatch, tmp_path):
    calls = []

    async def extract(settings, schema, segments, current_values):
        calls.append(settings.effective_llm_provider)
        return {key: "Результат внешней модели" if key == "complaints" else None for key in schema["properties"]}

    monkeypatch.setattr(providers, "extract", extract)
    settings = providers.Settings(asr_provider="demo", llm_provider="deepseek")
    with TestClient(create_app(settings, tmp_path / "mixed", extraction_delay=0)) as client:
        config = client.get("/api/v1/config").json()
        assert config["demoMode"] is True
        assert config["llmDemoMode"] is False
        assert config["llmProvider"] == "deepseek"
        session = create(client)
        response = client.post(f"/api/v1/sessions/{session['id']}/transcript", json={"text": "Тест"})
        assert response.json()["values"]["complaints"] == "Результат внешней модели"
        assert calls == ["deepseek"]
    settings = providers.Settings(asr_provider="openai-compatible", llm_provider="demo", asr_language="kk")
    with TestClient(create_app(settings, tmp_path / "live-asr")) as client:
        assert client.get("/api/v1/config").json()["asrLanguage"] == "kk"
        session = create(client)
        base = f"/api/v1/sessions/{session['id']}"
        assert client.post(base + "/transcript", json={"text": "Тест"}).status_code == 409
        assert client.post(base + "/demo").status_code == 404


def test_websocket_ack_duplicate_order_resume_and_second_socket(client):
    session = create(client)
    base = f"/api/v1/sessions/{session['id']}"
    audio = {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1}
    packet = struct.pack("<II", 1, 0) + b"\x00\x00" * 3200
    with client.websocket_connect(base + "/stream") as socket:
        assert socket.receive_json()["type"] == "session.snapshot"
        socket.send_bytes(packet)
        assert receive_type(socket, "error")["code"] == "409"
        socket.send_json({"type": "stream.start", "audio": audio})
        stream = receive_type(socket, "stream.ready")
        assert stream["throughSeq"] == 0
        with pytest.raises(Exception):
            with client.websocket_connect(base + "/stream") as other:
                other.receive_json()
        socket.send_bytes(packet)
        assert receive_type(socket, "audio.ack")["throughSeq"] == 1
        socket.send_bytes(packet)
        assert receive_type(socket, "audio.ack")["throughSeq"] == 1
        socket.send_bytes(struct.pack("<II", 1, 0) + b"\x01\x00" * 3200)
        assert receive_type(socket, "error")["code"] == "409"
        socket.send_bytes(struct.pack("<II", 3, 3200) + b"\x00\x00" * 100)
        assert receive_type(socket, "error")["code"] == "409"
    with client.websocket_connect(base + "/stream") as socket:
        receive_type(socket, "session.snapshot")
        socket.send_json({"type": "stream.resume", "streamId": stream["streamId"], "audio": audio})
        resumed = receive_type(socket, "stream.ready")
        assert resumed == {"streamId": stream["streamId"], "throughSeq": 1, "nextSampleOffset": 3200}
        socket.send_bytes(struct.pack("<II", 2, 3200) + b"\x00\x00" * 160)
        assert receive_type(socket, "audio.ack")["throughSeq"] == 2
        assert client.post(base + "/stop").json()["status"] == "stopped"
        socket.send_bytes(struct.pack("<II", 3, 3360) + b"\x00\x00" * 160)
        assert receive_type(socket, "error")["code"] == "409"
    store = client.app.state.service.store
    assert store.connection.execute("SELECT COUNT(*) FROM audio").fetchone()[0] == 2


def test_repeated_start_recovers_lost_ready_without_reopening_asr(client, monkeypatch):
    starts = []

    def create_asr(settings, on_event):
        adapter = providers.DemoASR()
        starts.append(adapter)
        return adapter

    monkeypatch.setattr(providers, "create_asr", create_asr)
    session = create(client)
    path = f"/api/v1/sessions/{session['id']}/stream"
    command = {"type": "stream.start", "audio": {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1}}
    with client.websocket_connect(path) as first:
        receive_type(first, "session.snapshot")
        first.send_json(command)
        ready = receive_type(first, "stream.ready")
    # Model a browser that did not retain the original ready response: no streamId.
    with client.websocket_connect(path) as second:
        receive_type(second, "session.snapshot")
        second.send_json(command)
        assert receive_type(second, "stream.ready") == ready
        assert len(starts) == 1
        second.send_bytes(struct.pack("<II", 1, 0) + b"\x00\x00" * 3200)
        assert receive_type(second, "audio.ack")["throughSeq"] == 1
        second.send_json(command)
        assert receive_type(second, "error")["code"] == "409"
        assert len(starts) == 1


def test_persisted_snapshot_and_interrupted_stream_survive_restart(tmp_path):
    with TestClient(create_app(providers.Settings(), tmp_path)) as client:
        session = create(client)
        base = f"/api/v1/sessions/{session['id']}"
        client.patch(base + "/fields/complaints", json={"value": "Сохранено", "expectedRevision": 0})
        with client.websocket_connect(base + "/stream") as socket:
            receive_type(socket, "session.snapshot")
            socket.send_json({"type": "stream.start", "audio": {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1}})
            receive_type(socket, "stream.ready")
    with TestClient(create_app(providers.Settings(), tmp_path)) as client:
        saved = client.get(base).json()
        assert saved["values"]["complaints"] == "Сохранено"
        assert saved["status"] == "error"
        assert saved["error"]


def test_schema_rejects_unknown_enum_metadata_and_clamps_unconstrained_text():
    schema = default_form()
    invalid = copy.deepcopy(schema)
    invalid["properties"]["allergy_status"]["x-enumLabels"]["wrong"] = "Wrong"
    with pytest.raises(InvalidForm):
        validate_form(invalid)
    with pytest.raises(InvalidForm):
        validate_values(schema, {key: None for key in schema["properties"]} | {"complaints": "x" * 8001})


@pytest.mark.asyncio
async def test_inflight_extraction_cannot_overwrite_manual_clear(monkeypatch, tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def extract(*args):
        entered.set()
        await release.wait()
        return {key: "Головная боль" if key == "complaints" else None for key in default_form()["properties"]}

    monkeypatch.setattr(providers, "extract", extract)
    service = Service(providers.Settings(), tmp_path, 0)
    session = service.create(default_form())
    await session.on_asr_event({"type": "final", "id": "segment", "revision": 1, "text": "Болит голова"})
    await asyncio.wait_for(entered.wait(), 2)
    await session.edit("complaints", None, 0)
    release.set()
    await session.extraction_task
    assert session.snapshot["values"]["complaints"] is None
    assert session.snapshot["fieldMeta"]["complaints"]["suggestion"] == "Головная боль"
    assert session.snapshot["fieldMeta"]["complaints"]["locked"] is True
    await service.close()


@pytest.mark.asyncio
async def test_transcript_revisions_replace_and_failed_extraction_is_explicit(monkeypatch, tmp_path):
    async def fail(*args):
        raise providers.ProviderError("failed", "Тестовая ошибка")

    monkeypatch.setattr(providers, "extract", fail)
    service = Service(providers.Settings(), tmp_path, 0)
    session = service.create(default_form())
    for revision, text in [(1, "Первая версия"), (1, "Повтор"), (2, "Уточнение")]:
        await session.on_asr_event({"type": "final", "id": "segment", "revision": revision, "text": text})
    final = await session.stop()
    assert len(final["transcript"]) == 1
    assert final["transcript"][0]["text"] == "Уточнение"
    assert final["transcriptRevision"] == 2
    assert final["status"] == "error"
    assert "Тестовая ошибка" in final["error"]
    await service.close()


@pytest.mark.asyncio
async def test_correction_discards_inflight_candidate_before_next_extraction(monkeypatch, tmp_path):
    first_started, second_started = asyncio.Event(), asyncio.Event()
    first_release, second_release = asyncio.Event(), asyncio.Event()

    async def extract(settings, schema, segments, current_values):
        if segments[0]["revision"] == 1:
            first_started.set()
            await first_release.wait()
        else:
            second_started.set()
            await second_release.wait()
        return {key: segments[0]["text"] if key == "complaints" else None for key in schema["properties"]}

    monkeypatch.setattr(providers, "extract", extract)
    service = Service(providers.Settings(), tmp_path, 0)
    session = service.create(default_form())
    await session.on_asr_event({"type": "final", "id": "segment", "revision": 1, "text": "Неверное распознавание"})
    await asyncio.wait_for(first_started.wait(), 2)
    await session.on_asr_event({"type": "final", "id": "segment", "revision": 2, "text": "Исправлено"})
    first_release.set()
    await asyncio.wait_for(second_started.wait(), 2)
    assert session.snapshot["values"]["complaints"] is None
    second_release.set()
    await session.extraction_task
    assert session.snapshot["values"]["complaints"] == "Исправлено"
    await service.close()
