"""Clinical reasoning contracts and lifecycle; no real patient data or API calls."""

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import clinical, providers
from app.main import Service, create_app, default_form
from app.schemas import InvalidForm
from app.storage import Store


VALUES = {key: None for key in clinical.FIELDS}
VALUES.update({"diagnosis": "Предварительная гипотеза", "missing_data": "Уточнить возраст и аллергию"})
SETTINGS = providers.Settings(llm_provider="openai-compatible", asr_provider="demo",
                              llm_base_url="http://localhost:11434/v1", llm_model="test-model")


@pytest.mark.asyncio
async def test_clinical_request_uses_separate_schema_and_local_model(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(VALUES, ensure_ascii=False),
        }}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs
    ))
    values = {key: None for key in default_form()["properties"]}
    values["complaints"] = "Синтетические жалобы"
    result = await clinical.assess(SETTINGS, [{"text": "Вопрос врача", "speaker": "doctor"}], values, default_form())
    assert result == VALUES
    assert "authorization" not in requests[0].headers
    payload = json.loads(requests[0].content)
    assert payload["model"] == "test-model"
    context = json.loads(payload["messages"][1]["content"])
    assert context["currentValues"]["complaints"]["value"] == "Синтетические жалобы"
    assert context["transcript"][0]["speaker"] == "doctor"
    assert "diagnosis" in payload["messages"][0]["content"]
    assert payload["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", [{}, {**VALUES, "diagnosis": 0}, {**VALUES, "extra": "value"},
                                       {**VALUES, "treatment": "x" * 8001}])
async def test_clinical_rejects_invalid_fields(monkeypatch, candidate):
    async def generate(*args):
        return candidate
    monkeypatch.setattr(providers, "generate_values", generate)
    with pytest.raises(InvalidForm):
        await clinical.assess(SETTINGS, [], {}, default_form())


def test_clinical_demo_refuses_to_fabricate_diagnosis(tmp_path):
    with TestClient(create_app(providers.Settings(), tmp_path, 0)) as client:
        snapshot = client.post("/api/v1/sessions", json={"formSchema": default_form()}).json()
        base = f"/api/v1/sessions/{snapshot['id']}"
        assert client.post(base + "/clinical-assessment").status_code == 409
        assert client.get(base + "/clinical-assessment/export").status_code == 404
        assert client.get(base).json()["clinicalAssessment"] is None


def test_manual_assessment_persists_and_export_marks_stale(tmp_path, monkeypatch):
    calls = []

    async def assess(settings, segments, values, schema):
        calls.append(copy.deepcopy(values))
        return VALUES
    monkeypatch.setattr(clinical, "assess", assess)
    with TestClient(create_app(SETTINGS, tmp_path, 0)) as client:
        snapshot = client.post("/api/v1/sessions", json={"formSchema": default_form()}).json()
        base = f"/api/v1/sessions/{snapshot['id']}"
        assert client.post(base + "/clinical-assessment").status_code == 422
        client.patch(base + "/fields/complaints", json={"value": "Синтетический симптом", "expectedRevision": 0})
        response = client.post(base + "/clinical-assessment")
        assert response.status_code == 200
        snapshot = response.json()
        assert snapshot["clinicalStatus"] == "ready"
        assert snapshot["clinicalAssessment"]["values"] == VALUES
        assert snapshot["values"]["diagnosis"] is None
        assert snapshot["fieldMeta"]["complaints"]["locked"]
        assert calls[0]["complaints"] == "Синтетический симптом"
        export = client.get(base + "/clinical-assessment/export").json()
        assert export["stale"] is False
        assert export["reviewStatus"] == "requires_doctor_review"
        assert client.get(base + "/export").json()["diagnosis"] is None
        client.patch(base + "/fields/complaints", json={"value": "Уточнение", "expectedRevision": 1})
        assert client.get(base + "/clinical-assessment/export").json()["stale"] is True
        client.post(base + "/clinical-assessment")
        assert client.get(base + "/clinical-assessment/export").json()["stale"] is False

    # Old forms and sessions are also supported without schema migration.
    with TestClient(create_app(SETTINGS, tmp_path, 0)) as client:
        assert client.get(base).json()["clinicalAssessment"]["values"] == VALUES


@pytest.mark.asyncio
async def test_auto_analysis_after_stop_and_provider_failure_preserves_previous(tmp_path, monkeypatch):
    async def extract(settings, schema, segments, values):
        return {**values, "complaints": "Синтетический симптом"}

    async def assess(*args):
        return VALUES
    monkeypatch.setattr(providers, "extract", extract)
    monkeypatch.setattr(clinical, "assess", assess)
    service = Service(SETTINGS, tmp_path, 0)
    try:
        session = service.create(default_form())
        await session.insert_transcript("Синтетический разговор", "patient")
        assert session.clinical_task is not None
        await session.clinical_task
        assert session.snapshot["clinicalStatus"] == "ready"
        previous = copy.deepcopy(session.snapshot["clinicalAssessment"])

        async def fail(*args):
            raise providers.ProviderError("llm_unavailable", "Недоступна модель", True)
        monkeypatch.setattr(clinical, "assess", fail)
        result = await session.start_assessment()
        assert result["clinicalStatus"] == "error"
        assert result["clinicalError"] == "Недоступна модель"
        assert result["clinicalAssessment"] == previous
        assert result["status"] == "stopped"
        assert result["error"] is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_concurrent_edits_discard_result_and_duplicate_analysis_is_rejected(tmp_path, monkeypatch):
    from fastapi import HTTPException
    entered = asyncio.Event()
    release = asyncio.Event()

    async def assess(*args):
        entered.set()
        await release.wait()
        return VALUES
    monkeypatch.setattr(clinical, "assess", assess)
    service = Service(SETTINGS, tmp_path, 0)
    try:
        session = service.create(default_form())
        await session.edit("complaints", "Синтетический симптом", 0)
        task = session.start_assessment()
        await entered.wait()
        with pytest.raises(HTTPException) as error:
            session.start_assessment()
        assert error.value.status_code == 409
        await session.edit("complaints", "Уточнение во время анализа", 1)
        release.set()
        await task
        assert session.snapshot["clinicalAssessment"] is None
        assert session.snapshot["clinicalStatus"] == "error"
        assert "изменились" in session.snapshot["clinicalError"]
    finally:
        await service.close()


def test_restart_recovers_interrupted_clinical_analysis(tmp_path):
    service = Service(SETTINGS, tmp_path, 0)
    session = service.create(default_form())
    session.snapshot["clinicalStatus"] = "processing"
    session.save()
    service.store.close()
    store = Store(tmp_path)
    try:
        snapshot = store.get(session.snapshot["id"])
        assert snapshot["clinicalStatus"] == "error"
        assert "перезапущен" in snapshot["clinicalError"]
        assert snapshot["status"] == "ready"
    finally:
        store.close()
