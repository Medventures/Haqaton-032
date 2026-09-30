from contextlib import nullcontext
import hashlib
from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app import rukk
from app.main import Settings, create_app, load_model


def vocabulary():
    tokens = {index: str(index) for index in range(46)}
    tokens.update({14: "а", 18: "ә", 20: "_", 42: "|", 45: "[UNK]"})
    return tokens


def write_vocabulary(path, tokens=None):
    path.write_text("\n".join(f"{token}\t{index}" for index, token in (tokens or vocabulary()).items()), encoding="utf-8")


def wav(samples=1600):
    source = BytesIO()
    with wave.open(source, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * samples)
    return source.getvalue()


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def to(self, **kwargs):
        self.to_args = kwargs
        return self

    def unsqueeze(self, dimension):
        result = Tensor(np.expand_dims(self.values, dimension))
        result.to_args = self.to_args
        return result

    def __getitem__(self, item):
        return Tensor(self.values[item])

    def argmax(self, dim):
        return Tensor(self.values.argmax(axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()

    def all(self):
        return Tensor(self.values.all())

    def item(self):
        return self.values.item()


def mock_torch(monkeypatch):
    module = ModuleType("torch")
    module.from_numpy = Tensor
    module.float32 = np.float32
    module.inference_mode = nullcontext
    module.is_tensor = lambda item: isinstance(item, Tensor)
    module.isfinite = lambda item: Tensor(np.isfinite(item.values))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def logits(ids):
    scores = np.zeros((1, len(ids), 47), dtype=np.float32)
    for position, token in enumerate(ids):
        scores[0, position, token] = 1
    return Tensor(scores)


def test_ctc_collapses_repeats_before_removing_blank():
    assert rukk.decode_ctc([14, 14, 46, 14], vocabulary()) == "аа"
    assert rukk.decode_ctc([46, 46, 14, 14, 46], vocabulary()) == "а"
    assert rukk.decode_ctc([46, 46], vocabulary()) == ""


def test_ctc_treats_both_separators_as_spaces_and_keeps_unknown_visible():
    assert rukk.decode_ctc([20, 14, 42, 42, 18, 20, 45, 46, 14, 20], vocabulary()) == "а ә [неразборчиво] а"


@pytest.mark.parametrize("bad_id", [-1, 47, 1.0, True, "14"])
def test_invalid_token_ids_are_explicit_errors(bad_id):
    with pytest.raises(rukk.RukkError) as error:
        rukk.decode_ctc([14, bad_id], vocabulary())
    assert error.value.code == "rukk_token_id"


def test_forward_preserves_original_pcm_and_pads_only_short_tail(monkeypatch):
    mock_torch(monkeypatch)
    captured = []

    def model(waveform):
        captured.append(waveform)
        return (logits([14, 14, 46, 14, 20, 18]), "ignored lengths")

    audio = np.array([0.25, -1.0], dtype=np.float32)
    assert rukk.transcribe_waveform(model, audio, vocabulary(), "cuda:0") == "аа ә"
    waveform = captured[0]
    assert waveform.shape == (1, 400)
    assert waveform.to_args == {"device": "cuda:0", "dtype": np.float32}
    np.testing.assert_array_equal(waveform.values[0, :2], audio)
    assert np.count_nonzero(waveform.values[0, 2:]) == 0
    original = np.linspace(-0.7, 0.3, 1000, dtype=np.float32)
    rukk.transcribe_waveform(model, original, vocabulary(), "cpu")
    np.testing.assert_array_equal(captured[1].values[0], original)


@pytest.mark.parametrize("output", [None, [], Tensor(np.zeros((1, 2, 47))), ("not a tensor",), (Tensor(np.zeros((2, 47))),), (Tensor(np.zeros((2, 3, 47))),), (Tensor(np.zeros((1, 3, 46))),), (Tensor(np.zeros((1, 0, 47))),)])
def test_model_output_contract_rejects_wrong_structure_and_dimensions(monkeypatch, output):
    mock_torch(monkeypatch)
    with pytest.raises(rukk.RukkError) as error:
        rukk.transcribe_waveform(lambda _: output, np.zeros(400, dtype=np.float32), vocabulary(), "cpu")
    assert error.value.code == "rukk_output_shape"


def test_nonfinite_model_scores_are_not_decoded_as_fake_text(monkeypatch):
    mock_torch(monkeypatch)
    output = Tensor(np.full((1, 2, 47), np.nan))
    with pytest.raises(rukk.RukkError) as error:
        rukk.transcribe_waveform(lambda _: (output,), np.zeros(400, dtype=np.float32), vocabulary(), "cpu")
    assert error.value.code == "rukk_output_values"


def test_presence_vad_preserves_the_entire_waveform(monkeypatch):
    audio = np.zeros(1600, dtype=np.float32)
    monkeypatch.setattr(rukk, "decode_wav", lambda _: audio)
    monkeypatch.setattr(rukk, "has_speech", lambda _: False)
    calls = []
    monkeypatch.setattr(rukk, "transcribe_waveform", lambda *args: calls.append(args) or "аллергии нет")
    model = object()
    tokens = vocabulary()
    recognizer = rukk.RukkRecognizer(model, tokens, "cpu")
    segments, info = recognizer.transcribe(BytesIO(), language="kk")
    assert list(segments) == [] and info is None and calls == []
    monkeypatch.setattr(rukk, "has_speech", lambda _: True)
    segments, info = recognizer.transcribe(BytesIO(), language="auto", beam_size=5)
    assert [segment.text for segment in segments] == ["аллергии нет"]
    assert calls[0][0] is model and calls[0][1] is audio and calls[0][2] is tokens
    assert calls[0][3] == "cpu"


def test_wrong_model_digest_is_rejected_before_jit_load(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"a deliberately wrong test artifact")
    calls = []
    module = mock_torch(monkeypatch)
    module.jit = SimpleNamespace(load=lambda *args, **kwargs: calls.append(args))
    settings = Settings(engine="rukk", rukk_model_path=str(model_path))
    with pytest.raises(rukk.RukkError) as error:
        load_model(settings)
    assert error.value.code == "rukk_model_checksum_mismatch"
    assert calls == []


def test_verified_loader_uses_selected_device_float32_eval_and_cpu_threads(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"mock model, never executable")
    token_path = tmp_path / "tokens.lst"
    write_vocabulary(token_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calls = []
    module = mock_torch(monkeypatch)

    class Model:
        def float(self):
            calls.append("float32")
            return self

        def eval(self):
            calls.append("eval")
            return self

    model = Model()
    module.set_num_threads = lambda count: calls.append(("threads", count))
    module.jit = SimpleNamespace(load=lambda path, **kwargs: calls.append(("jit.load", path, kwargs)) or model)
    settings = Settings(engine="rukk", rukk_model_path=str(model_path), rukk_tokens_path=str(token_path), rukk_device="cuda:0", rukk_threads=6, rukk_model_sha256=digest, rukk_tokens_sha256=hashlib.sha256(token_path.read_bytes()).hexdigest())
    loaded = load_model(settings)
    assert isinstance(loaded, rukk.RukkRecognizer)
    assert loaded.model is model and loaded.tokens == vocabulary() and loaded.device == "cuda:0"
    assert calls == [("threads", 6), ("jit.load", str(model_path), {"map_location": "cuda:0"}), "float32", "eval"]


def test_token_file_must_have_contiguous_unique_ids(monkeypatch, tmp_path):
    path = tmp_path / "tokens.lst"
    write_vocabulary(path)
    assert rukk.load_tokens(path) == vocabulary()
    for text in ("а\t0\nб\t0", "а\t0\nә\t2", "а\tnot-an-id", ""):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(rukk.RukkError) as error:
            rukk.load_tokens(path)
        assert error.value.code == "rukk_tokens_invalid"


def test_permuted_vocabulary_fails_checksum_before_loading_model(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"mock model")
    token_path = tmp_path / "tokens.lst"
    write_vocabulary(token_path)
    original_digest = hashlib.sha256(token_path.read_bytes()).hexdigest()
    permuted = vocabulary()
    permuted[14], permuted[18] = permuted[18], permuted[14]
    write_vocabulary(token_path, permuted)
    assert len(rukk.load_tokens(token_path)) == 46
    module = mock_torch(monkeypatch)
    calls = []
    module.jit = SimpleNamespace(load=lambda *args, **kwargs: calls.append(True))
    settings = Settings(engine="rukk", rukk_model_path=str(model_path), rukk_tokens_path=str(token_path), rukk_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), rukk_tokens_sha256=original_digest)
    with pytest.raises(rukk.RukkError) as error:
        load_model(settings)
    assert error.value.code == "rukk_tokens_checksum_mismatch" and calls == []


def test_rukk_settings_and_health_are_lazy_and_enforce_25_second_limit(monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "rukk")
    monkeypatch.setenv("RUKK_MODEL_PATH", "/custom/model.pt")
    monkeypatch.setenv("RUKK_TOKENS_PATH", "/custom/tokens.lst")
    monkeypatch.setenv("RUKK_DEVICE", "cuda:0")
    monkeypatch.setenv("RUKK_CPU_THREADS", "3")
    monkeypatch.setenv("RUKK_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("RUKK_TOKENS_SHA256", "b" * 64)
    settings = Settings.from_env()
    assert settings.rukk_model_path == "/custom/model.pt" and settings.rukk_tokens_path == "/custom/tokens.lst"
    assert settings.rukk_threads == 3 and settings.max_audio_seconds == 25
    assert settings.rukk_tokens_sha256 == "b" * 64
    calls = []
    with TestClient(create_app(settings, lambda _: calls.append(True))) as client:
        assert client.get("/health").json() == {
            "status": "ok", "modelLoaded": False, "engine": "rukk", "device": "cuda:0",
            "model": "kazakh-russian-mixed-stt/rukk", "revision": "a" * 64,
        }
        response = client.post("/v1/audio/transcriptions", files={"file": ("too-long.wav", wav(25 * 16000 + 1))})
        assert response.status_code == 422 and calls == []
    for values in ({"rukk_model_sha256": "wrong"}, {"rukk_tokens_sha256": "wrong"}, {"rukk_threads": 0}, {"rukk_device": "mps"}):
        with pytest.raises(ValueError):
            Settings(engine="rukk", **values)


def test_rukk_contract_errors_return_structured_http_detail():
    class WrongOutput:
        def transcribe(self, *_args, **_kwargs):
            raise rukk.RukkError("rukk_output_shape", "Некорректная форма выхода RUKK")

    with TestClient(create_app(Settings(engine="rukk"), lambda _: WrongOutput())) as client:
        response = client.post("/v1/audio/transcriptions", files={"file": ("speech.wav", wav())})
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "rukk_output_shape", "message": "Некорректная форма выхода RUKK"}}

    def wrong_digest(_settings):
        raise rukk.RukkError("rukk_model_checksum_mismatch", "Контрольная сумма не совпадает")

    with TestClient(create_app(Settings(engine="rukk"), wrong_digest)) as client:
        response = client.post("/v1/models/load")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "rukk_model_checksum_mismatch"
