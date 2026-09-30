from contextlib import nullcontext
from io import BytesIO
import struct
import sys
from types import ModuleType, SimpleNamespace
import wave

import numpy as np
import pytest

from app import gigaam
from app.main import Settings, load_model


def source_wav(samples=(0, 32767, -32768, 8192)):
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return BytesIO(output.getvalue())


def test_pcm_decoding_preserves_sign_scale_and_rejects_long_chunks():
    audio = gigaam.decode_wav(source_wav())
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0, 32767 / 32768, -1, 0.25])
    with pytest.raises(ValueError, match="25 seconds"):
        gigaam.decode_wav(source_wav([0] * (16000 * 25 + 1)))


def test_vad_skips_silence_without_running_asr_and_preserves_full_speech_waveform(monkeypatch):
    audio = np.zeros(1600, dtype=np.float32)
    monkeypatch.setattr(gigaam, "decode_wav", lambda source: audio)
    monkeypatch.setattr(gigaam, "has_speech", lambda candidate: False)
    calls = []
    monkeypatch.setattr(gigaam, "transcribe_waveform", lambda model, candidate: calls.append(candidate) or SimpleNamespace(text=" Болит горло "))
    recognizer = gigaam.GigaAMRecognizer(object())
    segments, info = recognizer.transcribe(BytesIO(), language="ru")
    assert list(segments) == [] and info is None and calls == []
    monkeypatch.setattr(gigaam, "has_speech", lambda candidate: True)
    segments, info = recognizer.transcribe(BytesIO(), language="kk", beam_size=5)
    assert [segment.text for segment in segments] == ["Болит горло"]
    assert calls[0] is audio


def test_vad_options_keep_short_negations_and_only_check_presence(monkeypatch):
    package = ModuleType("faster_whisper")
    vad = ModuleType("faster_whisper.vad")
    captured = []
    vad.VadOptions = lambda **kwargs: kwargs
    vad.get_speech_timestamps = lambda audio, **kwargs: captured.append((audio, kwargs)) or [{"start": 10, "end": 20}]
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    audio = np.zeros(1600, dtype=np.float32)
    assert gigaam.has_speech(audio) is True
    assert captured[0][0] is audio
    assert captured[0][1]["sampling_rate"] == 16000
    assert captured[0][1]["vad_options"]["min_speech_duration_ms"] == 0


def test_tensor_path_pads_short_tail_and_uses_decoded_text_not_result_repr(monkeypatch):
    captured = {}

    class Tensor:
        def __init__(self, audio):
            self.audio = audio
            self.shape = audio.shape

        def to(self, **kwargs):
            captured["to"] = kwargs
            return self

        def unsqueeze(self, dimension):
            assert dimension == 0
            self.shape = (1, *self.shape)
            return self

    def from_numpy(audio):
        captured["audio"] = audio
        return Tensor(audio)

    def full(shape, length, **kwargs):
        captured["length_kwargs"] = kwargs
        assert shape == [1]
        return np.array([length])

    def forward(waveform, lengths):
        assert waveform.shape == (1, 320)
        assert lengths.tolist() == [320]
        return "encoded", "encoded-length"

    def decode(encoded, encoded_length, lengths, word_timestamps):
        assert (encoded, encoded_length, word_timestamps) == ("encoded", "encoded-length", False)
        assert lengths.tolist() == [320]
        return [("лекарства не принимал", None)]

    torch = ModuleType("torch")
    torch.from_numpy = from_numpy
    torch.full = full
    torch.long = "int64"
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    inner = SimpleNamespace(_device="cuda:0", _dtype="float32", forward=forward, _decode=decode)
    result = gigaam.transcribe_waveform(SimpleNamespace(model=inner), np.array([0.25, -0.5], dtype=np.float32))
    assert result.text == "лекарства не принимал"
    np.testing.assert_allclose(captured["audio"][:2], [0.25, -0.5])
    assert np.count_nonzero(captured["audio"][2:]) == 0
    assert captured["to"] == {"device": "cuda:0", "dtype": "float32"}
    assert captured["length_kwargs"] == {"device": "cuda:0", "dtype": "int64"}


def test_loader_pins_snapshot_and_loads_explicit_local_classes(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "modeling_gigaam.py").write_text('''
import importlib
from dataclasses import dataclass

calls = {}

@dataclass
class FeatureExtractor:
    marker: str = "registered during execution"

class GigaAMConfig:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        calls["config"] = (path, kwargs)
        return cls()

class GigaAMModel:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        calls["model"] = (path, kwargs)
        # Hydra imports these exact canonical module targets.
        assert importlib.import_module("modeling_gigaam").FeatureExtractor is FeatureExtractor
        return cls()

    def to(self, device):
        calls["device"] = device
        return self

    def eval(self):
        calls["eval"] = True
        return self
''', encoding="utf-8")
    downloads = []
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: downloads.append(kwargs) or str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "modeling_gigaam", None)
    settings = Settings(engine="gigaam", gigaam_device="cpu", download_root=str(tmp_path))
    result = load_model(settings)
    module = sys.modules["modeling_gigaam"]
    calls = module.calls
    assert isinstance(result, gigaam.GigaAMRecognizer)
    assert isinstance(result.model, module.GigaAMModel)
    assert downloads == [{
        "repo_id": "ai-sage/GigaAM-Multilingual", "revision": settings.gigaam_revision,
        "allow_patterns": ["config.json", "modeling_gigaam.py", "pytorch_model.bin"],
        "cache_dir": str(tmp_path / "gigaam"),
    }]
    assert calls["config"] == (str(snapshot), {"local_files_only": True})
    assert calls["model"][0] == str(snapshot)
    assert isinstance(calls["model"][1]["config"], module.GigaAMConfig)
    assert {key: value for key, value in calls["model"][1].items() if key != "config"} == {
        "weights_only": True, "use_safetensors": False, "local_files_only": True,
    }
    assert calls["device"] == "cpu" and calls["eval"] is True
    assert gigaam._load_model_code(snapshot / "modeling_gigaam.py") is module


def test_model_code_refuses_conflicting_canonical_module(monkeypatch, tmp_path):
    existing = ModuleType("modeling_gigaam")
    existing.__file__ = str(tmp_path / "another-revision" / "modeling_gigaam.py")
    monkeypatch.setitem(sys.modules, "modeling_gigaam", existing)
    source = tmp_path / "modeling_gigaam.py"
    source.write_text("raise AssertionError('must not execute conflicting code')", encoding="utf-8")
    with pytest.raises(RuntimeError, match="another source"):
        gigaam._load_model_code(source)
    assert sys.modules["modeling_gigaam"] is existing


def test_failed_model_code_import_can_be_retried(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "modeling_gigaam", None)
    source = tmp_path / "modeling_gigaam.py"
    source.write_text("raise ImportError('missing required dependency')", encoding="utf-8")
    with pytest.raises(ImportError, match="missing required dependency"):
        gigaam._load_model_code(source)
    assert "modeling_gigaam" not in sys.modules
    source.write_text("loaded_after_retry = True", encoding="utf-8")
    assert gigaam._load_model_code(source).loaded_after_retry is True


def test_whisper_loader_remains_independent(monkeypatch):
    calls = []
    module = ModuleType("faster_whisper")
    module.WhisperModel = lambda name, **kwargs: calls.append((name, kwargs)) or "whisper-model"
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    settings = Settings(model="small", device="cpu", compute_type="int8", cpu_threads=6)
    assert load_model(settings) == "whisper-model"
    assert calls == [("small", {
        "device": "cpu", "compute_type": "int8", "cpu_threads": 6, "num_workers": 1, "download_root": "/models",
    })]
