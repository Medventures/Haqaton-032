"""GigaAM adapter for the audited, immutable multilingual model revision.

The pinned model's public transcribe API accepts only a path and invokes FFmpeg.
Use its equivalent tensor forward/decode path to keep request audio in memory.
Heavy modules are imported lazily; importing this module loads no model/code.
"""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any
import wave


SAMPLE_RATE = 16000
MAX_SAMPLES = 25 * SAMPLE_RATE
_MODEL_CODE_LOCK = threading.Lock()


def decode_wav(source: BytesIO):
    import numpy as np

    source.seek(0)
    with wave.open(source, "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("GigaAM requires PCM16 mono 16000 Hz WAV")
        frames = wav.getnframes()
        if not 1 <= frames <= MAX_SAMPLES:
            raise ValueError("GigaAM accepts at most 25 seconds per chunk")
        pcm = wav.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("Incomplete WAV data")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def has_speech(audio) -> bool:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # Only a presence check. Preserve the full original waveform, including
    # quiet words and pauses; do not concatenate/crop detected speech regions.
    return bool(get_speech_timestamps(
        audio, sampling_rate=SAMPLE_RATE,
        vad_options=VadOptions(min_speech_duration_ms=0, min_silence_duration_ms=400, speech_pad_ms=200),
    ))


def transcribe_waveform(model: Any, audio) -> Any:
    import numpy as np
    import torch

    # The pinned FeatureExtractor has n_fft=320 and center=False. A flushed
    # browser tail can be shorter than this; preserve it and pad only the end.
    if len(audio) < 320:
        audio = np.pad(audio, (0, 320 - len(audio)))
    inner = model.model
    waveform = torch.from_numpy(audio).to(device=inner._device, dtype=inner._dtype).unsqueeze(0)
    lengths = torch.full([1], waveform.shape[-1], device=inner._device, dtype=torch.long)
    with torch.inference_mode():
        encoded, encoded_lengths = inner.forward(waveform, lengths)
        text, _words = inner._decode(encoded, encoded_lengths, lengths, False)[0]
    if not isinstance(text, str):
        raise TypeError("GigaAM returned a non-text transcription")
    return SimpleNamespace(text=text)


class GigaAMRecognizer:
    def __init__(self, model: Any):
        self.model = model

    def transcribe(self, source: BytesIO, **_kwargs):
        # This multilingual CTC model has no language/beam-size argument.
        audio = decode_wav(source)
        if not has_speech(audio):
            return iter(()), None
        result = transcribe_waveform(self.model, audio)
        text = result.text
        if not isinstance(text, str):
            raise TypeError("GigaAM transcription result.text must be a string")
        return iter((SimpleNamespace(text=text.strip()),)), None


def _load_model_code(source_path: Path):
    # The audited config uses Hydra targets such as modeling_gigaam.CTCHead.
    # Keep this exact name and register before execution (also needed by
    # dataclasses); do not replace unrelated code already using the name.
    module_name = "modeling_gigaam"
    with _MODEL_CODE_LOCK:
        existing = sys.modules.get(module_name)
        if existing is not None:
            origin = getattr(existing, "__file__", None)
            if origin is None or Path(origin).resolve() != source_path.resolve():
                raise RuntimeError("modeling_gigaam is already loaded from another source")
            return existing
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned GigaAM model source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(module_name) is module:
                del sys.modules[module_name]
            raise
        return module


def load_gigaam(settings) -> GigaAMRecognizer:
    from huggingface_hub import snapshot_download

    # AutoModel's dynamic dependency scanner requires pyannote even though it
    # appears only in unused long-form/diarization functions. Import the same
    # audited code directly from the immutable snapshot and instantiate its
    # explicit HF classes. Code, config and weights share the pinned revision.
    snapshot = Path(snapshot_download(
        repo_id=settings.gigaam_model, revision=settings.gigaam_revision,
        allow_patterns=["config.json", "modeling_gigaam.py", "pytorch_model.bin"],
        cache_dir=str(Path(settings.download_root) / "gigaam"),
    ))
    module = _load_model_code(snapshot / "modeling_gigaam.py")
    config = module.GigaAMConfig.from_pretrained(str(snapshot), local_files_only=True)
    model = module.GigaAMModel.from_pretrained(
        str(snapshot), config=config, weights_only=True,
        use_safetensors=False, local_files_only=True,
    ).to(settings.gigaam_device).eval()
    # Keep preprocessor weights in FP32. The audited encoder handles CUDA
    # autocast internally; applying model.half() would also change preprocessing.
    return GigaAMRecognizer(model)
