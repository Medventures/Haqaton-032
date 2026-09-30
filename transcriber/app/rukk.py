"""Local TorchScript adapter for the supplied Kazakh/Russian CTC model.

Loading is explicit and verifies the expected model digest before torch.jit.load.
Importing this module does not import PyTorch or execute the model artifact.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .gigaam import decode_wav, has_speech


DEFAULT_MODEL_SHA256 = "92fe1791d7c93f385fc10a959dd216caa98c6a571df72f11d1605c6327393b88"
DEFAULT_TOKENS_SHA256 = "d6e8335d2268efc64e3e5e4dfaeadc749998248f126aa6a9f2a3190c317db4d1"
MODEL_NAME = "kazakh-russian-mixed-stt/rukk"
TOKEN_COUNT = 46
MIN_INPUT_SAMPLES = 400


class RukkError(ValueError):
    """A safe, structured model-contract error suitable for the HTTP boundary."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def detail(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def verify_model_checksum(path: Path, expected: str) -> None:
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        raise RukkError("rukk_model_unavailable", "Файл модели RUKK недоступен. Проверьте подключение файлов модели.") from None
    if digest != expected.lower():
        raise RukkError("rukk_model_checksum_mismatch", "Контрольная сумма модели RUKK не совпадает с настроенной; модель не загружена.")


def verify_tokens_checksum(path: Path, expected: str) -> None:
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        raise RukkError("rukk_tokens_unavailable", "Словарь RUKK недоступен. Проверьте подключение файла токенов.") from None
    if digest != expected.lower():
        raise RukkError("rukk_tokens_checksum_mismatch", "Контрольная сумма словаря RUKK не совпадает с настроенной; модель не загружена.")


def load_tokens(path: Path) -> dict[int, str]:
    try:
        if path.stat().st_size > 65536:
            raise ValueError("Token file is too large")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        tokens: dict[int, str] = {}
        for line in lines:
            if not line.strip():
                continue
            token, index_text = line.rsplit(maxsplit=1)
            index = int(index_text)
            if index in tokens or not token or any(char.isspace() for char in token):
                raise ValueError("Duplicate or invalid token")
            tokens[index] = token
        if set(tokens) != set(range(TOKEN_COUNT)):
            raise ValueError("Expected token IDs 0 through 45")
        return tokens
    except (OSError, UnicodeError, ValueError):
        raise RukkError("rukk_tokens_invalid", "Некорректный словарь RUKK: ожидаются 46 токенов с уникальными номерами от 0 до 45.") from None


def decode_ctc(token_ids: list[int], tokens: dict[int, str]) -> str:
    blank = max(tokens) + 1
    previous = None
    pieces: list[str] = []
    for token_id in token_ids:
        if type(token_id) is not int or (token_id != blank and token_id not in tokens):
            raise RukkError("rukk_token_id", "Модель RUKK вернула неизвестный номер токена.")
        # CTC repeats collapse before blank removal: a,a,blank,a must become aa.
        if token_id == previous:
            continue
        previous = token_id
        if token_id == blank:
            continue
        token = tokens[token_id]
        if token in {"|", "_"}:
            pieces.append(" ")
        elif token == "[UNK]":
            pieces.append(" [неразборчиво] ")
        else:
            pieces.append(token)
    return " ".join("".join(pieces).split())


def transcribe_waveform(model: Any, audio, tokens: dict[int, str], device: str) -> str:
    import numpy as np
    import torch

    # PCM scaling is done once by decode_wav. Do not normalize the waveform:
    # the published TorchScript interface expects the original float32 PCM.
    if len(audio) < MIN_INPUT_SAMPLES:
        audio = np.pad(audio, (0, MIN_INPUT_SAMPLES - len(audio)))
    waveform = torch.from_numpy(audio).to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        output = model(waveform)
        if not isinstance(output, (tuple, list)) or not output:
            raise RukkError("rukk_output_shape", "Модель RUKK вернула результат неподдерживаемой структуры.")
        logits = output[0]
        classes = max(tokens) + 2  # Tokens 0..45 plus CTC blank 46.
        if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[2] != classes:
            raise RukkError("rukk_output_shape", "Некорректная форма выхода RUKK: ожидается [1, T, 47].")
        if not torch.isfinite(logits).all().item():
            raise RukkError("rukk_output_values", "Модель RUKK вернула некорректные числовые значения.")
        token_ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
    return decode_ctc(token_ids, tokens)


class RukkRecognizer:
    def __init__(self, model: Any, tokens: dict[int, str], device: str):
        self.model = model
        self.tokens = tokens
        self.device = device

    def transcribe(self, source: BytesIO, **_kwargs):
        audio = decode_wav(source)
        # VAD only determines whether any speech exists. Keep pauses, quiet
        # negations, and the original waveform; do not crop/concatenate regions.
        if not has_speech(audio):
            return iter(()), None
        text = transcribe_waveform(self.model, audio, self.tokens, self.device)
        return iter((SimpleNamespace(text=text),)), None


def load_rukk(settings) -> RukkRecognizer:
    model_path = Path(settings.rukk_model_path)
    verify_model_checksum(model_path, settings.rukk_model_sha256)
    tokens_path = Path(settings.rukk_tokens_path)
    verify_tokens_checksum(tokens_path, settings.rukk_tokens_sha256)
    tokens = load_tokens(tokens_path)

    import torch

    torch.set_num_threads(settings.rukk_threads)
    model = torch.jit.load(str(model_path), map_location=settings.rukk_device).float().eval()
    return RukkRecognizer(model, tokens, settings.rukk_device)
