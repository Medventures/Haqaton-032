"""Check real faster-whisper/PyAV decoding without loading any model weights.

Mandatory during Docker build; also runnable as ``python -m app.decoder_smoke``.
Mocked inference tests cannot catch incompatible native audio dependencies.
"""

from array import array
from importlib.metadata import version
from io import BytesIO
import json
import math
import sys
import wave


def verify_audio_decoder() -> dict[str, object]:
    from faster_whisper.audio import decode_audio

    sample_counts = {}
    for source_rate in (16000, 48000):
        samples = array("h", (round(10000 * math.sin(2 * math.pi * 440 * index / source_rate)) for index in range(source_rate)))
        if sys.byteorder != "little":
            samples.byteswap()
        source = BytesIO()
        with wave.open(source, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(source_rate)
            wav.writeframes(samples.tobytes())
        decoded = decode_audio(BytesIO(source.getvalue()), sampling_rate=16000)
        if decoded.shape != (16000,) or str(decoded.dtype) != "float32":
            raise RuntimeError(f"Decoder output mismatch for {source_rate} Hz WAV: {decoded.shape}, {decoded.dtype}")
        peak = float(abs(decoded).max())
        if not 0.25 < peak < 0.4:
            raise RuntimeError(f"Decoded PCM amplitude mismatch: {peak}")
        sample_counts[str(source_rate)] = len(decoded)
    return {
        "status": "ok", "fasterWhisper": version("faster-whisper"), "pyav": version("av"),
        "decodedSamplesAt16000Hz": sample_counts,
    }


if __name__ == "__main__":
    print(json.dumps(verify_audio_decoder()))
