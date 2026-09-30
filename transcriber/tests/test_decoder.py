"""Real dependency test; the Docker build runs the same check unconditionally."""

import pytest

from app.decoder_smoke import verify_audio_decoder


def test_real_faster_whisper_decoder_without_model_download():
    pytest.importorskip("av", reason="Install transcriber/requirements.txt for the native decoder test")
    pytest.importorskip("faster_whisper", reason="Install transcriber/requirements.txt for the native decoder test")
    result = verify_audio_decoder()
    assert result["status"] == "ok"
    assert result["decodedSamplesAt16000Hz"] == {"16000": 16000, "48000": 16000}
