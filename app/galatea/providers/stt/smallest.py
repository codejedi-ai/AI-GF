"""
Smallest AI Pulse (and Lightning) STT plugin for LiveKit agents.

Pulse is the default STT mode (32+ languages). Lightning is optional via stt.mode.
Docs-first request flow:
- Pulse: POST /api/v1/pulse/get_text?file_format=wav
- Lightning: POST /api/v1/lightning/get_text?model=<...>&file_format=wav
- Legacy fallback (optional): POST /api/v1/speech-to-text
"""

import asyncio
import io
import logging
import os
import wave
from typing import Optional

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectOptions, APIConnectionError, stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .smallest_api import (
    default_model_for_mode,
    normalize_bool,
    normalize_mode,
    normalize_retry_count,
    transcribe_with_retries_sync,
)

logger = logging.getLogger(__name__)


class SmallestSTTSpeechStream(stt.SpeechStream):
    """RecognizeStream that buffers audio and runs Smallest AI STT on flush."""

    def __init__(
        self,
        *,
        stt_instance: "SmallestSTT",
        conn_options: APIConnectOptions,
        sample_rate: Optional[int] = None,
        language: str = "en",
        mode: str = "pulse",
        model: str = "lightning",
        max_empty_retries: int = 2,
        legacy_fallback: bool = True,
        api_key: str = "",
    ) -> None:
        super().__init__(
            stt=stt_instance,
            conn_options=conn_options,
            sample_rate=sample_rate if sample_rate is not None else NOT_GIVEN,
        )
        self._stt_instance = stt_instance
        self._language = language
        self._mode = mode
        self._model = model
        self._max_empty_retries = max_empty_retries
        self._legacy_fallback = legacy_fallback
        self._api_key = api_key

    async def _run(self) -> None:
        def is_flush(item):
            return type(item).__name__ == "_FlushSentinel"

        buffer: list[bytes] = []
        sr = 0
        ch = 1
        loop = asyncio.get_event_loop()
        try:
            async for item in self._input_ch:
                if is_flush(item):
                    if not buffer or sr <= 0:
                        continue
                    pcm = b"".join(buffer)
                    buffer.clear()
                    request_id = utils.shortuuid()
                    try:
                        text = await loop.run_in_executor(
                            None,
                            _transcribe_sync,
                            self._api_key,
                            self._language,
                            self._mode,
                            self._model,
                            self._max_empty_retries,
                            self._legacy_fallback,
                            pcm,
                            sr,
                            ch,
                        )
                    except Exception as exc:
                        logger.exception("Smallest AI STT failed: %s", exc)
                        raise APIConnectionError() from exc
                    duration_sec = len(pcm) / (2 * sr) if sr else 0
                    if not text.strip():
                        err = ValueError(
                            "Smallest STT returned empty transcript after retries; user should repeat."
                        )
                        self._stt_instance._emit_error(err, recoverable=True)  # type: ignore[attr-defined]
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=request_id,
                            alternatives=[stt.SpeechData(language=self._language, text=text.strip() or "")],
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=request_id,
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                else:
                    assert isinstance(item, rtc.AudioFrame)
                    buffer.append(bytes(item.data))
                    if sr <= 0:
                        sr = item.sample_rate
                        ch = item.num_channels
        except Exception as exc:
            logger.exception("Smallest AI STT stream error: %s", exc)
            raise


class SmallestSTT(stt.STT):
    """STT using Smallest AI docs endpoints (Pulse / Lightning), with optional legacy fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        mode: str = "pulse",
        model: Optional[str] = None,
        max_empty_retries: int = 2,
        legacy_fallback: bool = True,
    ) -> None:
        # Use non-streaming mode for stability with AgentSession turn handling.
        # We still expose recognize() and let LiveKit segment audio with VAD.
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._language = language
        self._mode = normalize_mode(mode)
        self._model = (model or default_model_for_mode(self._mode)).strip()
        self._max_empty_retries = normalize_retry_count(max_empty_retries, default=2)
        self._legacy_fallback = normalize_bool(legacy_fallback, default=True)
        resolved_key = (api_key or os.getenv("SMALLEST_API_KEY", "")).strip().strip('"').strip("'")
        if not resolved_key:
            raise ValueError(
                "SMALLEST_API_KEY is not set. "
                "Set it in .env or pass api_key= when using Smallest AI STT."
            )
        self._api_key = resolved_key

    @property
    def provider(self) -> str:
        return "smallestai"

    @property
    def model(self) -> str:
        return f"{self._mode}:{self._model}"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        from livekit.agents.utils.misc import is_given

        frames = buffer if isinstance(buffer, list) else [buffer]
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=utils.shortuuid(),
                alternatives=[stt.SpeechData(language=self._language, text="")],
                recognition_usage=stt.RecognitionUsage(audio_duration=0.0),
            )
        lang = self._language
        if is_given(language):
            lang = language or "en"
        sr = frames[0].sample_rate
        ch = frames[0].num_channels
        pcm = b"".join(bytes(f.data) for f in frames)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                _transcribe_sync,
                self._api_key,
                lang,
                self._mode,
                self._model,
                self._max_empty_retries,
                self._legacy_fallback,
                pcm,
                sr,
                ch,
            )
        except Exception as exc:
            logger.exception("Smallest AI STT recognize failed: %s", exc)
            raise APIConnectionError() from exc

        duration_sec = sum(f.duration for f in frames)
        if not text.strip():
            empty_err = ValueError(
                "Smallest STT returned empty transcript after retries; asking user to repeat."
            )
            self._emit_error(empty_err, recoverable=True)

        logger.info(
            "Smallest AI STT recognized %.2fs audio (chars=%d, language=%s, mode=%s, model=%s)",
            duration_sec,
            len(text.strip()),
            lang,
            self._mode,
            self._model,
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=lang, text=text.strip() or "")],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
        )

    def stream(
        self,
        *,
        language: Optional[str] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        sample_rate: Optional[int] = None,
    ) -> SmallestSTTSpeechStream:
        return SmallestSTTSpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            sample_rate=sample_rate,
            language=language or self._language,
            mode=self._mode,
            model=self._model,
            max_empty_retries=self._max_empty_retries,
            legacy_fallback=self._legacy_fallback,
            api_key=self._api_key,
        )


def _downmix_to_mono_int16(pcm_bytes: bytes, num_channels: int) -> bytes:
    """Convert interleaved int16 PCM to mono int16 when needed."""
    if num_channels <= 1:
        return pcm_bytes
    if not pcm_bytes:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    usable = (arr.size // num_channels) * num_channels
    if usable <= 0:
        return b""
    arr = arr[:usable].reshape(-1, num_channels)
    mono = arr.mean(axis=1).astype(np.int16)
    return mono.tobytes()


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM int16 bytes in a proper WAV container for the API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _transcribe_sync(
    api_key: str,
    language: str,
    mode: str,
    model: str,
    max_empty_retries: int,
    legacy_fallback: bool,
    pcm_bytes: bytes,
    sample_rate: int,
    num_channels: int,
) -> str:
    """Call Smallest AI STT synchronously; returns transcript text (or empty string)."""
    mono_pcm = _downmix_to_mono_int16(pcm_bytes, num_channels)
    wav_data = _pcm_to_wav_bytes(mono_pcm, sample_rate, num_channels=1)
    text, _ = transcribe_with_retries_sync(
        api_key=api_key,
        wav_bytes=wav_data,
        mode=mode,
        model=model,
        language=language,
        max_empty_retries=max_empty_retries,
        legacy_fallback=legacy_fallback,
    )
    return text


def create(cfg: dict) -> "SmallestSTT":
    """Factory for the provider registry: build SmallestSTT from a config `stt` section."""
    import os as _os

    return SmallestSTT(
        api_key=cfg.get("api_key") or _os.getenv("SMALLEST_API_KEY"),
        language=cfg.get("language") or "en",
        mode=cfg.get("mode") or "pulse",
        model=cfg.get("model") or cfg.get("stt_model"),
        max_empty_retries=normalize_retry_count(cfg.get("max_empty_retries"), 2),
        legacy_fallback=normalize_bool(cfg.get("legacy_fallback"), True),
    )
