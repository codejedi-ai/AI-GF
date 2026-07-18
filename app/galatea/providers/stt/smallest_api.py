"""
Smallest AI STT HTTP transport helpers.

Request builders and transcript extraction for Smallest AI Pulse/Lightning STT.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SMALLEST_API_BASE = "https://waves-api.smallest.ai/api/v1"
PULSE_STT_URL = f"{SMALLEST_API_BASE}/pulse/get_text"
LIGHTNING_STT_URL = f"{SMALLEST_API_BASE}/lightning/get_text"
LEGACY_STT_URL = f"{SMALLEST_API_BASE}/speech-to-text"

DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=60.0, write=10.0, pool=10.0)


def normalize_mode(mode: str | None) -> str:
    value = (mode or "pulse").strip().lower()
    if value not in ("pulse", "lightning"):
        raise ValueError(f"Unsupported smallest STT mode: {mode!r}")
    return value


def default_model_for_mode(mode: str) -> str:
    if mode == "lightning":
        return "lightning-v2"
    return "lightning"


def normalize_retry_count(value: Any, default: int = 2) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return max(int(stripped), 0)
        except ValueError:
            return default
    return default


def normalize_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("1", "true", "yes", "on", "y", "t"):
            return True
        if raw in ("0", "false", "no", "off", "n", "f"):
            return False
    return default


def build_docs_request(mode: str, model: str, language: str) -> tuple[str, dict[str, str]]:
    normalized_mode = normalize_mode(mode)
    params = {"file_format": "wav"}
    if language:
        params["language"] = language
    if normalized_mode == "pulse":
        return PULSE_STT_URL, params
    params["model"] = model
    return LIGHTNING_STT_URL, params


def _post_wav_request(
    api_key: str,
    wav_bytes: bytes,
    url: str,
    params: dict[str, str],
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav",
    }
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(url, headers=headers, params=params, content=wav_bytes)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text if exc.response is not None else ""
        raise ValueError(
            f"Smallest STT request rejected ({exc.response.status_code if exc.response else 'unknown'}): {body}"
        ) from exc
    return response.json()


def _post_legacy_request(
    api_key: str,
    wav_bytes: bytes,
    language: str,
    model: str,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": model,
        "language": language,
        "word_timestamps": "false",
        "age_detection": "false",
        "gender_detection": "false",
        "emotion_detection": "false",
    }
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(LEGACY_STT_URL, headers=headers, files=files, data=data)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text if exc.response is not None else ""
        raise ValueError(
            f"Smallest legacy STT request rejected ({exc.response.status_code if exc.response else 'unknown'}): {body}"
        ) from exc
    return response.json()


def extract_transcript(payload: Any) -> str:
    candidates: list[str] = []
    target_keys = {
        "transcript",
        "transcription",
        "text",
        "normalized_text",
        "utterance",
        "sentence",
        "content",
    }

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k).lower()
                if key in target_keys:
                    if isinstance(v, str):
                        text = v.strip()
                        if text:
                            candidates.append(text)
                    elif isinstance(v, list):
                        pieces = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
                        joined = " ".join(piece for piece in pieces if piece)
                        if joined:
                            candidates.append(joined)
                walk(v, key)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key)
        elif isinstance(node, str):
            if parent_key in target_keys:
                text = node.strip()
                if text:
                    candidates.append(text)

    walk(payload)
    if not candidates:
        return ""
    return max(candidates, key=len)


def transcribe_with_retries_sync(
    api_key: str,
    wav_bytes: bytes,
    mode: str,
    model: str,
    language: str,
    max_empty_retries: int,
    legacy_fallback: bool,
) -> tuple[str, dict[str, Any]]:
    last_payload: dict[str, Any] = {}

    languages = []
    primary_lang = (language or "en").strip() or "en"
    languages.append(primary_lang)
    if primary_lang.lower() != "multi":
        languages.append("multi")

    for lang in languages:
        for _ in range(max_empty_retries + 1):
            url, params = build_docs_request(mode, model, lang)
            payload = _post_wav_request(
                api_key=api_key,
                wav_bytes=wav_bytes,
                url=url,
                params=params,
            )
            last_payload = payload
            text = extract_transcript(payload)
            if text:
                return text, payload

    if legacy_fallback:
        for lang in languages:
            payload = _post_legacy_request(
                api_key=api_key,
                wav_bytes=wav_bytes,
                language=lang,
                model=model,
            )
            last_payload = payload
            text = extract_transcript(payload)
            if text:
                return text, payload

    if last_payload:
        logger.warning("Smallest STT returned empty transcript. Response: %s", str(last_payload)[:800])
    else:
        logger.warning("Smallest STT returned empty transcript without a response payload.")
    return "", last_payload
