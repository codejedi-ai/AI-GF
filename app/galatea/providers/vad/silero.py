"""Silero VAD provider (wraps livekit-plugins-silero)."""
from __future__ import annotations

import logging

from livekit.plugins import silero

logger = logging.getLogger(__name__)


def create(cfg: dict):
    """
    Factory for the provider registry: load Silero VAD from a config `vad` section.
    The bundled silero_vad.onnx is used unless onnx_file_path (or url) points
    to a custom model file.
    """
    load_kwargs = {}
    onnx_file_path = cfg.get("onnx_file_path") or cfg.get("url")
    if onnx_file_path:
        load_kwargs["onnx_file_path"] = onnx_file_path
    if cfg.get("model"):
        logger.info("VAD config: model=%s", cfg["model"])
    return silero.VAD.load(**load_kwargs)
