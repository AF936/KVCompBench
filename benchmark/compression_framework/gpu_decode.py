"""Versioned CPU/GPU codec frames and GPU decode backend registry.

The frame is deliberately algorithm-neutral. Encoders run on CPU or GPU, then
write their payload with a registered backend ID. GPU decoding dispatches only
through that ID and rejects a mismatched format before launching a backend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from struct import Struct
from typing import Callable


MAGIC = b"KVG1"
FRAME_VERSION = 1
HEADER = Struct("<4sBBHI")  # magic, frame version, backend ID bytes, reserved, raw bytes


@dataclass(frozen=True)
class Backend:
    identifier: str
    description: str
    decoder: Callable[[bytes, int], bytes] | None = None
    cpu_encoder: Callable[[bytes, dict], bytes] | None = None


BACKENDS = {}
_PROVIDERS = {}
_loaded=False
def _load():
    global _loaded
    if _loaded:return
    _loaded=True
    from pathlib import Path
    import importlib.util,json,sys
    for manifest in sorted((Path(__file__).resolve().parents[1]/"compressors").glob("*/frame_provider.json")):
        folder=manifest.parent.resolve();path=(folder/json.loads(manifest.read_text())["file"]).resolve()
        if not path.is_relative_to(folder):raise ValueError("Provider escapes package")
        spec=importlib.util.spec_from_file_location("frame_provider_"+folder.name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        for value in module.list_backends():
            register_backend(value);_PROVIDERS[value.identifier]=module

def backend(identifier: str) -> Backend:
    _load()
    try:
        return BACKENDS[identifier]
    except KeyError as error:
        raise ValueError(f"Unknown GPU decode backend: {identifier}") from error


def list_backends():
    _load()
    return tuple(BACKENDS.values())


def register_backend(value: Backend) -> None:
    """Register a framework-owned custom GPU decoder before processing frames."""
    if value.identifier in BACKENDS:
        raise ValueError(f"GPU decode backend already exists: {value.identifier}")
    BACKENDS[value.identifier] = value


def encode_cpu_payload(raw: bytes, backend_id: str, encoder_options: dict | None = None) -> bytes:
    """Produce a payload that the declared GPU backend accepts."""
    selected = backend(backend_id)
    if selected.cpu_encoder is None:
        raise ValueError(f"{backend_id} has no registered CPU encoder")
    return selected.cpu_encoder(raw, dict(encoder_options or {}))


def pack_frame(backend_id: str, raw_bytes: int, payload: bytes) -> bytes:
    if raw_bytes < 0:
        raise ValueError("raw_bytes must be nonnegative")
    backend(backend_id)
    name = backend_id.encode("ascii")
    if len(name) > 255:
        raise ValueError("GPU backend ID is too long")
    return HEADER.pack(MAGIC, FRAME_VERSION, len(name), 0, raw_bytes) + name + payload


def unpack_frame(frame: bytes):
    if len(frame) < HEADER.size:
        raise ValueError("Truncated GPU codec frame")
    magic, version, name_size, reserved, raw_bytes = HEADER.unpack_from(frame)
    if magic != MAGIC or version != FRAME_VERSION or reserved:
        raise ValueError("Unsupported GPU codec frame")
    stop = HEADER.size + name_size
    if stop > len(frame):
        raise ValueError("Truncated GPU backend ID")
    try:
        backend_id = frame[HEADER.size:stop].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid GPU backend ID") from error
    backend(backend_id)
    return backend_id, raw_bytes, frame[stop:]


def decode_gpu_frame(frame,expected_bytes=None):
    identifier,size,payload=unpack_frame(frame)
    if expected_bytes is not None and size!=expected_bytes:raise ValueError("Frame length mismatch")
    if identifier in _PROVIDERS:return _PROVIDERS[identifier].decode_gpu_frame(frame,expected_bytes)
    raw=bytes(backend(identifier).decoder(payload,size))
    if len(raw)!=size:raise ValueError("Decoded length mismatch")
    return raw

def compress_cpu_to_gpu_frame(raw: bytes, backend_id: str, encoder_options: dict | None = None) -> bytes:
    return pack_frame(backend_id, len(raw), encode_cpu_payload(raw, backend_id, encoder_options))
