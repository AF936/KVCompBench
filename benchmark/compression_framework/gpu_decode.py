"""Versioned CPU/GPU codec frames and GPU decode backend registry.

The frame is deliberately algorithm-neutral. Encoders run on CPU or GPU, then
write their payload with a registered backend ID. GPU decoding dispatches only
through that ID and rejects a mismatched format before launching a backend.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from functools import lru_cache
from struct import Struct
from typing import Callable
import zlib

import lz4.block
import snappy
import zstandard

MAGIC = b"KVG1"
FRAME_VERSION = 1
HEADER = Struct("<4sBBHI")  # magic, frame version, backend ID bytes, reserved, raw bytes


@dataclass(frozen=True)
class Backend:
    identifier: str
    description: str
    nvcomp_algorithm: str | None = None
    decoder: Callable[[bytes, int], bytes] | None = None
    cpu_encoder: Callable[[bytes, dict], bytes] | None = None


def _zstd_encoder(raw: bytes, options: dict) -> bytes:
    return zstandard.ZstdCompressor(level=int(options.get("level", 3))).compress(raw)


def _raw_deflate_encoder(raw: bytes, options: dict) -> bytes:
    stream = zlib.compressobj(int(options.get("level", 6)), zlib.DEFLATED, -15)
    return stream.compress(raw) + stream.flush()


def _identity_gpu_decoder(payload: bytes, raw_bytes: int) -> bytes:
    """GPU-resident copy path used as the lossless no-compression baseline."""
    if len(payload) != raw_bytes:
        raise ValueError("Identity payload length mismatch")
    import torch
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to("cuda:0").cpu().numpy().tobytes()


BACKENDS = {
    "identity-gpu-copy": Backend("identity-gpu-copy", "GPU copy baseline without compression", decoder=_identity_gpu_decoder, cpu_encoder=lambda raw, options: bytes(raw)),
    "nvcomp-zstd-raw": Backend("nvcomp-zstd-raw", "standard Zstandard stream", nvcomp_algorithm="Zstd", cpu_encoder=_zstd_encoder),
    "nvcomp-snappy-raw": Backend("nvcomp-snappy-raw", "standard Snappy stream", nvcomp_algorithm="Snappy", cpu_encoder=lambda raw, options: snappy.compress(raw)),
    "nvcomp-gzip-raw": Backend("nvcomp-gzip-raw", "standard gzip stream", nvcomp_algorithm="Gzip", cpu_encoder=lambda raw, options: gzip.compress(raw, compresslevel=int(options.get("level", 9)))),
    "nvcomp-lz4-block-raw": Backend("nvcomp-lz4-block-raw", "raw LZ4 block; not lz4.frame", nvcomp_algorithm="LZ4", cpu_encoder=lambda raw, options: lz4.block.compress(raw, store_size=False)),
    "nvcomp-deflate-raw": Backend("nvcomp-deflate-raw", "raw Deflate; not zlib-wrapped", nvcomp_algorithm="Deflate", cpu_encoder=_raw_deflate_encoder),
}


def backend(identifier: str) -> Backend:
    try:
        return BACKENDS[identifier]
    except KeyError as error:
        raise ValueError(f"Unknown GPU decode backend: {identifier}") from error


def list_backends():
    return tuple(BACKENDS.values())


def register_backend(value: Backend) -> None:
    """Register a framework-owned custom GPU decoder before processing frames."""
    if value.identifier in BACKENDS:
        raise ValueError(f"GPU decode backend already exists: {value.identifier}")
    if bool(value.nvcomp_algorithm) == bool(value.decoder):
        raise ValueError("A backend requires exactly one decoder implementation")
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


@lru_cache(maxsize=None)
def _nvcomp_codec(backend_id: str):
    from nvidia import nvcomp
    algorithm = backend(backend_id).nvcomp_algorithm
    if algorithm is None:
        raise ValueError(f"{backend_id} is not an nvCOMP backend")
    return nvcomp.Codec(algorithm=algorithm, bitstream_kind=nvcomp.BitstreamKind.RAW)


def decode_gpu_frame(frame: bytes, expected_bytes: int | None = None) -> bytes:
    """Decode one framed block on the assigned GPU and validate its length."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to an assigned GPU")
    backend_id, raw_bytes, payload = unpack_frame(frame)
    if expected_bytes is not None and expected_bytes != raw_bytes:
        raise ValueError("Expected byte count conflicts with GPU frame")
    from nvidia import nvcomp
    selected = backend(backend_id)
    if selected.decoder:
        raw = bytes(selected.decoder(payload, raw_bytes))
    else:
        raw = bytes(_nvcomp_codec(backend_id).decode(nvcomp.as_array(payload)).cpu())
    if len(raw) != raw_bytes:
        raise ValueError("GPU decoder returned an unexpected length")
    return raw


def compress_cpu_to_gpu_frame(raw: bytes, backend_id: str, encoder_options: dict | None = None) -> bytes:
    return pack_frame(backend_id, len(raw), encode_cpu_payload(raw, backend_id, encoder_options))
