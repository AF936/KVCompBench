"""Modulo-256 byte delta followed by standard Zstd level 3."""
import numpy as np
import zstandard


def compress(raw, metadata):
    values = np.frombuffer(raw, dtype=np.uint8)
    delta = values.copy()
    delta[1:] = values[1:] - values[:-1]
    return zstandard.ZstdCompressor(level=3).compress(delta.tobytes())


def decompress(packet, expected_bytes, metadata):
    delta = zstandard.ZstdDecompressor().decompress(packet, max_output_size=max(1, expected_bytes))
    if len(delta) != expected_bytes:
        raise ValueError('Delta-Zstd output length mismatch')
    return np.cumsum(np.frombuffer(delta, dtype=np.uint8), dtype=np.uint64).astype(np.uint8).tobytes()
