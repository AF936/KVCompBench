"""OpenZL default Serial/Zstd graph with shared nvCOMP GPU decoding."""
from pathlib import Path
import importlib.util
import sys
folder = Path(__file__).resolve().parent
sys.path.insert(0, str(folder))
import xxhash
from openzl_gpu_parser import parse
from openzl_gpu_zstd import standard_zstd_payload
from gpu_decode import decode_gpu_frame, pack_frame

spec = importlib.util.spec_from_file_location('kv_openzl_cpu_api', folder / 'api.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def cpu_encode(raw):
    return api.compress(raw, {})


def gpu_decode(payload, expected_bytes):
    flags, checksum, pieces = parse(payload, expected_bytes, xxhash.xxh3_64_intdigest)
    import zstandard
    result = []
    total = 0
    for encoded, checks, stored in pieces:
        stream = encoded if stored else standard_zstd_payload(encoded, 27)
        size = len(stream) if stored else zstandard.frame_content_size(stream)
        if size < 0 or total + size > expected_bytes:
            raise ValueError('OpenZL Zstd content size mismatch')
        if stored and size == 0:
            raw = b''
        else:
            raw = decode_gpu_frame(pack_frame('identity-gpu-copy' if stored else 'nvcomp-zstd-raw', size, stream), size)
        if checks[0] is not None and xxhash.xxh3_64_intdigest(raw) & 0xFFFFFFFF != checks[0]:
            raise ValueError('OpenZL content checksum mismatch')
        result.append(raw)
        total += len(raw)
    if total != expected_bytes:
        raise ValueError('OpenZL decoded size mismatch')
    return b''.join(result)
